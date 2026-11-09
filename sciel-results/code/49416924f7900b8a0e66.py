"""Root finder: hybrid bracketing + Newton with adaptive step control.

Strategy:
  1. Evaluate f(x0). If |f(x0)| <= 1e-8, return immediately.
  2. Find a bracket [a, b] with sign change via geometric expansion from x0.
  3. Use a hybrid method: Newton's method with safe step (damped if needed)
     combined with bisection to guarantee convergence within the bracket.
  4. Track best x (smallest |f(x)|) throughout; return it if budget exhausted.
  5. If no bracket found, fall back to pure Newton with damping.

This approach minimizes function calls by using Newton's fast quadratic
convergence when safe, and falling back to bisection for robustness.
"""
import time
import math

_TOL = 1e-8
_CALL_CAP = 495
_DEADLINE_S = 130.0


def find_root(f, x0):
    x0 = float(x0)
    t_start = time.monotonic()
    calls = 0

    def _time_ok():
        return (time.monotonic() - t_start) < _DEADLINE_S

    def _fx(x):
        nonlocal calls
        if calls >= _CALL_CAP:
            return float('nan')
        calls += 1
        try:
            v = float(f(x))
            if math.isnan(v) or math.isinf(v):
                return float('nan')
            return v
        except Exception:
            return float('nan')

    # Phase 0: evaluate at x0
    f0 = _fx(x0)
    if f0 != f0:
        return x0
    if abs(f0) <= _TOL:
        return x0

    best_x = x0
    best_abs = abs(f0)

    # Phase 1: find a bracket [a, b] with f(a)*f(b) < 0
    # Use geometric expansion from x0 in both directions
    a, b, fa, fb = None, None, None, None
    found_bracket = False

    # Try expanding outward from x0 with geometric steps
    base_scale = max(1.0, abs(x0)) * 0.1
    for m in [0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
        if not _time_ok() or calls >= _CALL_CAP:
            break
        step = m * base_scale
        for sign in (1, -1):
            if not _time_ok() or calls >= _CALL_CAP:
                break
            x_try = x0 + sign * step
            f_try = _fx(x_try)
            if f_try != f_try:
                continue
            if abs(f_try) < best_abs:
                best_abs = abs(f_try)
                best_x = x_try
            if abs(f_try) <= _TOL:
                return x_try
            if f0 * f_try < 0.0:
                a, fa = x0, f0
                b, fb = x_try, f_try
                found_bracket = True
                break
        if found_bracket:
            break

    # If no bracket with x0 as endpoint, try scanning for any sign change
    if not found_bracket:
        # Try a coarse grid scan around x0
        base = max(1.0, abs(x0))
        for p in [0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5000, 10000]:
            if not _time_ok() or calls >= _CALL_CAP:
                break
            for sign in (1, -1):
                if not _time_ok() or calls >= _CALL_CAP:
                    break
                x_try = x0 + sign * p * base
                f_try = _fx(x_try)
                if f_try != f_try:
                    continue
                if abs(f_try) < best_abs:
                    best_abs = abs(f_try)
                    best_x = x_try
                if abs(f_try) <= _TOL:
                    return x_try
                if f0 * f_try < 0.0:
                    a, fa = x0, f0
                    b, fb = x_try, f_try
                    found_bracket = True
                    break
            if found_bracket:
                break

    # If still no bracket, try symmetric points
    if not found_bracket:
        for p in [0.01, 0.1, 1, 10, 100]:
            if not _time_ok() or calls >= _CALL_CAP:
                break
            x1 = x0 - p * max(1.0, abs(x0))
            x2 = x0 + p * max(1.0, abs(x0))
            f1 = _fx(x1)
            f2 = _fx(x2)
            if f1 != f1 or f2 != f2:
                continue
            for x_try, f_try in [(x1, f1), (x2, f2)]:
                if abs(f_try) < best_abs:
                    best_abs = abs(f_try)
                    best_x = x_try
                if abs(f_try) <= _TOL:
                    return x_try
            if f1 != f1 or f2 != f2:
                continue
            if f1 * f2 < 0:
                a, fa = x1, f1
                b, fb = x2, f2
                found_bracket = True
                break
            if f0 * f1 < 0:
                a, fa = x0, f0
                b, fb = x1, f1
                found_bracket = True
                break
            if f0 * f2 < 0:
                a, fa = x0, f0
                b, fb = x2, f2
                found_bracket = True
                break
        if found_bracket:
            pass

    if not found_bracket:
        # No bracket found; use damped Newton from x0
        x = x0
        h = 1e-7 * max(1.0, abs(x))
        for _ in range(50):
            if not _time_ok() or calls >= _CALL_CAP:
                break
            fv = _fx(x)
            if fv != fv:
                break
            if abs(fv) < best_abs:
                best_abs = abs(fv)
                best_x = x
            if abs(fv) <= _TOL:
                return x
            # Numerical derivative
            fp = _fx(x + h)
            if fp != fp:
                break
            d = (fp - fv) / h
            if d == 0.0 or d != d:
                break
            nx = x - fv / d
            if nx != nx or math.isinf(nx):
                break
            # Damp the step if it's too large
            step = nx - x
            max_step = 2.0 * max(1.0, abs(x))
            if abs(step) > max_step:
                nx = x + math.copysign(max_step, step)
            x = nx
        return best_x

    # Phase 2: Hybrid Newton + bisection on bracket [a, b]
    # Ensure |fa| <= |fb|
    if abs(fa) > abs(fb):
        a, b = b, a
        fa, fb = fb, fa

    h = 1e-7 * max(1.0, abs(a), abs(b))

    for _ in range(100):
        if not _time_ok() or calls >= _CALL_CAP:
            break

        # Check convergence
        if abs(fa) <= _TOL:
            return a
        if abs(fb) <= _TOL:
            return b

        # Try Newton step from the point with smaller |f|
        x_newton = a if abs(fa) <= abs(fb) else b
        f_newton = fa if abs(fa) <= abs(fb) else fb
        x_other = b if x_newton == a else a
        f_other = fb if x_newton == a else fa

        # Compute derivative at x_newton
        fp = _fx(x_newton + h)
        if fp != fp:
            # Fall back to bisection
            mid = 0.5 * (a + b)
            fm = _fx(mid)
            if fm != fm:
                break
            if abs(fm) < best_abs:
                best_abs = abs(fm)
                best_x = mid
            if abs(fm) <= _TOL:
                return mid
            if fa * fm < 0:
                b, fb = mid, fm
            else:
                a, fa = mid, fm
            continue

        d = (fp - f_newton) / h
        if d == 0.0 or d != d:
            # Fall back to bisection
            mid = 0.5 * (a + b)
            fm = _fx(mid)
            if fm != fm:
                break
            if abs(fm) < best_abs:
                best_abs = abs(fm)
                best_x = mid
            if abs(fm) <= _TOL:
                return mid
            if fa * fm < 0:
                b, fb = mid, fm
            else:
                a, fa = mid, fm
            continue

        # Newton step
        x_n = x_newton - f_newton / d
        if x_n != x_n or math.isinf(x_n):
            # Fall back to bisection
            mid = 0.5 * (a + b)
            fm = _fx(mid)
            if fm != fm:
                break
            if abs(fm) < best_abs:
                best_abs = abs(fm)
                best_x = mid
            if abs(fm) <= _TOL:
                return mid
            if fa * fm < 0:
                b, fb = mid, fm
            else:
                a, fa = mid, fm
            continue

        # Check if Newton step is within bracket
        if min(a, b) < x_n < max(a, b):
            f_n = _fx(x_n)
            if f_n != f_n:
                # Fall back to bisection
                mid = 0.5 * (a + b)
                fm = _fx(mid)
                if fm != fm:
                    break
                if abs(fm) < best_abs:
                    best_abs = abs(fm)
                    best_x = mid
                if abs(fm) <= _TOL:
                    return mid
                if fa * fm < 0:
                    b, fb = mid, fm
                else:
                    a, fa = mid, fm
                continue
            if abs(f_n) < best_abs:
                best_abs = abs(f_n)
                best_x = x_n
            if abs(f_n) <= _TOL:
                return x_n
            # Update bracket
            if fa * f_n < 0:
                b, fb = x_n, f_n
            else:
                a, fa = x_n, f_n
        else:
            # Newton step outside bracket, use bisection
            mid = 0.5 * (a + b)
            fm = _fx(mid)
            if fm != fm:
                break
            if abs(fm) < best_abs:
                best_abs = abs(fm)
                best_x = mid
            if abs(fm) <= _TOL:
                return mid
            if fa * fm < 0:
                b, fb = mid, fm
            else:
                a, fa = mid, fm

        # Ensure |fa| <= |fb| for next iteration
        if abs(fa) > abs(fb):
            a, b = b, a
            fa, fb = fb, fa

    # Return best point
    if abs(fa) <= _TOL:
        return a
    if abs(fb) <= _TOL:
        return b
    return best_x
