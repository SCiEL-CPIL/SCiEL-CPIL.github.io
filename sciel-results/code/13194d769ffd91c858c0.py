"""Root finder: Brent's method (bisection + secant + inverse quadratic interpolation)
with smart bracket expansion, for smooth polynomial equations.

Strategy:
  1. Evaluate f(x0). If |f(x0)| <= 1e-8, return immediately.
  2. Expand a bracket around x0 until a sign change is found.
  3. Apply Brent's method to converge to the root with minimal function calls.
  4. Track the best x (smallest |f(x)|) throughout; return it if we run out of budget.
"""
import time
import math

_TOL = 1e-8
_CALL_CAP = 490
_DEADLINE_S = 140.0


def find_root(f, x0):
    x0 = float(x0)
    t_start = time.monotonic()
    deadline = t_start + _DEADLINE_S
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
            return v
        except Exception:
            return float('nan')

    # Phase 0: evaluate at x0
    f0 = _fx(x0)
    if f0 != f0:  # nan
        return x0
    if abs(f0) <= _TOL:
        return x0

    best_x = x0
    best_abs = abs(f0)

    # Phase 1: find a bracket [a, b] with f(a)*f(b) < 0
    a, b, fa, fb = None, None, None, None
    # Try expanding outward from x0
    scale = max(1.0, abs(x0)) * 0.5
    # Try a set of multipliers to find a sign change
    multipliers = [
        0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0
    ]
    found_bracket = False
    for m in multipliers:
        if not _time_ok() or calls >= _CALL_CAP:
            break
        # Try x0 + m*scale and x0 - m*scale
        for sign in (1, -1):
            if not _time_ok() or calls >= _CALL_CAP:
                break
            x_try = x0 + sign * m * scale
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
        # Also try the other side
        for sign in (-1, 1):
            if not _time_ok() or calls >= _CALL_CAP:
                break
            x_try = x0 + sign * m * scale
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

    if not found_bracket:
        # Try a coarser scan
        base = max(1.0, abs(x0))
        for p in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000):
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
        if not found_bracket:
            # Last resort: try around x0 with small steps
            for p in (1e-2, 5e-2, 0.25, 1.0, 4.0, 16.0, 64.0, 256.0, 1024.0, 4096.0):
                if not _time_ok() or calls >= _CALL_CAP:
                    break
                for sign in (1, -1):
                    if not _time_ok() or calls >= _CALL_CAP:
                        break
                    x_try = x0 + sign * p * max(1.0, abs(x0))
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

    if not found_bracket:
        # No bracket found; try Newton's method from x0
        x = x0
        for _ in range(30):
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
            h = 1e-6 * max(1.0, abs(x))
            fp = _fx(x + h)
            fm = _fx(x - h)
            if fp != fp or fm != fm:
                break
            d = (fp - fm) / (2.0 * h)
            if d == 0.0 or d != d:
                break
            nx = x - fv / d
            if nx != nx:
                break
            step = nx - x
            if abs(step) > 0.5 * max(1.0, abs(x)):
                nx = x + 0.5 * step
            x = nx
        return best_x

    # Phase 2: Brent's method
    # Ensure |fa| <= |fb|
    if abs(fa) > abs(fb):
        a, b = b, a
        fa, fb = fb, fa

    c, fc = a, fa
    d = b - a
    e = d
    mflag = True

    for _ in range(200):
        if not _time_ok() or calls >= _CALL_CAP:
            break

        # Check convergence
        if abs(fa) <= _TOL:
            return a
        if abs(fb) <= _TOL:
            return b

        # Invariant: fa != fb, b and c are on the same side of a
        if fa * fb > 0:
            b, a = a, b
            fb, fa = fa, fb
            c, b = b, c
            fc, fb = fb, fc

        if abs(fc) < abs(fb):
            b, a = a, b
            fb, fa = fa, fb
            c, b = b, c
            fc, fb = fb, fc

        d = b - c
        e = c - a
        mflag = True

        # Choose between bisection, secant, and inverse quadratic interpolation
        if abs(d) > 1e-15 * max(1.0, abs(b)):
            # Try inverse quadratic interpolation
            r = (fc - fb) / (b - c)
            s = (fa - fb) / (a - c)
            t = (fa - fc) / (a - b)
            p = fa * t * s
            q = r - s
            if p > 0:
                q = -q
            p = abs(p)
            # Check if IQI is acceptable
            if 2.0 * p < min(3.0 * e * abs(q) - p * (q - t), abs(e) * abs(fc)):
                mflag = False
                new_c = c
            else:
                # Secant method
                new_c = b - e * (fa * (fb - fc)) / (fa - fc) if abs(fa - fc) > 1e-300 else b - e / 2.0
                mflag = False
                # Actually let me redo this properly
                # Secant: c_new = b - e * fa*(fb-fc) / (fa-fc)  -- wait, this is wrong
                # Let me use the standard Brent formulation
                # I'll restart with a cleaner implementation
                break
        else:
            # d is tiny, use bisection
            new_c = c
            mflag = True

        # This is getting complicated; let me just use a clean bisection + secant hybrid
        break

    # Clean Brent's method implementation
    # Reset
    a, b = a, b
    fa, fb = fa, fb
    # Ensure |fa| <= |fb|
    if abs(fa) > abs(fb):
        a, b = b, a
        fa, fb = fb, fa

    c, fc = a, fa
    e_prev = b - a
    mflag = True

    for _ in range(300):
        if not _time_ok() or calls >= _CALL_CAP:
            break

        if abs(fa) <= _TOL:
            return a
        if abs(fb) <= _TOL:
            return b

        # Maintain invariant: |fa| <= |fb| and b, c on same side of a
        if fa * fb > 0:
            b, a = a, b
            fb, fa = fa, fb
            c, b = b, c
            fc, fb = fb, fc

        if abs(fc) < abs(fb):
            b, a = a, b
            fb, fa = fa, fb
            c, b = b, c
            fc, fb = fb, fc

        d = b - c
        e_old = c - a
        mflag = True

        if abs(d) > 1e-15 * max(1.0, abs(b)):
            # Try inverse quadratic interpolation
            r1 = (fc - fb) / d
            s1 = (fa - fb) / (a - c)
            t1 = (fa - fc) / (a - b)
            p_num = fa * t1 * s1
            q_den = r1 - s1
            if p_num > 0:
                q_den = -q_den
            p_num = abs(p_num)
            # Accept IQI if:
            # 2*p < min(3*|e_old|*|q| - p*(q-t), |e_old|*|fc|)
            min_val = min(3.0 * abs(e_old) * abs(q_den) - p_num * (q_den - t1),
                          abs(e_old) * abs(fc))
            if 2.0 * p_num < min_val and p_num < abs(e_old) * abs(fc):
                mflag = False
                new_c = c
            else:
                # Secant
                mflag = False
                denom = fa - fc
                if abs(denom) > 1e-300:
                    new_c = b - e_old * fa * (fb - fc) / denom
                else:
                    new_c = b - e_old / 2.0
                # Wait, secant formula: x_new = b - f(b)*(b-c)/(f(b)-f(c))
                # That's: b - fb*(b-c)/(fb-fc)
                # Let me use the correct secant step
                denom2 = fb - fc
                if abs(denom2) > 1e-300:
                    new_c = b - fb * (b - c) / denom2
                else:
                    new_c = b - e_old / 2.0
                    mflag = True
        else:
            # Bisection
            new_c = c
            mflag = True

        # Check if new_c is too close to a or b
        if new_c <= min(a, b) + 1e-15 * max(1.0, abs(a), abs(b)) or \
           new_c >= max(a, b) - 1e-15 * max(1.0, abs(a), abs(b)):
            new_c = 0.5 * (a + b)
            mflag = True

        # Evaluate f at new_c
        fc_new = _fx(new_c)
        if fc_new != fc_new:
            # Bad evaluation, use bisection
            new_c = 0.5 * (a + b)
            fc_new = _fx(new_c)
            if fc_new != fc_new:
                break

        if abs(fc_new) < best_abs:
            best_abs = abs(fc_new)
            best_x = new_c
        if abs(fc_new) <= _TOL:
            return new_c

        # Update bracket
        if fa * fc_new < 0:
            b = new_c
            fb = fc_new
        else:
            if abs(fc_new) < abs(fb) or not mflag:
                c = new_c
                fc = fc_new
            else:
                a = new_c
                fa = fc_new
            e_prev = b - c

    # Return best point
    if abs(fa) <= _TOL:
        return a
    if abs(fb) <= _TOL:
        return b
    return best_x
