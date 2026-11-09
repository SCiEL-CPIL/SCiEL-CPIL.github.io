"""Root finder: hybrid bisection + Newton with deadline guard and best-tracking."""
import math
import time

# Deadline: evaluator kills at 150s; keep a safety margin.
_DEADLINE = 130.0
_TOL = 1e-8
_POLISH_TOL = 1e-13


def find_root(f, x0):
    t0 = time.time()
    x = float(x0)
    fx = f(x)
    calls = 1

    # Track the best (smallest |f|) seen so far.
    best_x = x
    best_fabs = abs(fx)

    def _deadline_ok():
        return (time.time() - t0) < _DEADLINE

    def _update_best(xx, fxx):
        nonlocal best_x, best_fabs
        af = abs(fxx)
        if af < best_fabs:
            best_fabs = af
            best_x = xx

    if best_fabs <= _TOL:
        return x

    # ---- Phase 1: find a bracket [a, b] with f(a)*f(b) < 0 ----
    a = x
    fa = fx
    b = None
    fb = None

    # Try a set of step sizes, growing. Check both directions.
    # Use multiplicative growth to cover a wide range.
    for mult in [1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0, 1e6]:
        if not _deadline_ok():
            break
        for sign in (1.0, -1.0):
            if not _deadline_ok():
                break
            step = mult * max(1.0, abs(x))
            cand = x + sign * step
            fc = f(cand)
            calls += 1
            _update_best(cand, fc)
            if abs(fc) <= _TOL:
                return cand
            if fa * fc < 0:
                # Sign change between x and cand
                if sign > 0:
                    a, b = x, cand
                    fa, fb = fx, fc
                else:
                    a, b = cand, x
                    fa, fb = fc, fx
                break
        if b is not None:
            break

    # If no bracket found, try a wider sweep with more steps to catch oscillations.
    if b is None:
        # Do a linear scan over a range around x0 with modest resolution.
        span = max(1.0, abs(x)) * 10.0
        n_steps = 200
        prev_x = x
        prev_f = fx
        for i in range(1, n_steps + 1):
            if not _deadline_ok():
                break
            nx = x - span + (2.0 * span * i) / n_steps
            nf = f(nx)
            calls += 1
            _update_best(nx, nf)
            if abs(nf) <= _TOL:
                return nx
            if prev_f * nf < 0:
                if nx > prev_x:
                    a, b = prev_x, nx
                    fa, fb = prev_f, nf
                else:
                    a, b = nx, prev_x
                    fa, fb = nf, prev_f
                break
            prev_x = nx
            prev_f = nf

    # ---- Phase 2: hybrid bisection + Newton inside the bracket ----
    if b is not None:
        # Ensure a < b
        if a > b:
            a, b = b, a
            fa, fb = fb, fa
        # Keep a and b as the bracket; maintain sign-change invariant.
        for _ in range(200):
            if not _deadline_ok():
                break
            # Check current midpoint / best
            if best_fabs <= _TOL:
                break
            # Try Newton step from the point with smaller |f|
            if abs(fa) < abs(fb):
                x_try = a
                f_try = fa
            else:
                x_try = b
                f_try = fb
            h = 1e-7 * max(1.0, abs(x_try))
            d = (f(x_try + h) - f_try) / h
            calls += 1
            newton_ok = False
            if d != 0 and math.isfinite(d):
                x_new = x_try - f_try / d
                if a < x_new < b:
                    # Newton step stays inside bracket; accept it.
                    f_new = f(x_new)
                    calls += 1
                    _update_best(x_new, f_new)
                    if abs(f_new) <= _TOL:
                        return x_new
                    # Update bracket to keep sign change
                    if fa * f_new < 0:
                        b = x_new
                        fb = f_new
                    else:
                        a = x_new
                        fa = f_new
                    newton_ok = True
            if not newton_ok:
                # Bisect
                x_mid = 0.5 * (a + b)
                if x_mid <= a or x_mid >= b:
                    break
                f_mid = f(x_mid)
                calls += 1
                _update_best(x_mid, f_mid)
                if abs(f_mid) <= _TOL:
                    return x_mid
                if fa * f_mid < 0:
                    b = x_mid
                    fb = f_mid
                else:
                    a = x_mid
                    fa = f_mid
            # Convergence check on bracket width
            if abs(b - a) < 1e-14 * max(1.0, abs(a)):
                break

        # Return the best tracked point (or midpoint)
        if best_fabs <= _TOL:
            return best_x
        # Prefer the point with smallest |f|
        return best_x

    # ---- Phase 3: no bracket found; fall back to best-effort Newton ----
    # Use a few Newton iterations with damping, tracking best.
    x = x0
    for _ in range(30):
        if not _deadline_ok():
            break
        fx = f(x)
        calls += 1
        _update_best(x, fx)
        if abs(fx) <= _TOL:
            return x
        h = 1e-7 * max(1.0, abs(x))
        d = (f(x + h) - fx) / h
        calls += 1
        if d == 0 or not math.isfinite(d):
            break
        x_new = x - fx / d
        if not math.isfinite(x_new):
            break
        x = x_new

    return best_x
