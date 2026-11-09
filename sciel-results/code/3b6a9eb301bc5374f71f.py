"""Root finding for ill-conditioned equations: robust adaptive hybrid."""
import time
import math
import numpy as np
from scipy.optimize import brentq

# Global deadline: leave a safety margin before the 150s kill.
_DEADLINE_SECONDS = 140.0

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8, using at most 500 evaluations of f.
    Returns the best x found (smallest |f(x)|) within the budget.
    """
    start_time = time.time()
    eval_count = 0
    max_evals = 500
    tol = 1e-8

    # State to track the best point seen so far
    best_x = float(x0)
    best_fx = f(best_x)
    eval_count += 1
    best_abs_fx = abs(best_fx)

    def _time_left():
        return _DEADLINE_SECONDS - (time.time() - start_time)

    def _eval(x):
        """Evaluate f(x) with eval-count and time guards. Returns (fx, ok)."""
        nonlocal eval_count, best_x, best_fx, best_abs_fx
        if eval_count >= max_evals or _time_left() <= 0.5:
            return None, False
        try:
            fx = float(f(x))
        except Exception:
            return None, False
        eval_count += 1
        if not math.isfinite(fx):
            return None, False
        # Update best
        if abs(fx) < best_abs_fx:
            best_x = x
            best_fx = fx
            best_abs_fx = abs(fx)
        return fx, True

    def _check_done():
        return best_abs_fx <= tol

    # If already good enough
    if _check_done():
        return best_x

    # ------------------------------------------------------------------
    # Phase 1: Safeguarded Newton with central-difference derivative
    # ------------------------------------------------------------------
    x = float(x0)
    fx, _ = _eval(x)
    if fx is None:
        return best_x

    prev_x = x
    prev_fx = fx

    def _fd_deriv(x, fx):
        """Compute derivative using central difference if budget allows, else forward."""
        nonlocal eval_count
        h = 1e-6 * max(1.0, abs(x))
        # Try central difference: 2 evals
        if eval_count + 2 <= max_evals and _time_left() > 1.0:
            fxp, ok1 = _eval(x + h)
            if not ok1:
                return None
            fxm, ok2 = _eval(x - h)
            if not ok2:
                return None
            return (fxp - fxm) / (2.0 * h)
        # Fallback: forward difference: 1 eval
        fxp, ok = _eval(x + h)
        if not ok:
            return None
        return (fxp - fx) / h

    newton_iters = 0
    max_newton = 40
    while newton_iters < max_newton and eval_count < max_evals and _time_left() > 1.0:
        if _check_done():
            return best_x

        # Try Newton step with central-difference derivative
        d = _fd_deriv(x, fx)
        if d is None or abs(d) < 1e-15:
            # Fallback to secant if we have a previous point
            if newton_iters > 0 and abs(x - prev_x) > 1e-15:
                d = (fx - prev_fx) / (x - prev_x)
            if d is None or abs(d) < 1e-15:
                break

        step = fx / d
        x_new = x - step

        # Safeguard: limit step size to avoid runaway
        max_step = 10.0 * max(1.0, abs(x))
        if abs(step) > max_step:
            x_new = x - math.copysign(max_step, step)

        fx_new, ok = _eval(x_new)
        if not ok:
            break

        # Check for convergence
        if abs(fx_new) <= tol:
            return x_new

        # If the full step didn't improve, try a damped step
        if abs(fx_new) >= abs(fx) * 0.9:
            # Damped Newton: half step
            x_damp = x - 0.5 * (fx / d)
            max_step_d = 5.0 * max(1.0, abs(x))
            damp_step = 0.5 * fx / d
            if abs(damp_step) > max_step_d:
                x_damp = x - math.copysign(max_step_d, damp_step)
            fx_damp, ok_d = _eval(x_damp)
            if ok_d and abs(fx_damp) < abs(fx):
                x_new = x_damp
                fx_new = fx_damp
            elif abs(fx_new) >= abs(fx) * 0.9 and abs(x_new - x) < 1e-10:
                # Stagnating, break to try bracketing
                break
            else:
                # Neither full nor damped helped much; break
                break

        # Accept the step
        prev_x, prev_fx = x, fx
        x, fx = x_new, fx_new
        newton_iters += 1

    if _check_done():
        return best_x

    # ------------------------------------------------------------------
    # Phase 2: Bracketing + Brent's method
    # ------------------------------------------------------------------
    # Try to find a bracket [a, b] with sign change around current best
    a = x
    fa = fx
    if fa is None:
        fa, _ = _eval(a)
        if fa is None:
            return best_x

    # Expand bracket outward from current point and from x0
    b = None
    fb = None
    scale = 1.0
    found_bracket = False

    for _ in range(25):
        if _time_left() < 1.0 or eval_count >= max_evals - 10:
            break
        # Try expanding from current x
        b_pos = a + scale * max(1.0, abs(a))
        fb_pos, ok1 = _eval(b_pos)
        if ok1 and fa * fb_pos < 0:
            b = b_pos
            fb = fb_pos
            found_bracket = True
            break
        # Try the other direction
        b_neg = a - scale * max(1.0, abs(a))
        fb_neg, ok2 = _eval(b_neg)
        if ok2 and fa * fb_neg < 0:
            b = b_neg
            fb = fb_neg
            found_bracket = True
            break
        # Also try expanding from x0
        x0_scale = scale * max(1.0, abs(float(x0)))
        b_x0_pos = float(x0) + x0_scale
        fb_x0_pos, ok3 = _eval(b_x0_pos)
        if ok3:
            fx0, _ = _eval(float(x0))
            if fx0 is not None and fx0 * fb_x0_pos < 0:
                a = float(x0)
                fa = fx0
                b = b_x0_pos
                fb = fb_x0_pos
                found_bracket = True
                break
        b_x0_neg = float(x0) - x0_scale
        fb_x0_neg, ok4 = _eval(b_x0_neg)
        if ok4:
            fx0, _ = _eval(float(x0))
            if fx0 is not None and fx0 * fb_x0_neg < 0:
                a = float(x0)
                fa = fx0
                b = b_x0_neg
                fb = fb_x0_neg
                found_bracket = True
                break
        scale *= 2.0

    if found_bracket and b is not None and fb is not None and fa * fb < 0:
        # Use Brent's method
        try:
            def wrapped_f(x_val):
                nonlocal eval_count
                if eval_count >= max_evals or _time_left() < 0.5:
                    raise RuntimeError("budget exceeded")
                fx_val = float(f(x_val))
                eval_count += 1
                if not math.isfinite(fx_val):
                    raise RuntimeError("non-finite")
                if abs(fx_val) < best_abs_fx:
                    best_x = x_val
                    best_fx = fx_val
                    best_abs_fx = abs(fx_val)
                return fx_val

            x_root = brentq(wrapped_f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
            if abs(wrapped_f(x_root)) <= tol:
                return x_root
        except Exception:
            pass

    if _check_done():
        return best_x

    # ------------------------------------------------------------------
    # Phase 3: Secant method from current point
    # ------------------------------------------------------------------
    x1 = x
    f1 = fx
    x2 = x + 1e-6 * max(1.0, abs(x))
    f2, ok = _eval(x2)
    if not ok:
        return best_x

    secant_iters = 0
    max_secant = 50
    while secant_iters < max_secant and eval_count < max_evals and _time_left() > 1.0:
        if _check_done():
            return best_x
        if abs(x1 - x2) < 1e-15:
            break
        d = (f1 - f2) / (x1 - x2)
        if abs(d) < 1e-15:
            break
        x_new = x2 - f2 / d
        f_new, ok = _eval(x_new)
        if not ok:
            break
        if abs(f_new) <= tol:
            return x_new
        x1, f1 = x2, f2
        x2, f2 = x_new, f_new
        secant_iters += 1

    if _check_done():
        return best_x

    # ------------------------------------------------------------------
    # Phase 4: Small restart portfolio
    # ------------------------------------------------------------------
    offsets = [1e-4, -1e-4, 1e-3, -1e-3, 0.1, -0.1, 1.0, -1.0]
    for off in offsets:
        if eval_count >= max_evals - 20 or _time_left() < 2.0:
            break
        x_try = float(x0) + off * max(1.0, abs(x0))
        fx_try, ok = _eval(x_try)
        if not ok:
            continue
        if abs(fx_try) <= tol:
            return x_try

        # Quick Newton from this point
        for _ in range(10):
            if _check_done():
                return best_x
            if eval_count >= max_evals or _time_left() < 1.0:
                break
            d = _fd_deriv(x_try, fx_try)
            if d is None or abs(d) < 1e-15:
                break
            step = fx_try / d
            x_new = x_try - step
            max_step = 10.0 * max(1.0, abs(x_try))
            if abs(step) > max_step:
                x_new = x_try - math.copysign(max_step, step)
            fx_new, ok = _eval(x_new)
            if not ok:
                break
            if abs(fx_new) <= tol:
                return x_new
            # Damped fallback
            if abs(fx_new) >= abs(fx_try) * 0.9:
                x_damp = x_try - 0.5 * step
                if abs(0.5 * step) > 5.0 * max(1.0, abs(x_try)):
                    x_damp = x_try - math.copysign(5.0 * max(1.0, abs(x_try)), 0.5 * step)
                fx_damp, ok_d = _eval(x_damp)
                if ok_d and abs(fx_damp) < abs(fx_try):
                    x_try, fx_try = x_damp, fx_damp
                else:
                    break
            else:
                x_try, fx_try = x_new, fx_new

    if _check_done():
        return best_x

    # ------------------------------------------------------------------
    # Phase 5: Final tight-tolerance polish
    # ------------------------------------------------------------------
    if best_abs_fx > tol and best_abs_fx < 1e-4:
        x = best_x
        fx = best_fx
        for _ in range(20):
            if _time_left() < 1.0 or eval_count >= max_evals:
                break
            if abs(fx) <= 1e-12:
                return x
            h = 1e-8 * max(1.0, abs(x))
            d = _fd_deriv(x, fx)
            if d is None or abs(d) < 1e-15:
                break
            x_new = x - fx / d
            fx_new, ok = _eval(x_new)
            if not ok:
                break
            if abs(fx_new) < abs(fx):
                x, fx = x_new, fx_new
                if abs(fx) < best_abs_fx:
                    best_x = x
                    best_fx = fx
                    best_abs_fx = abs(fx)
            else:
                break

    return best_x
