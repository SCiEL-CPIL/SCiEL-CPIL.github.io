"""Root finding for ill-conditioned equations: robust adaptive hybrid with polish."""
import time
import math
import numpy as np
from scipy.optimize import brentq

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8, using at most 500 evaluations of f.
    Returns the best x found (smallest |f(x)|) within the budget.
    """
    start_time = time.time()
    eval_count = 0
    max_evals = 500
    tol = 1e-8
    deadline = 135.0

    # State to track the best point seen so far
    best_x = float(x0)
    try:
        best_fx = float(f(best_x))
        eval_count += 1
        best_abs_fx = abs(best_fx) if math.isfinite(best_fx) else float('inf')
    except Exception:
        best_fx = 0.0
        best_abs_fx = float('inf')

    def _time_left():
        return deadline - (time.time() - start_time)

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
    # Phase 1: Safeguarded Newton with finite difference derivative
    # ------------------------------------------------------------------
    x = float(x0)
    fx, _ = _eval(x)
    if fx is None:
        return best_x

    def _fd_deriv(x, fx):
        """Compute derivative using central difference if possible, else forward."""
        h = 1e-7 * max(1.0, abs(x))
        # Try central difference: 2 evals
        if eval_count + 2 < max_evals and _time_left() > 1.0:
            fxp, ok1 = _eval(x + h)
            if ok1:
                fxm, ok2 = _eval(x - h)
                if ok2:
                    return (fxp - fxm) / (2 * h)
        # Fallback to forward difference: 1 eval
        if eval_count + 1 < max_evals and _time_left() > 0.5:
            fxp, ok = _eval(x + h)
            if ok:
                return (fxp - fx) / h
        return None

    newton_iters = 0
    max_newton = 40
    prev_x = x
    prev_fx = fx

    while newton_iters < max_newton and eval_count < max_evals and _time_left() > 1.0:
        if _check_done():
            return best_x

        # Try Newton step
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

        # Check for progress
        if abs(fx_new) >= abs(fx) * 0.9 and abs(x_new - x) < 1e-10:
            # Stagnating, break to try bracketing
            break

        # Accept the step
        prev_x, prev_fx = x, fx
        x, fx = x_new, fx_new
        newton_iters += 1

    if _check_done():
        return best_x

    # ------------------------------------------------------------------
    # Phase 2: Bracketing + Brent's method (using scipy for robustness)
    # ------------------------------------------------------------------
    a = x
    fa = fx
    if fa is None:
        fa, _ = _eval(a)
        if fa is None:
            return best_x

    # Expand bracket outward
    b = None
    fb = None
    scale = 1.0
    for _ in range(20):
        if _time_left() < 1.0 or eval_count >= max_evals - 10:
            break
        b = a + scale * max(1.0, abs(a))
        fb, ok = _eval(b)
        if not ok:
            break
        if fa * fb < 0:
            break
        # Also try the other direction
        b_neg = a - scale * max(1.0, abs(a))
        fb_neg, ok2 = _eval(b_neg)
        if ok2 and fa * fb_neg < 0:
            b = b_neg
            fb = fb_neg
            break
        scale *= 2.0

    if b is not None and fb is not None and fa * fb < 0:
        # Use scipy's brentq for robustness
        try:
            x_root = brentq(f, a, b, xtol=1e-15, rtol=1e-15, maxiter=100)
            fx_root, ok = _eval(x_root)
            if ok and abs(fx_root) <= tol:
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
            x_try, fx_try = x_new, fx_new

    if _check_done():
        return best_x

    # ------------------------------------------------------------------
    # Phase 5: Final high-precision polish pass
    # ------------------------------------------------------------------
    # Try to polish the best_x found so far with tighter Newton iterations
    if eval_count < max_evals - 10 and _time_left() > 2.0:
        x_polish = best_x
        fx_polish, ok = _eval(x_polish)
        if ok and abs(fx_polish) < 1e-6:  # Only polish if reasonably close
            for _ in range(20):
                if _check_done():
                    return best_x
                if eval_count >= max_evals or _time_left() < 1.0:
                    break
                d = _fd_deriv(x_polish, fx_polish)
                if d is None or abs(d) < 1e-15:
                    break
                step = fx_polish / d
                x_new = x_polish - step
                max_step = 1.0 * max(1.0, abs(x_polish))  # Smaller steps for polish
                if abs(step) > max_step:
                    x_new = x_polish - math.copysign(max_step, step)
                fx_new, ok = _eval(x_new)
                if not ok:
                    break
                if abs(fx_new) <= tol:
                    return x_new
                if abs(fx_new) < abs(fx_polish):
                    x_polish, fx_polish = x_new, fx_new
                else:
                    break

    return best_x
