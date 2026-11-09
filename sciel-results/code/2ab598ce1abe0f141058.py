"""Root finding for ill-conditioned equations: robust adaptive hybrid."""
import time
import math
import numpy as np
from scipy.optimize import brentq, minimize

# Global deadline: leave a safety margin before the 150s kill.
_DEADLINE_SECONDS = 135.0

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
    # Phase 1: Inverse quadratic interpolation (history-based)
    # ------------------------------------------------------------------
    x = float(x0)
    fx, _ = _eval(x)
    if fx is None:
        return best_x

    history = [(x, fx)]

    def _interp_next(hist):
        n = len(hist)
        if n < 2:
            return None

        pts = sorted(hist, key=lambda p: p[0])

        if n == 2:
            x1, f1 = pts[0]
            x2, f2 = pts[1]
            if abs(x2 - x1) < 1e-15:
                return None
            d = (f2 - f1) / (x2 - x1)
            if abs(d) < 1e-15:
                return None
            return x2 - f2 / d

        x1, f1 = pts[0]
        x2, f2 = pts[1]
        x3, f3 = pts[2]

        if abs(x2 - x1) < 1e-14 or abs(x3 - x2) < 1e-14 or abs(x3 - x1) < 1e-14:
            return None

        d12 = x1 - x2
        d13 = x1 - x3
        d23 = x2 - x3

        if abs(d12) < 1e-14 or abs(d13) < 1e-14 or abs(d23) < 1e-14:
            return None

        c1 = f1 / (d12 * d13)
        c2 = f2 / ((x2 - x1) * (x2 - x3))
        c3 = f3 / ((x3 - x1) * (x3 - x2))

        A = c1 + c2 + c3
        B = -c1 * (x2 + x3) - c2 * (x1 + x3) - c3 * (x1 + x2)
        C = c1 * x2 * x3 + c2 * x1 * x3 + c3 * x1 * x2

        if abs(A) < 1e-30:
            if abs(B) < 1e-30:
                return None
            return -C / B

        disc = B * B - 4.0 * A * C
        if disc < 0:
            return None

        sqrt_disc = math.sqrt(disc)
        x_root1 = (-B + sqrt_disc) / (2.0 * A)
        x_root2 = (-B - sqrt_disc) / (2.0 * A)

        x_recent = hist[-1][0]
        if abs(x_root1 - x_recent) <= abs(x_root2 - x_recent):
            return x_root1
        else:
            return x_root2

    max_interp_iters = 30
    interp_iters = 0
    while interp_iters < max_interp_iters and eval_count < max_evals and _time_left() > 1.0:
        if _check_done():
            return best_x

        x_new = _interp_next(history)
        if x_new is None:
            break

        max_step = 20.0 * max(1.0, abs(x))
        if abs(x_new - x) > max_step:
            x_new = x + math.copysign(max_step, x_new - x)

        fx_new, ok = _eval(x_new)
        if not ok:
            break

        if abs(fx_new) <= tol:
            return x_new

        history.append((x_new, fx_new))
        if len(history) > 3:
            history.pop(0)

        if len(history) >= 2:
            if abs(history[-1][0] - history[-2][0]) < 1e-12 and abs(history[-1][1] - history[-2][1]) < 1e-12:
                break

        x, fx = x_new, fx_new
        interp_iters += 1

    if _check_done():
        return best_x

    # ------------------------------------------------------------------
    # Phase 2: Safeguarded Secant
    # ------------------------------------------------------------------
    # Initialize with two points
    x1 = x
    f1 = fx
    # Get a second point
    x2 = x1 + 1e-6 * max(1.0, abs(x1))
    f2, ok = _eval(x2)
    if not ok:
        return best_x

    secant_iters = 0
    max_secant = 30
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
        
        # Safeguard: if new point is worse and far away, step back
        if abs(f_new) > abs(f2) * 10 and abs(x_new - x2) > 10 * max(1.0, abs(x2)):
             # Use bisection-like step towards the better of x1, x2
             x_new = 0.5 * (x1 + x2)
             f_new, ok = _eval(x_new)
             if not ok: break
             if abs(f_new) <= tol: return x_new

        # Replace the older point
        x1, f1 = x2, f2
        x2, f2 = x_new, f_new
        secant_iters += 1

    if _check_done():
        return best_x

    # ------------------------------------------------------------------
    # Phase 3: Bracketing + Brent
    # ------------------------------------------------------------------
    a = x2
    fa = f2
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
        b_neg = a - scale * max(1.0, abs(a))
        fb_neg, ok2 = _eval(b_neg)
        if ok2 and fa * fb_neg < 0:
            b = b_neg
            fb = fb_neg
            break
        scale *= 2.0

    if b is not None and fb is not None and fa * fb < 0:
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

            x_root = brentq(wrapped_f, a, b, xtol=1e-12, rtol=1e-12, maxiter=50)
            if abs(wrapped_f(x_root)) <= tol:
                return x_root
        except Exception:
            pass

    if _check_done():
        return best_x

    # ------------------------------------------------------------------
    # Phase 4: L-BFGS-B Minimization of f(x)^2
    # ------------------------------------------------------------------
    # This is the strategic change requested.
    # We minimize g(x) = f(x)^2.
    # We use a wrapper to count evals and stop if budget exceeded.
    
    def objective(x_arr):
        nonlocal eval_count, best_x, best_fx, best_abs_fx
        if eval_count >= max_evals or _time_left() < 0.5:
            raise RuntimeError("budget exceeded")
        x_val = float(x_arr[0])
        try:
            fx_val = float(f(x_val))
        except Exception:
            # Return a large penalty if f fails
            return 1e100
        eval_count += 1
        if not math.isfinite(fx_val):
            return 1e100
        if abs(fx_val) < best_abs_fx:
            best_x = x_val
            best_fx = fx_val
            best_abs_fx = abs(fx_val)
        return fx_val * fx_val

    # Start from the best point found so far
    x_start = np.array([best_x])
    
    try:
        # L-BFGS-B is good for ill-conditioned problems due to its line search
        # approx_grad=True uses finite differences for gradient (costs extra evals)
        # But it is robust.
        res = minimize(objective, x_start, method='L-BFGS-B', 
                       options={'maxiter': 50, 'maxls': 50, 'approx_grad': True, 'ftol': 1e-12, 'gtol': 1e-12})
        
        if res.success and res.fun < 1e100:
            x_candidate = float(res.x[0])
            fx_candidate, ok = _eval(x_candidate)
            if ok and abs(fx_candidate) <= tol:
                return x_candidate
            # Update best if better
            if ok and abs(fx_candidate) < best_abs_fx:
                best_x = x_candidate
                best_fx = fx_candidate
                best_abs_fx = abs(fx_candidate)
                
    except Exception:
        pass

    if _check_done():
        return best_x

    # ------------------------------------------------------------------
    # Phase 5: Final Polish
    # ------------------------------------------------------------------
    # If we are close, try a few more secant steps with tight tolerance
    if best_abs_fx < 1e-4:
        x1 = best_x
        f1 = best_fx
        # Get a neighbor
        h = 1e-8 * max(1.0, abs(x1))
        x2 = x1 + h
        f2, ok = _eval(x2)
        if ok:
            for _ in range(10):
                if _check_done():
                    return best_x
                if eval_count >= max_evals or _time_left() < 0.5:
                    break
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

    return best_x
