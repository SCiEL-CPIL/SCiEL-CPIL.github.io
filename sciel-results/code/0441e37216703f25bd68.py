import math
import time

_START_TIME = time.time()
_WALL_LIMIT = 115.0  # safety margin before 120s

def _time_left():
    return _WALL_LIMIT - (time.time() - _START_TIME)

def _is_finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)

def find_root(f, x0):
    x = float(x0)
    
    # Check initial point
    try:
        fx = f(x)
    except Exception:
        return x
    if not _is_finite(fx):
        return x
    if abs(fx) <= 1e-8:
        return x
    
    best_x = x
    best_fx = fx
    
    # Try multiple perturbations to find a good second point
    # Use adaptive step sizes to bracket a sign change if possible
    x_prev = None
    fx_prev = None
    step_sizes = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
    base_step = max(1.0, abs(x))
    
    for ss in step_sizes:
        for sign in [1, -1]:
            xp = x + sign * ss * base_step
            try:
                fpx = f(xp)
            except Exception:
                continue
            if not _is_finite(fpx):
                continue
            if abs(fpx) < abs(best_fx):
                best_x = xp
                best_fx = fpx
            if abs(fpx) <= 1e-8:
                return xp
            # Prefer a point that brackets a sign change
            if x_prev is None:
                if fx * fpx < 0:
                    x_prev = xp
                    fx_prev = fpx
                    break
                elif x_prev is None:
                    x_prev = xp
                    fx_prev = fpx
        if x_prev is not None and fx * fx_prev < 0:
            break
    
    # If we couldn't find any valid second point, return best
    if x_prev is None or fx_prev is None:
        return best_x
    
    if abs(fx_prev) < abs(best_fx):
        best_x = x_prev
        best_fx = fx_prev
    if abs(fx_prev) <= 1e-8:
        return x_prev
    
    # Now we have (x_prev, fx_prev) and (x, fx)
    # Use a hybrid secant/bisection approach
    # Track a bracket if one exists
    a = x_prev
    fa = fx_prev
    b = x
    fb = fx
    
    # Ensure a < b for bisection reference
    if a > b:
        a, b = b, a
        fa, fb = fb, fa
    
    for i in range(80):
        if _time_left() < 2.0:
            break
        
        # Secant step
        denom = fx - fx_prev
        if abs(denom) < 1e-300:
            # Degenerate: fall back to bisection-like step
            x_new = 0.5 * (x + x_prev)
        else:
            x_new = x - fx * (x - x_prev) / denom
        
        if not _is_finite(x_new):
            break
        
        # Check if secant step is diverging (too far from current points)
        max_dist = max(abs(x - x_prev), 1e-10)
        if abs(x_new - x) > 100 * max_dist and abs(x_new - x_prev) > 100 * max_dist:
            # Secant is diverging, use bisection on bracket if available
            if fa * fb < 0:
                x_new = 0.5 * (a + b)
            else:
                # No bracket, dampen the secant step
                x_new = x + 0.5 * (x - x_prev)
        
        try:
            fx_new = f(x_new)
        except Exception:
            break
        
        if not _is_finite(fx_new):
            break
        
        if abs(fx_new) < abs(best_fx):
            best_x = x_new
            best_fx = fx_new
        
        if abs(fx_new) <= 1e-8:
            return x_new
        
        # Update bracket if we have a sign change
        if fx_new * fx < 0:
            # (x, fx) and (x_new, fx_new) bracket a root
            if x < x_new:
                a, b = x, x_new
                fa, fb = fx, fx_new
            else:
                a, b = x_new, x
                fa, fb = fx_new, fx
        elif fx_new * fx_prev < 0:
            # (x_prev, fx_prev) and (x_new, fx_new) bracket a root
            if x_prev < x_new:
                a, b = x_prev, x_new
                fa, fb = fx_prev, fx_new
            else:
                a, b = x_new, x_prev
                fa, fb = fx_new, fx_prev
        
        # Shift: (x_prev, fx_prev) -> (x, fx), (x, fx) -> (x_new, fx_new)
        x_prev = x
        fx_prev = fx
        x = x_new
        fx = fx_new
    
    # Final polish: if we're close, try a few more Newton-like steps with tighter tol
    if abs(best_fx) < 1e-4:
        xp = best_x
        fp = best_fx
        for i in range(20):
            if _time_left() < 1.0:
                break
            # Estimate derivative using secant from last two points
            h = 1e-6 * max(1.0, abs(xp))
            try:
                fp2 = f(xp + h)
            except Exception:
                break
            if not _is_finite(fp2):
                break
            deriv = (fp2 - fp) / h
            if abs(deriv) < 1e-300:
                break
            x_new = xp - fp / deriv
            if not _is_finite(x_new):
                break
            try:
                f_new = f(x_new)
            except Exception:
                break
            if not _is_finite(f_new):
                break
            if abs(f_new) < abs(best_fx):
                best_x = x_new
                best_fx = f_new
            if abs(f_new) <= 1e-8:
                return x_new
            xp = x_new
            fp = f_new
    
    return best_x
