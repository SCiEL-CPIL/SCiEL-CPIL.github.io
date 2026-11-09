"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Hybrid root finder: secant method with adaptive stepping and deadline guard.
    Returns x with |f(x)| <= 1e-8, minimizing function call count.
    """
    deadline = time.monotonic() + 110.0  # safety margin before 120s limit
    x = float(x0)
    
    # Check initial point
    fx = f(x)
    if abs(fx) <= 1e-8:
        return x
    
    # Try secant method with two initial points
    # Use a small perturbation for the second point
    h = 1e-4 * max(1.0, abs(x))
    x1 = x
    x2 = x + h
    f1 = fx
    f2 = f(x2)
    
    if abs(f2) <= 1e-8:
        return x2
    
    best_x = x if abs(fx) < abs(f2) else x2
    best_f = f1 if abs(f1) < abs(f2) else f2
    
    # Secant method iterations
    for _ in range(50):
        if time.monotonic() > deadline:
            break
        if f1 == f2:
            # Degenerate case: try a different step
            h = 1e-3 * max(1.0, abs(x1))
            x2 = x1 + h
            f2 = f(x2)
            if abs(f2) <= 1e-8:
                return x2
            if f1 == f2:
                break
            continue
        
        # Secant step
        denom = f2 - f1
        if denom == 0:
            break
        x_new = x2 - f2 * (x2 - x1) / denom
        
        # Check for non-finite
        if not math.isfinite(x_new):
            # Try bisection-like fallback
            if f1 * f2 > 0:
                # No sign change; nudge x2
                x2 = x1 + 0.5 * h
                f2 = f(x2)
                if abs(f2) <= 1e-8:
                    return x2
                continue
            else:
                # Bisection
                x_new = (x1 + x2) / 2.0
                f_new = f(x_new)
                if abs(f_new) <= 1e-8:
                    return x_new
                if f1 * f_new <= 0:
                    x2, f2 = x_new, f_new
                else:
                    x1, f1 = x_new, f_new
                continue
        
        f_new = f(x_new)
        
        if abs(f_new) <= 1e-8:
            return x_new
        
        if abs(f_new) < abs(best_f):
            best_x = x_new
            best_f = f_new
        
        # Update secant pair: keep the two most recent points
        x1, f1 = x2, f2
        x2, f2 = x_new, f_new
        
        # Check convergence of x
        if abs(x2 - x1) < 1e-12 * max(1.0, abs(x2)):
            break
    
    # If secant didn't converge well, try Newton with finite difference as fallback
    x = best_x
    fx = f(x)
    if abs(fx) <= 1e-8:
        return x
    
    for _ in range(20):
        if time.monotonic() > deadline:
            break
        h = 1e-6 * max(1.0, abs(x))
        d = (f(x + h) - fx) / h
        if d == 0:
            break
        x_new = x - fx / d
        if not math.isfinite(x_new):
            break
        fx_new = f(x_new)
        if abs(fx_new) <= 1e-8:
            return x_new
        if abs(fx_new) < abs(best_f):
            best_x = x_new
            best_f = fx_new
        x = x_new
        fx = fx_new
    
    # Final polish: if we're close but not quite at 1e-8, do a few more secant steps
    if abs(best_f) < 1e-6:
        x1, x2 = best_x, best_x + 1e-8 * max(1.0, abs(best_x))
        f1 = best_f
        f2 = f(x2)
        for _ in range(10):
            if time.monotonic() > deadline:
                break
            if f1 == f2:
                break
            denom = f2 - f1
            if denom == 0:
                break
            x_new = x2 - f2 * (x2 - x1) / denom
            if not math.isfinite(x_new):
                break
            f_new = f(x_new)
            if abs(f_new) <= 1e-8:
                return x_new
            if abs(f_new) < abs(best_f):
                best_x = x_new
                best_f = f_new
            x1, f1 = x2, f2
            x2, f2 = x_new, f_new
    
    return best_x
# EVOLVE-BLOCK-END
