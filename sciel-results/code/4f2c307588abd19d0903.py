"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """Find root of f near x0 using damped Newton with fallback to bisection."""
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s kill
    x = float(x0)
    fx = f(x)
    best_x = x
    best_fx = abs(fx)
    
    # Try damped Newton's method
    for i in range(80):
        if time.time() > deadline:
            break
        if abs(fx) <= 1e-12:
            return x
        if abs(fx) < best_fx:
            best_fx = abs(fx)
            best_x = x
        
        # Central difference derivative with adaptive step
        h = 1e-7 * max(1.0, abs(x))
        fp = f(x + h)
        fm = f(x - h)
        d = (fp - fm) / (2.0 * h)
        
        if d == 0 or not math.isfinite(d):
            break
        
        # Newton step with damping
        step = fx / d
        # Limit step size to prevent wild jumps
        max_step = 10.0 * max(1.0, abs(x))
        if abs(step) > max_step:
            step = max_step * math.copysign(1.0, step)
        
        x_new = x - step
        
        # Check if new point is valid
        if not math.isfinite(x_new):
            break
        
        fx_new = f(x_new)
        if not math.isfinite(fx_new):
            break
        
        # Accept step if it reduces |f|, otherwise damp
        if abs(fx_new) < abs(fx):
            x = x_new
            fx = fx_new
        else:
            # Damp the step
            x_new = x - 0.5 * step
            fx_new = f(x_new)
            if abs(fx_new) < abs(fx):
                x = x_new
                fx = fx_new
            else:
                # Try even smaller step
                x_new = x - 0.25 * step
                fx_new = f(x_new)
                if abs(fx_new) < abs(fx):
                    x = x_new
                    fx = fx_new
                else:
                    break  # Newton is struggling
    
    if abs(fx) < best_fx:
        best_x = x
        best_fx = abs(fx)
    
    if best_fx <= 1e-8:
        return best_x
    
    # Fallback: try to bracket and use bisection
    # Expand bracket around best_x
    a = best_x - 1.0
    b = best_x + 1.0
    fa = f(a)
    fb = f(b)
    
    # Try to find a bracket
    for _ in range(20):
        if time.time() > deadline:
            break
        if not math.isfinite(fa) or not math.isfinite(fb):
            break
        if fa * fb <= 0:
            break
        # Expand
        a = best_x - (b - a)
        b = best_x + (b - a)
        fa = f(a)
        fb = f(b)
    
    if math.isfinite(fa) and math.isfinite(fb) and fa * fb <= 0:
        # Bisection
        for _ in range(60):
            if time.time() > deadline:
                break
            mid = 0.5 * (a + b)
            fm = f(mid)
            if abs(fm) <= 1e-12:
                return mid
            if fa * fm <= 0:
                b = mid
                fb = fm
            else:
                a = mid
                fa = fm
            if abs(b - a) < 1e-14:
                break
        return 0.5 * (a + b)
    
    return best_x
# EVOLVE-BLOCK-END
