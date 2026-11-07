"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    deadline = time.time() + 115.0  # safety margin before 120s kill
    x = float(x0)
    fx = f(x)
    
    # Phase 1: Secant method (derivative-free, 2 calls per iteration after first)
    # For smooth polynomials, secant converges superlinearly (~1.618 order)
    if abs(fx) <= 1e-12:
        return x
    
    # Initial step to get second point
    h = 1e-8 * max(1.0, abs(x))
    x1 = x + h
    fx1 = f(x1)
    
    if abs(fx1) <= 1e-12:
        return x1
    
    best_x = x if abs(fx) < abs(fx1) else x1
    best_fx = fx if abs(fx) < abs(fx1) else fx1
    
    # Secant iterations
    for i in range(50):
        if time.time() > deadline:
            return best_x
        if abs(fx1) < abs(fx):
            x, x1 = x1, x
            fx, fx1 = fx1, fx
        # x is now the better estimate
        if abs(fx) <= 1e-12:
            return x
        
        # Secant step
        denom = fx1 - fx
        if abs(denom) < 1e-30:
            break
        x_new = x1 - fx1 * (x1 - x) / denom
        
        # Safeguard: if step is too large, dampen
        if abs(x_new - x1) > 10.0 * max(1.0, abs(x1)):
            x_new = x1 + 0.5 * (x - x1)
        
        fx_new = f(x_new)
        
        if not math.isfinite(x_new) or not math.isfinite(fx_new):
            # Fallback: try a smaller step
            x_new = x1 + 0.1 * (x - x1)
            fx_new = f(x_new)
            if not math.isfinite(x_new) or not math.isfinite(fx_new):
                break
        
        if abs(fx_new) < abs(best_fx):
            best_x = x_new
            best_fx = fx_new
        
        if abs(fx_new) <= 1e-12:
            return x_new
        
        # Shift: x = old x1, x1 = new point
        x = x1
        fx = fx1
        x1 = x_new
        fx1 = fx_new
    
    # Phase 2: High-precision polish using Newton with central difference
    # Use the best estimate found
    x = best_x
    fx = f(x)
    if abs(fx) <= 1e-12:
        return x
    
    for i in range(30):
        if time.time() > deadline:
            return x
        h = 1e-10 * max(1.0, abs(x))
        d = (f(x + h) - f(x - h)) / (2 * h)
        if abs(d) < 1e-30:
            break
        x_new = x - fx / d
        if not math.isfinite(x_new):
            break
        fx_new = f(x_new)
        if abs(fx_new) < abs(fx):
            x = x_new
            fx = fx_new
        else:
            break
        if abs(fx) <= 1e-14:
            break
    
    return x
# EVOLVE-BLOCK-END
