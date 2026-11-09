"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def find_root(f, x0):
    deadline = time.time() + 110  # safety margin before 120s limit
    x = float(x0)
    
    # Check initial point
    fx = f(x)
    if abs(fx) <= 1e-8:
        return x
    
    best_x = x
    best_fx = fx
    
    # Try Newton's method with bisection fallback
    # First, try to establish a bracket if possible
    h = 1e-6 * max(1.0, abs(x))
    
    for i in range(80):
        if time.time() > deadline:
            break
        
        # Finite difference derivative
        d = (f(x + h) - fx) / h
        
        if d == 0 or not all(v == v for v in [d, fx]):  # NaN/inf check
            break
        
        # Newton step
        x_new = x - fx / d
        
        # Safeguard: if Newton step is too large, dampen it
        if abs(x_new - x) > 10 * h:
            x_new = x + 10 * h * (1 if x_new > x else -1)
        
        fx_new = f(x_new)
        
        if abs(fx_new) < abs(fx):
            x = x_new
            fx = fx_new
            h = 1e-6 * max(1.0, abs(x))
            if abs(fx) <= 1e-8:
                return x
            if abs(fx) <= 1e-12:
                return x
        else:
            # Newton step made things worse; take a small step in the better direction
            x = x + h * (1 if fx_new > fx else -1)
            fx = f(x)
            if abs(fx) <= 1e-8:
                return x
    
    return x
# EVOLVE-BLOCK-END
