"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin before 120s limit
    
    x = float(x0)
    fx = f(x)
    
    if abs(fx) <= 1e-8:
        return x
    
    # Establish second point with a small geometric step
    h = 1e-4 * max(1.0, abs(x))
    x2 = x + h
    fx2 = f(x2)
    
    if abs(fx2) <= 1e-8:
        return x2
    
    # If both points are on the same side and far apart, try a different step
    # Secant method iteration
    best_x = x if abs(fx) < abs(fx2) else x2
    best_fx = fx if abs(fx) < abs(fx2) else fx2
    
    for i in range(50):
        if time.time() > deadline:
            break
        
        denom = fx2 - fx
        if denom == 0.0 or abs(denom) < 1e-30:
            # Degenerate: try a new point
            h = 1e-4 * max(1.0, abs(x2))
            x_new = x2 + h
            fx_new = f(x_new)
            if abs(fx_new) <= 1e-8:
                return x_new
            x, fx = x2, fx2
            x2, fx2 = x_new, fx_new
            continue
        
        x_new = x2 - fx2 * (x2 - x) / (fx2 - fx)
        
        # Check for non-finite
        if not (x_new == x_new) or x_new == float('inf') or x_new == float('-inf'):
            # Fallback: bisection-like step
            x_new = 0.5 * (x + x2)
        
        fx_new = f(x_new)
        
        if not (fx_new == fx_new) or fx_new == float('inf') or fx_new == float('-inf'):
            # Try a smaller step
            x_new = x2 - 0.5 * h
            fx_new = f(x_new)
            if abs(fx_new) <= 1e-8:
                return x_new
            x, fx = x2, fx2
            x2, fx2 = x_new, fx_new
            continue
        
        if abs(fx_new) <= 1e-8:
            return x_new
        
        if abs(fx_new) < abs(best_fx):
            best_x = x_new
            best_fx = fx_new
        
        # Shift points: keep the two most recent
        x, fx = x2, fx2
        x2, fx2 = x_new, fx_new
    
    return best_x
# EVOLVE-BLOCK-END
