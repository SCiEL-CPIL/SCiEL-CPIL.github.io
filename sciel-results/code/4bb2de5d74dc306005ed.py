"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def find_root(f, x0):
    start_time = time.time()
    # 150s total limit for 3 equations. 10s per equation is safe.
    deadline = start_time + 10.0
    
    x = float(x0)
    best_x = x
    try:
        best_f = f(x)
    except Exception:
        return x

    if not math.isfinite(best_f):
        return x

    if abs(best_f) <= 1e-8:
        return best_x

    for _ in range(100):
        if time.time() > deadline:
            break
            
        try:
            fx = f(x)
        except Exception:
            break
            
        if not math.isfinite(fx):
            break
            
        if abs(fx) < abs(best_f):
            best_f = fx
            best_x = x
            
        if abs(fx) <= 1e-12:
            return x

        # Central difference for better accuracy
        h = 1e-5 * max(1.0, abs(x))
        try:
            fx_plus = f(x + h)
            fx_minus = f(x - h)
        except Exception:
            break
        
        if not (math.isfinite(fx_plus) and math.isfinite(fx_minus)):
            break
            
        dfx = (fx_plus - fx_minus) / (2.0 * h)
        
        if dfx == 0.0:
            break
            
        x_new = x - fx / dfx
        
        if not math.isfinite(x_new):
            break
            
        # Safeguard: Limit step size to prevent divergence
        max_step = 1.0 + abs(x)
        step = x_new - x
        if abs(step) > max_step:
            x_new = x + math.copysign(max_step, step)
            
        x = x_new
        
    return best_x
# EVOLVE-BLOCK-END
