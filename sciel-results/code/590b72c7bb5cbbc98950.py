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
    
    # Secant method: uses two previous points to estimate derivative
    # Each iteration costs 1 function call (reuses previous fx)
    x_prev = x - 1e-4 * max(1.0, abs(x))  # perturb to get second point
    try:
        fx_prev = f(x_prev)
    except Exception:
        return best_x
    if not _is_finite(fx_prev):
        return best_x
    
    if abs(fx_prev) < abs(best_fx):
        best_x = x_prev
        best_fx = fx_prev
    if abs(fx_prev) <= 1e-8:
        return x_prev
    
    # Now we have (x_prev, fx_prev) and (x, fx)
    # Secant iteration
    for i in range(80):
        if _time_left() < 2.0:
            break
        
        denom = fx - fx_prev
        if abs(denom) < 1e-300:
            # Degenerate: fall back to bisection-like step
            x_new = 0.5 * (x + x_prev)
        else:
            x_new = x - fx * (x - x_prev) / denom
        
        if not _is_finite(x_new):
            break
        
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
        
        # Shift: (x_prev, fx_prev) -> (x, fx), (x, fx) -> (x_new, fx_new)
        x_prev = x
        fx_prev = fx
        x = x_new
        fx = fx_new
    
    return best_x
