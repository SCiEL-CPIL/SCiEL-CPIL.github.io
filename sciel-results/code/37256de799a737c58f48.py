"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
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
    best_abs = abs(fx)
    
    # Newton's method with finite-difference derivative
    for i in range(50):
        if _time_left() < 2.0:
            break
        
        h = 1e-7 * max(1.0, abs(x))
        try:
            fx_plus = f(x + h)
            fx_minus = f(x - h)
        except Exception:
            break
        
        if not _is_finite(fx_plus) or not _is_finite(fx_minus):
            break
        
        # Central difference for better accuracy
        d = (fx_plus - fx_minus) / (2 * h)
        
        if not _is_finite(d) or abs(d) < 1e-30:
            # Try a different step
            h2 = 1e-5 * max(1.0, abs(x))
            try:
                fx_plus2 = f(x + h2)
                fx_minus2 = f(x - h2)
            except Exception:
                break
            if not _is_finite(fx_plus2) or not _is_finite(fx_minus2):
                break
            d = (fx_plus2 - fx_minus2) / (2 * h2)
            if not _is_finite(d) or abs(d) < 1e-30:
                break
        
        x_new = x - fx / d
        
        if not _is_finite(x_new):
            # Damping
            x_new = x - 0.5 * fx / d
            if not _is_finite(x_new):
                break
        
        try:
            fx_new = f(x_new)
        except Exception:
            break
        
        if not _is_finite(fx_new):
            break
        
        abs_fx_new = abs(fx_new)
        
        if abs_fx_new < best_abs:
            best_x = x_new
            best_fx = fx_new
            best_abs = abs_fx_new
        
        if abs_fx_new <= 1e-8:
            return x_new
        
        # Check for convergence
        if abs(x_new - x) < 1e-14 * max(1.0, abs(x_new)):
            break
        
        x = x_new
    
    return best_x
# EVOLVE-BLOCK-END
