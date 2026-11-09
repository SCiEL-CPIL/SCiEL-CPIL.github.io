import math
import time

_START_TIME = time.time()
_WALL_LIMIT = 145.0  # Safety margin before 150s

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
    call_count = 1  # Count the initial evaluation
    
    # Secant method initialization
    # Use a relative perturbation to find a second point
    # 1e-4 provides a good balance for secant initialization
    scale = max(1.0, abs(x))
    x_prev = x + 1e-4 * scale
    try:
        fx_prev = f(x_prev)
        call_count += 1
    except Exception:
        return best_x
    if not _is_finite(fx_prev):
        return best_x
    
    if abs(fx_prev) < abs(best_fx):
        best_x = x_prev
        best_fx = fx_prev
    if abs(fx_prev) <= 1e-8:
        return x_prev
    
    # Secant iteration
    # We use a slightly higher internal tolerance target (1e-12) to ensure the 1e-8 check passes robustly
    for i in range(100):
        if _time_left() < 2.0:
            break
        if call_count >= 490:  # Stay well under 500
            break
        
        denom = fx - fx_prev
        
        # If the denominator is too small, the points are effectively the same or the slope is undefined
        if abs(denom) < 1e-300:
            # Fallback: try a small Newton step with FD derivative to escape
            h = 1e-7 * max(1.0, abs(x))
            try:
                fx_plus = f(x + h)
                call_count += 1
            except Exception:
                break
            if not _is_finite(fx_plus):
                break
            d = (fx_plus - fx) / h
            if d == 0 or not _isfinite(d):
                break
            x_new = x - fx / d
            if not _is_finite(x_new):
                break
            try:
                fx_new = f(x_new)
                call_count += 1
            except Exception:
                break
            if not _is_finite(fx_new):
                break
            if abs(fx_new) < abs(best_fx):
                best_x = x_new
                best_fx = fx_new
            if abs(fx_new) <= 1e-8:
                return x_new
            x_prev = x
            fx_prev = fx
            x = x_new
            fx = fx_new
            continue
        
        x_new = x - fx * (x - x_prev) / denom
        
        if not _is_finite(x_new):
            break
        
        try:
            fx_new = f(x_new)
            call_count += 1
        except Exception:
            break
        
        if not _is_finite(fx_new):
            break
        
        if abs(fx_new) < abs(best_fx):
            best_x = x_new
            best_fx = fx_new
        
        if abs(fx_new) <= 1e-8:
            return x_new
        
        # Stagnation check: if x isn't moving, we are stuck
        if abs(x_new - x) < 1e-14 * max(1.0, abs(x)):
            # Try a small kick to escape local minimum/flat region
            # For polynomials, this is rare. Just break to save time.
            break

        # Shift points
        x_prev = x
        fx_prev = fx
        x = x_new
        fx = fx_new
    
    # Final polish pass with tighter tolerance if we got close
    # Use secant-style iteration for efficiency (1 call per step)
    if abs(best_fx) <= 1e-6 and abs(best_fx) > 1e-8 and call_count < 490:
        x = best_x
        fx = best_fx
        # Initialize secant pair from best point
        h_init = 1e-6 * max(1.0, abs(x))
        x_prev = x - h_init
        try:
            fx_prev = f(x_prev)
            call_count += 1
        except Exception:
            return best_x
        if not _is_finite(fx_prev):
            return best_x
        
        for i in range(40):
            if _time_left() < 1.0:
                break
            if call_count >= 498:
                break
            
            denom = fx - fx_prev
            if abs(denom) < 1e-300:
                break
            
            x_new = x - fx * (x - x_prev) / denom
            
            if not _is_finite(x_new):
                break
            
            try:
                fx_new = f(x_new)
                call_count += 1
            except Exception:
                break
            
            if not _is_finite(fx_new):
                break
            
            if abs(fx_new) < abs(best_fx):
                best_x = x_new
                best_fx = fx_new
            
            if abs(fx_new) <= 1e-8:
                return x_new
            
            # Stagnation check
            if abs(x_new - x) < 1e-14 * max(1.0, abs(x)):
                break
            
            # Shift points
            x_prev = x
            fx_prev = fx
            x = x_new
            fx = fx_new
    
    return best_x
