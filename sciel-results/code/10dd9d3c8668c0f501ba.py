"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

DEADLINE = time.time() + 110.0  # Safety margin before 120s limit

def _check_deadline():
    return time.time() < DEADLINE

def find_root(f, x0):
    x = float(x0)
    best_x = x
    best_fx = None
    
    # Phase 1: Fast convergence using Newton with central difference
    for _ in range(40):
        if not _check_deadline():
            break
        if not math.isfinite(x):
            break
        try:
            fx = f(x)
        except:
            break
        if not math.isfinite(fx):
            break
        
        # Track best
        if best_fx is None or abs(fx) < abs(best_fx):
            best_x = x
            best_fx = fx
        
        if abs(fx) <= 1e-10:  # Tighter internal tolerance for polish
            # Phase 2: High-precision polish
            x = _polish(f, x, fx)
            return x
        
        # Central difference for derivative
        h = 1e-5 * max(1.0, abs(x))
        try:
            f_xp = f(x + h)
            f_xm = f(x - h)
        except:
            break
        if not math.isfinite(f_xp) or not math.isfinite(f_xm):
            break
            
        d = (f_xp - f_xm) / (2.0 * h)
        
        if abs(d) < 1e-14:
            # Derivative too small, try secant-like step
            step = -fx * 1e-3
            x_new = x + step
        else:
            x_new = x - fx / d
        
        if not math.isfinite(x_new):
            break
            
        # Damping: prevent large jumps
        step = x_new - x
        if abs(step) > 2.0:
            x_new = x + (2.0 if step > 0 else -2.0)
            
        x = x_new
    
    # If we didn't converge tightly, try secant method as fallback
    if best_fx is not None and abs(best_fx) > 1e-8 and _check_deadline():
        x = _secant_fallback(f, x, best_x, best_fx)
    
    # Final polish if close
    if best_fx is not None and abs(best_fx) <= 1e-6 and _check_deadline():
        x = _polish(f, x, f(x))
    
    return x

def _polish(f, x, fx):
    """High-precision Newton polish with tighter tolerance."""
    for _ in range(20):
        if not _check_deadline():
            break
        if not math.isfinite(x) or fx is None or not math.isfinite(fx):
            break
        if abs(fx) <= 1e-10:
            break
        
        # Use smaller h for higher precision
        h = 1e-7 * max(1.0, abs(x))
        try:
            f_xp = f(x + h)
            f_xm = f(x - h)
        except:
            break
        if not math.isfinite(f_xp) or not math.isfinite(f_xm):
            break
        
        d = (f_xp - f_xm) / (2.0 * h)
        if abs(d) < 1e-16:
            break
        
        x_new = x - fx / d
        if not math.isfinite(x_new):
            break
        
        # Small damping for polish
        step = x_new - x
        if abs(step) > 0.1:
            x_new = x + (0.1 if step > 0 else -0.1)
        
        x = x_new
        try:
            fx = f(x)
        except:
            break
        if not math.isfinite(fx):
            break
    
    return x

def _secant_fallback(f, x, x_prev, fx_prev):
    """Secant method fallback when Newton struggles."""
    for _ in range(30):
        if not _check_deadline():
            break
        if not math.isfinite(x) or not math.isfinite(x_prev):
            break
        try:
            fx = f(x)
        except:
            break
        if not math.isfinite(fx):
            break
        if abs(fx) <= 1e-10:
            return x
        
        if abs(fx) < abs(fx_prev):
            x_prev, fx_prev = x, fx
        elif abs(fx_prev) < 1e-10:
            return x_prev
        
        denom = fx - fx_prev
        if abs(denom) < 1e-15:
            break
        
        x_new = x - fx * (x - x_prev) / denom
        if not math.isfinite(x_new):
            break
        
        x_prev, fx_prev = x, fx
        x = x_new
    
    return x
# EVOLVE-BLOCK-END
