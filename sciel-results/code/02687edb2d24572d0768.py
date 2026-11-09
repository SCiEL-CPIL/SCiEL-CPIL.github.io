import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    # 120s limit, keep 5s margin
    deadline = start_time + 115.0
    
    tol = 1e-8
    best_x = float(x0)
    try:
        best_fx = f(best_x)
    except Exception:
        return best_x
        
    if not math.isfinite(best_fx):
        return best_x
        
    best_abs = abs(best_fx)
    
    if best_abs <= tol:
        return best_x

    x = float(x0)
    fx = best_fx
    
    # Variables for Secant method
    x_s0 = None
    fx_s0 = None
    x_s1 = None
    fx_s1 = None

    # Phase 1: Newton's Method
    # Use central difference for derivative
    for i in range(20):
        if time.time() > deadline:
            break
            
        if abs(fx) <= tol:
            return x
            
        h = 1e-5 * max(1.0, abs(x))
        try:
            fx_plus = f(x + h)
            fx_minus = f(x - h)
        except Exception:
            break
            
        # Update best with perturbed points
        if math.isfinite(fx_plus) and abs(fx_plus) < best_abs:
            best_abs = abs(fx_plus)
            best_x = x + h
        if math.isfinite(fx_minus) and abs(fx_minus) < best_abs:
            best_abs = abs(fx_minus)
            best_x = x - h
            
        if not (math.isfinite(fx_plus) and math.isfinite(fx_minus)):
            break
            
        d = (fx_plus - fx_minus) / (2 * h)
        
        if abs(d) < 1e-12:
            break
            
        x_new = x - fx / d
        
        if not math.isfinite(x_new):
            break
            
        try:
            fx_new = f(x_new)
        except Exception:
            break
            
        if not math.isfinite(fx_new):
            break
            
        if abs(fx_new) < best_abs:
            best_abs = abs(fx_new)
            best_x = x_new
            
        if abs(fx_new) <= tol:
            return x_new
            
        # Check for divergence
        if abs(x_new - x) > 1e6 * max(1.0, abs(x)) or abs(fx_new) > 10 * abs(fx):
            # Diverging, switch to Secant
            x_s0 = x
            fx_s0 = fx
            x_s1 = x_new
            fx_s1 = fx_new
            break
        else:
            # Continue Newton, store points for Secant
            x_s0 = x
            fx_s0 = fx
            x_s1 = x_new
            fx_s1 = fx_new
            x = x_new
            fx = fx_new

    # Phase 2: Secant Method
    if x_s0 is None or x_s1 is None:
        # Initialize Secant points if Newton didn't run or failed immediately
        h_init = 1e-4 * max(1.0, abs(x0))
        x_s0 = float(x0)
        fx_s0 = best_fx
        x_s1 = x0 + h_init
        try:
            fx_s1 = f(x_s1)
        except Exception:
            return best_x
            
        if math.isfinite(fx_s1) and abs(fx_s1) < best_abs:
            best_abs = abs(fx_s1)
            best_x = x_s1
        if abs(fx_s1) <= tol:
            return x_s1

    x0_s = x_s0
    x1_s = x_s1
    fx0_s = fx_s0
    fx1_s = fx_s1

    for i in range(50):
        if time.time() > deadline:
            break
            
        if abs(fx1_s) <= tol:
            return x1_s
        if abs(fx0_s) <= tol:
            return x0_s
            
        if fx1_s == fx0_s:
            break
            
        x_new = x1_s - fx1_s * (x1_s - x0_s) / (fx1_s - fx0_s)
        
        if not math.isfinite(x_new):
            break
            
        try:
            fx_new = f(x_new)
        except Exception:
            break
            
        if not math.isfinite(fx_new):
            break
            
        if abs(fx_new) < best_abs:
            best_abs = abs(fx_new)
            best_x = x_new
            
        if abs(fx_new) <= tol:
            return x_new
            
        x0_s = x1_s
        fx0_s = fx1_s
        x1_s = x_new
        fx1_s = fx_new

    return best_x
# EVOLVE-BLOCK-END
