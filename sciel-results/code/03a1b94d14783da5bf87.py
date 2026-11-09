import math
import time
from scipy.optimize import brentq

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    x = float(x0)
    best_x = x
    best_fx = f(x)
    best_abs_fx = abs(best_fx)
    
    if best_abs_fx <= 1e-8:
        return best_x

    history = [(x, best_fx)]
    
    # Phase 1: Try Newton's method
    # Use a limited number of iterations
    for i in range(10):
        if time.time() > deadline:
            break
            
        # Check for sign change in history
        # This is O(N) but N is small (10)
        # Actually, we can just check if the last step crossed a root
        if len(history) >= 2:
            prev_x, prev_fx = history[-2]
            curr_x, curr_fx = history[-1]
            if prev_fx * curr_fx <= 0:
                # Bracket found in Newton steps
                try:
                    root = brentq(f, prev_x, curr_x, xtol=1e-12, rtol=1e-12, maxiter=50)
                    if math.isfinite(root) and abs(f(root)) <= 1e-8:
                        return root
                except Exception:
                    pass

        fx = best_fx
        h = 1e-6 * max(1.0, abs(x))
        try:
            d = (f(x + h) - fx) / h
        except Exception:
            break
            
        if not math.isfinite(d) or abs(d) < 1e-15:
            break
            
        x_new = x - fx / d
        
        if not math.isfinite(x_new):
            break
            
        # Check if step is reasonable
        if abs(x_new - x) > 10 * max(1.0, abs(x)):
            break
            
        fx_new = f(x_new)
        
        if not math.isfinite(fx_new):
            break
            
        if abs(fx_new) < best_abs_fx:
            best_x = x_new
            best_fx = fx_new
            best_abs_fx = abs(fx_new)
            
        if best_abs_fx <= 1e-8:
            return best_x
            
        history.append((x_new, fx_new))
        x = x_new
        
        # If we are moving away from the root in terms of function value, stop Newton
        if i > 2 and abs(fx_new) > abs(fx):
             # Heuristic: if it's getting worse, stop
             # But we already appended to history.
             pass

    # Phase 2: Bracketing from best_x or x0
    # If we found a sign change in history, we already tried brentq.
    # If not, we need to expand.
    
    # Try to find a bracket around best_x
    center = best_x
    step = 0.1
    max_step = 1e6 * max(1.0, abs(center))
    
    # Check immediate neighbors first
    a = center - step
    b = center + step
    fa = f(a)
    fb = f(b)
    
    while fa * fb > 0 and step < max_step:
        if time.time() > deadline:
            return best_x
            
        step *= 2.0
        a = center - step
        b = center + step
        fa = f(a)
        fb = f(b)
        
        # Optional: check if we crossed a root in the expansion
        # Not strictly necessary if we just want a bracket
        # But if fa and fb have same sign, we didn't cross.
        
    if fa * fb <= 0:
        try:
            root = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
            if math.isfinite(root) and abs(f(root)) <= 1e-8:
                return root
        except Exception:
            pass
    else:
        # If no bracket found, return best guess
        return best_x

    return best_x
