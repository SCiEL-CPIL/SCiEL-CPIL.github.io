"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import brentq

# EVOLVE-BLOCK-START

def find_root(f, x0):
    """
    Robust root-finding for potentially ill-conditioned problems.
    
    Strategy:
      1. Robust Secant Method with Damping (derivative-free, robust to noise)
      2. Aggressive bracketing search with exponential expansion
      3. Brent's method when a bracket is found (guaranteed convergence)
      4. Keep track of best x found (min |f(x)|)
      5. Deadline guard: stop if we approach 145s (safety margin before 150s kill)
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5 second safety margin
    
    x0 = float(x0)
    eval_count = 0
    max_evals = 500
    
    def safe_f(x):
        nonlocal eval_count
        if eval_count >= max_evals:
            return None
        eval_count += 1
        try:
            val = f(x)
            if not np.isfinite(val):
                return None
            return float(val)
        except:
            return None
    
    def check_deadline():
        return time.time() >= deadline
    
    # Initialize
    fx0 = safe_f(x0)
    if fx0 is None:
        return x0
    if abs(fx0) <= 1e-8:
        return x0
    
    best_x = x0
    best_f = fx0
    best_abs_f = abs(fx0)
    
    # Phase 1: Robust Secant Method with Damping
    # Use two points to start secant
    h = 1e-4 * max(1.0, abs(x0))
    x1 = x0 + h
    fx1 = safe_f(x1)
    
    if fx1 is None:
        x1 = x0 - h
        fx1 = safe_f(x1)
    
    if fx1 is None:
        # Fallback: try central difference approach
        x_m = x0 - h
        x_p = x0 + h
        fm = safe_f(x_m)
        fp = safe_f(x_p)
        if fm is not None and fp is not None:
            d = (fp - fm) / (2 * h)
            if d != 0:
                x1 = x0 - fx0 / d
                fx1 = safe_f(x1)
            else:
                fx1 = None
        if fx1 is None:
            return best_x
    
    # Track history for secant
    x_prev = x0
    fx_prev = fx0
    x_curr = x1
    fx_curr = fx1
    
    for i in range(80):
        if check_deadline() or eval_count >= max_evals:
            break
        if fx_curr is None:
            break
        if abs(fx_curr) < best_abs_f:
            best_x = x_curr
            best_f = fx_curr
            best_abs_f = abs(fx_curr)
        if abs(fx_curr) <= 1e-8:
            return x_curr
        
        # Secant step
        if fx_curr == fx_prev:
            # Derivative is zero or parallel, perturb
            x_curr = x_curr + 1e-4
            fx_curr = safe_f(x_curr)
            continue
        
        x_next = x_curr - fx_curr * (x_curr - x_prev) / (fx_curr - fx_prev)
        
        # Damping: if step is too large or doesn't improve, damp it
        step = x_next - x_curr
        max_step = 0.5 * abs(x_curr) + 1.0
        if abs(step) > max_step:
            x_next = x_curr + max_step * math.copysign(1, step)
        
        fx_next = safe_f(x_next)
        
        if fx_next is None:
            # Try smaller step
            x_next = x_curr + 0.5 * step
            fx_next = safe_f(x_next)
            if fx_next is None:
                break
        
        # Check for improvement
        if abs(fx_next) < abs(fx_curr):
            x_prev = x_curr
            fx_prev = fx_curr
            x_curr = x_next
            fx_curr = fx_next
        else:
            # Damp the step
            x_damped = x_curr + 0.5 * (x_next - x_curr)
            fx_damped = safe_f(x_damped)
            if fx_damped is not None and abs(fx_damped) < abs(fx_curr):
                x_prev = x_curr
                fx_prev = fx_curr
                x_curr = x_damped
                fx_curr = fx_damped
            else:
                # If no improvement, break to try bracketing
                break
    
    if abs(best_f) <= 1e-8:
        return best_x
    
    # Phase 2: Aggressive Bracketing
    # Try to find a bracket [a, b] with sign change
    # Start around best_x
    a = best_x
    fa = safe_f(a)
    if fa is None:
        return best_x
    
    # Expand interval exponentially in both directions
    r = 1e-4 * max(1.0, abs(a))
    found_bracket = False
    b = a
    fb = None
    
    for i in range(40):
        if check_deadline() or eval_count >= max_evals - 10:
            break
            
        # Try expanding to the right
        b = a + r
        fb = safe_f(b)
        if fb is not None:
            if fa * fb < 0:
                found_bracket = True
                break
        
        # Try expanding to the left
        b = a - r
        fb = safe_f(b)
        if fb is not None:
            if fa * fb < 0:
                found_bracket = True
                break
        
        r *= 2.0
    
    if found_bracket:
        # Phase 3: Brent's Method
        try:
            # Wrap f to count evaluations and check deadline
            def counted_f(x_val):
                nonlocal eval_count
                if check_deadline() or eval_count >= max_evals:
                    raise ValueError("Deadline exceeded")
                fx_val = f(x_val)
                eval_count += 1
                if abs(fx_val) < best_abs_f:
                    nonlocal best_x, best_f, best_abs_f
                    best_x = x_val
                    best_f = fx_val
                    best_abs_f = abs(fx_val)
                if abs(fx_val) <= 1e-8:
                    raise ValueError("Root found")
                return fx_val
            
            root = brentq(counted_f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
            if root is not None:
                fr = safe_f(root)
                if fr is not None and abs(fr) < abs(best_f):
                    return root
                if fr is not None and abs(fr) <= 1e-8:
                    return root
        except ValueError:
            pass
        except Exception:
            pass
    
    # Phase 4: If no bracket found, do a more extensive local search
    # Try multiple starting points around best_x with secant
    if abs(best_f) > 1e-8:
        for offset_factor in [0.1, 0.01, 0.001, -0.1, -0.01, -0.001]:
            if check_deadline() or eval_count >= max_evals - 20:
                break
            
            x_start = best_x + offset_factor * max(1.0, abs(best_x))
            fx_start = safe_f(x_start)
            if fx_start is None:
                continue
            
            if abs(fx_start) < best_abs_f:
                best_x = x_start
                best_f = fx_start
                best_abs_f = abs(fx_start)
            if abs(fx_start) <= 1e-8:
                return x_start
            
            # Try secant from this new starting point
            h2 = 1e-5 * max(1.0, abs(x_start))
            x_s1 = x_start + h2
            fx_s1 = safe_f(x_s1)
            if fx_s1 is None:
                x_s1 = x_start - h2
                fx_s1 = safe_f(x_s1)
            
            if fx_s1 is None:
                continue
            
            x_sp = x_start
            fx_sp = fx_start
            x_sc = x_s1
            fx_sc = fx_s1
            
            for j in range(20):
                if check_deadline() or eval_count >= max_evals:
                    break
                if fx_sc is None:
                    break
                if abs(fx_sc) < best_abs_f:
                    best_x = x_sc
                    best_f = fx_sc
                    best_abs_f = abs(fx_sc)
                if abs(fx_sc) <= 1e-8:
                    return x_sc
                
                if fx_sc == fx_sp:
                    break
                
                x_sn = x_sc - fx_sc * (x_sc - x_sp) / (fx_sc - fx_sp)
                step_s = x_sn - x_sc
                max_step_s = 0.5 * abs(x_sc) + 1.0
                if abs(step_s) > max_step_s:
                    x_sn = x_sc + max_step_s * math.copysign(1, step_s)
                
                fx_sn = safe_f(x_sn)
                if fx_sn is None:
                    break
                
                if abs(fx_sn) < abs(fx_sc):
                    x_sp = x_sc
                    fx_sp = fx_sc
                    x_sc = x_sn
                    fx_sc = fx_sn
                else:
                    break
    
    return best_x

# EVOLVE-BLOCK-END
