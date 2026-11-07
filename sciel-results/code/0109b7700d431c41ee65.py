"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import brentq

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    TIME_LIMIT = 140.0  # Safety margin before 150s kill
    
    def time_ok():
        return (time.time() - start_time) < TIME_LIMIT

    # Global cache to avoid redundant evaluations
    cache = {}
    call_count = 0
    MAX_CALLS = 450  # Hard limit to ensure we stay under 500

    def safe_eval(x):
        nonlocal call_count
        if not np.isfinite(x):
            return float('inf')
        
        # Check cache
        # Use a tuple key to handle float precision issues if necessary, 
        # but simple float key is usually fine for this task.
        key = x
        if key in cache:
            return cache[key]
        
        if call_count >= MAX_CALLS:
            return float('nan') # Signal to stop searching if budget exceeded
            
        try:
            val = f(x)
            if not np.isfinite(val):
                cache[key] = float('inf')
            else:
                cache[key] = float(val)
            call_count += 1
            return cache[key]
        except Exception:
            cache[key] = float('inf')
            call_count += 1
            return float('inf')

    x0 = float(x0)
    
    # 1. Check initial point
    f0 = safe_eval(x0)
    if abs(f0) <= 1e-8:
        return x0

    best_x = x0
    best_f = abs(f0)

    def update_best(x, fx):
        nonlocal best_x, best_f
        if np.isfinite(fx) and abs(fx) < best_f:
            best_x = x
            best_f = abs(fx)
            if best_f <= 1e-8:
                return True
        return False

    # 2. Local Bracket Search
    # Try small steps to find a sign change close to x0
    steps = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
    
    for step in steps:
        if not time_ok() or call_count > MAX_CALLS - 50:
            break
            
        # Check right
        x_r = x0 + step
        fr = safe_eval(x_r)
        if update_best(x_r, fr):
            return x_r
            
        # Check left
        x_l = x0 - step
        fl = safe_eval(x_l)
        if update_best(x_l, fl):
            return x_l
            
        # Check sign change (x0, x_r)
        if np.isfinite(f0) and np.isfinite(fr) and f0 * fr < 0:
            try:
                root = brentq(safe_eval, x0, x_r, xtol=1e-12, rtol=1e-12, maxiter=100)
                if np.isfinite(root):
                    froot = safe_eval(root)
                    if abs(froot) <= 1e-8:
                        return root
                    update_best(root, froot)
            except Exception:
                pass
        
        # Check sign change (x_l, x0)
        if np.isfinite(fl) and np.isfinite(f0) and fl * f0 < 0:
            try:
                root = brentq(safe_eval, x_l, x0, xtol=1e-12, rtol=1e-12, maxiter=100)
                if np.isfinite(root):
                    froot = safe_eval(root)
                    if abs(froot) <= 1e-8:
                        return root
                    update_best(root, froot)
            except Exception:
                pass

    # 3. Newton's Method with Fallback
    # If no bracket found locally, try Newton. Oscillatory functions are smooth.
    x = x0
    # Use a slightly larger step for derivative if x0 is large
    h = 1e-6 * max(1.0, abs(x))
    
    for _ in range(15):
        if not time_ok() or call_count > MAX_CALLS - 20:
            break
            
        fx = safe_eval(x)
        if not np.isfinite(fx):
            break
        if update_best(x, fx):
            return x
            
        # Central difference derivative
        fx_plus = safe_eval(x + h)
        fx_minus = safe_eval(x - h)
        
        if not np.isfinite(fx_plus) or not np.isfinite(fx_minus):
            break
            
        d = (fx_plus - fx_minus) / (2 * h)
        
        if d == 0 or not np.isfinite(d):
            break
            
        x_new = x - fx / d
        
        if not np.isfinite(x_new):
            break
            
        # Damping: if step is too large, reduce it
        step_size = abs(x_new - x)
        max_step = 10 * max(1.0, abs(x))
        if step_size > max_step:
            x_new = x + 0.5 * (x_new - x)
            
        x = x_new
        
        # If we moved to a new point, check it
        fx_new = safe_eval(x)
        if update_best(x, fx_new):
            return x
            
        # If we have a bracket from previous points, we could use it, 
        # but Newton usually converges fast. 
        # Check if we crossed a zero?
        # We don't easily track the previous sign here without extra storage.
        # But if |fx_new| is small, we return.

    # 4. Global Scan Fallback
    # If Newton failed, scan a wider area for a sign change.
    # Use a step size that is likely to catch the period of oscillation.
    # Common periods are pi (~3.14), 2pi (~6.28). A step of 1.0 is safe.
    scan_step = 1.0
    # Scan in a window around x0
    # We already checked +/- 10 in local search. 
    # Let's scan +/- 50 with step 1.0.
    
    # Generate points to check, avoiding duplicates
    points_to_check = []
    for i in range(-50, 51):
        if i == 0:
            continue
        xi = x0 + i * scan_step
        points_to_check.append(xi)
        
    # Evaluate and check for sign changes between consecutive points
    prev_x = None
    prev_f = None
    
    for xi in points_to_check:
        if not time_ok() or call_count > MAX_CALLS - 10:
            break
            
        fi = safe_eval(xi)
        if update_best(xi, fi):
            return xi
            
        if prev_x is not None and np.isfinite(prev_f) and np.isfinite(fi):
            if prev_f * fi < 0:
                try:
                    root = brentq(safe_eval, prev_x, xi, xtol=1e-12, rtol=1e-12, maxiter=100)
                    if np.isfinite(root):
                        froot = safe_eval(root)
                        if abs(froot) <= 1e-8:
                            return root
                        update_best(root, froot)
                except Exception:
                    pass
                    
        prev_x = xi
        prev_f = fi

    # 5. Final Fallback: Return best found
    return best_x
# EVOLVE-BLOCK-END
