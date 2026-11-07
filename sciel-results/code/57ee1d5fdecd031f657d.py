"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import brentq

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    TIME_LIMIT = 145.0  # safety margin before 150s kill
    x0 = float(x0)
    
    # Check initial point
    f0 = f(x0)
    if not np.isfinite(f0):
        f0 = float('inf')
    if abs(f0) <= 1e-8:
        return x0
    
    best_x = x0
    best_f = abs(f0)
    
    def time_ok():
        return (time.time() - start_time) < TIME_LIMIT
    
    def safe_f(x):
        try:
            val = f(x)
            if not np.isfinite(val):
                return float('inf')
            return float(val)
        except:
            return float('inf')
    
    # Multi-scale sign-change scan with adaptive spacing
    # Try various scales to bracket roots
    scales = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    brackets = []
    
    for scale in scales:
        if not time_ok():
            break
        
        # Scan in both directions from x0
        for direction in [1, -1]:
            if not time_ok():
                break
            
            # Evaluate points spaced by 'scale' on both sides
            # Start close to x0 and expand
            n_points = 50
            for i in range(1, n_points + 1):
                x1 = x0 + direction * i * scale
                x2 = x0 + direction * (i + 1) * scale
                
                f1 = safe_f(x1)
                f2 = safe_f(x2)
                
                if abs(f1) < best_f:
                    best_f = abs(f1)
                    best_x = x1
                if abs(f2) < best_f:
                    best_f = abs(f2)
                    best_x = x2
                
                # Check for sign change
                if np.isfinite(f1) and np.isfinite(f2) and f1 * f2 < 0:
                    brackets.append((x1, x2))
                    if len(brackets) >= 5:
                        break
                
                if not time_ok():
                    break
            
            if len(brackets) >= 5:
                break
        
        if len(brackets) >= 5:
            break
    
    # If we found brackets, use brentq to refine
    if brackets:
        for a, b in brackets:
            if not time_ok():
                break
            try:
                root = brentq(safe_f, a, b, xtol=1e-12, rtol=1e-14, maxiter=100)
                f_root = safe_f(root)
                if np.isfinite(f_root) and abs(f_root) < best_f:
                    best_f = abs(f_root)
                    best_x = root
                if abs(f_root) <= 1e-8:
                    return root
            except:
                continue
    
    # If no good bracket found, try Newton's method from x0
    if best_f > 1e-8:
        x = x0
        for _ in range(30):
            if not time_ok():
                break
            fx = safe_f(x)
            if abs(fx) < best_f:
                best_f = abs(fx)
                best_x = x
            if abs(fx) <= 1e-8:
                return x
            h = 1e-6 * max(1.0, abs(x))
            d = (safe_f(x + h) - fx) / h
            if d == 0 or not np.isfinite(d):
                break
            x_new = x - fx / d
            if not np.isfinite(x_new):
                break
            x = x_new
    
    # Final check
    if best_f <= 1e-8:
        return best_x
    
    # If all else fails, try a broader scan with more points
    if time_ok():
        # Coarse scan over a wider range
        for scale in [0.001, 0.01, 0.1, 1.0, 10.0]:
            if not time_ok():
                break
            for direction in [1, -1]:
                if not time_ok():
                    break
                prev_x = x0
                prev_f = f0
                for i in range(1, 200):
                    x_new = x0 + direction * i * scale
                    f_new = safe_f(x_new)
                    
                    if abs(f_new) < best_f:
                        best_f = abs(f_new)
                        best_x = x_new
                    
                    if np.isfinite(prev_f) and np.isfinite(f_new) and prev_f * f_new < 0:
                        try:
                            root = brentq(safe_f, prev_x, x_new, xtol=1e-12, rtol=1e-14, maxiter=100)
                            f_root = safe_f(root)
                            if np.isfinite(f_root) and abs(f_root) < best_f:
                                best_f = abs(f_root)
                                best_x = root
                            if abs(f_root) <= 1e-8:
                                return root
                        except:
                            pass
                    
                    prev_x = x_new
                    prev_f = f_new
                    
                    if not time_ok():
                        break
    
    return best_x
# EVOLVE-BLOCK-END
