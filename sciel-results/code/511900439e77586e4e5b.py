"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import brentq

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    TIME_LIMIT = 140.0  # safety margin before 150s kill
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
    
    def try_brent(a, b):
        """Try Brent's method on bracket [a,b]. Return root or None."""
        try:
            root = brentq(f, a, b, xtol=1e-14, rtol=1e-14, maxiter=60)
            if np.isfinite(root) and abs(f(root)) <= 1e-8:
                return root
        except Exception:
            pass
        return None
    
    def secant_polish(x1, f1, max_iter=30):
        """Secant method polish. Returns (x, f) or None if fails."""
        if not np.isfinite(f1):
            return None
        if abs(f1) <= 1e-8:
            return (x1, f1)
        
        # Need a second point
        h = 1e-4 * max(1.0, abs(x1))
        x2 = x1 + h
        f2 = f(x2)
        if not np.isfinite(f2):
            return None
        
        for _ in range(max_iter):
            if not time_ok():
                break
            if abs(f2) <= 1e-8:
                return (x2, f2)
            if abs(f1) <= 1e-8:
                return (x1, f1)
            
            denom = f2 - f1
            if abs(denom) < 1e-30:
                break
            
            x3 = x2 - f2 * (x2 - x1) / denom
            if not np.isfinite(x3):
                break
            
            # Damping if step too large
            if abs(x3 - x2) > 10 * max(1.0, abs(x2)):
                x3 = x2 + 0.5 * max(1.0, abs(x2)) * np.sign(x3 - x2)
            
            f3 = f(x3)
            if not np.isfinite(f3):
                break
            
            x1, f1 = x2, f2
            x2, f2 = x3, f3
        
        # Return the better of the last points
        if abs(f1) < abs(f2):
            return (x1, f1)
        else:
            return (x2, f2)
    
    # Phase 1: Geometric expansion to find a region with sign changes
    # Start with small step, expand geometrically
    base_h = 0.01 * max(1.0, abs(x0))
    if base_h < 1e-6:
        base_h = 1e-6
    
    bracket_a = None
    bracket_b = None
    bracket_fa = None
    bracket_fb = None
    
    # Try several base scales
    for base_scale in [0.01, 0.1, 0.5, 1.0, 5.0]:
        if not time_ok():
            break
        
        h = base_scale * max(1.0, abs(x0))
        if h < 1e-8:
            h = 1e-8
        
        # Scan in both directions with geometric steps
        for direction in [1, -1]:
            if not time_ok():
                break
            
            prev_x = x0
            prev_f = f0
            
            for k in range(1, 20):  # Up to 20 geometric steps
                if not time_ok():
                    break
                
                x = x0 + direction * h * (2 ** (k - 1))
                fx = f(x)
                
                if not np.isfinite(fx):
                    continue
                
                if abs(fx) < best_f:
                    best_f = abs(fx)
                    best_x = x
                
                if abs(fx) <= 1e-8:
                    return x
                
                # Check for sign change
                if prev_f * fx < 0:
                    bracket_a = prev_x
                    bracket_b = x
                    bracket_fa = prev_f
                    bracket_fb = fx
                    break
                
                prev_x = x
                prev_f = fx
            
            if bracket_a is not None:
                break
        
        if bracket_a is not None:
            break
    
    # Phase 2: If we found a bracket, refine with Brent
    if bracket_a is not None:
        root = try_brent(bracket_a, bracket_b)
        if root is not None:
            return root
        
        # If Brent failed, try secant polish from the better endpoint
        if abs(bracket_fa) < abs(bracket_fb):
            result = secant_polish(bracket_a, bracket_fa)
        else:
            result = secant_polish(bracket_b, bracket_fb)
        
        if result is not None:
            x_res, f_res = result
            if abs(f_res) <= 1e-8:
                return x_res
            if abs(f_res) < best_f:
                best_f = abs(f_res)
                best_x = x_res
    
    # Phase 3: Fine linear scan in a local neighborhood if no bracket found
    if bracket_a is None:
        # Try a fine scan around x0 with small steps
        for step in [1e-4, 1e-3, 1e-2, 0.1]:
            if not time_ok():
                break
            
            for direction in [1, -1]:
                if not time_ok():
                    break
                
                prev_x = x0
                prev_f = f0
                
                for i in range(1, 100):
                    x = x0 + direction * i * step
                    fx = f(x)
                    
                    if not np.isfinite(fx):
                        continue
                    
                    if abs(fx) < best_f:
                        best_f = abs(fx)
                        best_x = x
                    
                    if abs(fx) <= 1e-8:
                        return x
                    
                    if prev_f * fx < 0:
                        root = try_brent(prev_x, x)
                        if root is not None:
                            return root
                        
                        # Secant polish
                        if abs(prev_f) < abs(fx):
                            result = secant_polish(prev_x, prev_f)
                        else:
                            result = secant_polish(x, fx)
                        
                        if result is not None:
                            x_res, f_res = result
                            if abs(f_res) <= 1e-8:
                                return x_res
                            if abs(f_res) < best_f:
                                best_f = abs(f_res)
                                best_x = x_res
                        
                        break
                    
                    prev_x = x
                    prev_f = fx
                
                if not time_ok():
                    break
    
    # Final fallback: secant polish from best point
    result = secant_polish(best_x, f(best_x))
    if result is not None:
        x_res, f_res = result
        if abs(f_res) <= 1e-8:
            return x_res
        if abs(f_res) < best_f:
            best_f = abs(f_res)
            best_x = x_res
    
    return best_x
# EVOLVE-BLOCK-END
