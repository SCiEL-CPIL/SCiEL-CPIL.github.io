import time
import numpy as np
from scipy.optimize import brentq

def find_root(f, x0):
    start_time = time.time()
    TIME_LIMIT = 140.0  # safety margin before 150s kill
    x0 = float(x0)
    
    # Check initial point
    try:
        f0 = f(x0)
    except Exception:
        f0 = float('inf')
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
        except Exception:
            return float('inf')
    
    # Tiered scanning strategy:
    # Tier 1: Fine scan near x0 (small steps, many points)
    # Tier 2: Coarse scan (larger steps, fewer points)
    # This adapts to oscillatory functions by trying multiple scales
    
    # Define scan tiers: (step_size, num_points_per_direction)
    # Total calls per tier = 2 * num_points
    # Keep total under ~400 to leave room for refinement
    tiers = [
        (1e-3, 50),   # Fine: 100 calls
        (1e-2, 30),   # Medium-fine: 60 calls
        (0.1, 30),    # Medium: 60 calls
        (1.0, 20),    # Coarse: 40 calls
        (10.0, 10),   # Very coarse: 20 calls
        (100.0, 5),   # Extremely coarse: 10 calls
    ]
    
    bracket_found = False
    a, b, fa, fb = None, None, None, None
    
    for step, n_pts in tiers:
        if not time_ok():
            break
        
        for direction in [1, -1]:
            if not time_ok():
                break
            
            prev_x = x0
            prev_f = f0
            
            for i in range(1, n_pts + 1):
                if not time_ok():
                    break
                
                x = x0 + direction * i * step
                fx = safe_f(x)
                
                if not np.isfinite(fx):
                    continue
                
                if abs(fx) < best_f:
                    best_f = abs(fx)
                    best_x = x
                
                if abs(fx) <= 1e-8:
                    return x
                
                # Check for sign change
                if np.isfinite(prev_f) and prev_f * fx < 0:
                    a, b = prev_x, x
                    fa, fb = prev_f, fx
                    bracket_found = True
                    break
                
                prev_x = x
                prev_f = fx
            
            if bracket_found:
                break
        
        if bracket_found:
            break
    
    # If bracket found, use Brent's method for high-precision root
    if bracket_found and time_ok():
        try:
            root = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
            if np.isfinite(root):
                return root
        except Exception:
            pass
    
    # Fallback: Newton's method with safeguarding
    if time_ok():
        x = float(best_x)
        for _ in range(50):
            if not time_ok():
                break
            fx = safe_f(x)
            if not np.isfinite(fx):
                break
            if abs(fx) <= 1e-8:
                return x
            
            # Numerical derivative with adaptive step
            h = 1e-7 * max(1.0, abs(x))
            fxh = safe_f(x + h)
            if not np.isfinite(fxh):
                break
            d = (fxh - fx) / h
            if abs(d) < 1e-15 or not np.isfinite(d):
                break
            
            x_new = x - fx / d
            if not np.isfinite(x_new):
                break
            
            # Damping if step is too large
            if abs(x_new - x) > 10.0 * max(1.0, abs(x)):
                x_new = x + 0.5 * (x_new - x)
            
            x = x_new
        
        if np.isfinite(x) and abs(safe_f(x)) < best_f:
            best_x = x
            best_f = abs(safe_f(x))
    
    return best_x
