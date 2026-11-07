import math
import time
import numpy as np
from scipy.optimize import brentq

def find_root(f, x0):
    TOL = 1e-8
    MAX_CALLS = 495  # Leave 5 calls margin for safety
    START_TIME = time.time()
    DEADLINE = START_TIME + 115.0  # 120s limit, 5s margin
    
    calls = 0
    best_x = float(x0)
    best_f = float('inf')
    
    # Cache to avoid redundant calls
    f_cache = {}
    
    def safe_f(x):
        nonlocal calls, best_x, best_f
        if calls >= MAX_CALLS:
            return float('inf')
        
        # Time check
        if time.time() > DEADLINE:
            return float('inf')
            
        # Use rounding for cache key to handle floating point noise in repeated evaluations
        key = round(float(x), 14)
        
        if key in f_cache:
            val = f_cache[key]
            # Update best even from cache
            if abs(val) < best_f:
                best_f = abs(val)
                best_x = float(x)
            return val

        calls += 1
        try:
            val = f(float(x))
            if not math.isfinite(val):
                f_cache[key] = float('inf')
                return float('inf')
            f_cache[key] = val
        except Exception:
            f_cache[key] = float('inf')
            return float('inf')
        
        if abs(val) < best_f:
            best_f = abs(val)
            best_x = float(x)
            
        return val

    def try_brent(a, b, fa, fb):
        """
        Attempt to find a root in [a, b] where f(a) and f(b) have opposite signs.
        Returns (success, x, f_val).
        """
        if fa * fb > 0:
            return False, None, None
        
        # If one endpoint is already a root
        if abs(fa) <= TOL:
            return True, a, fa
        if abs(fb) <= TOL:
            return True, b, fb
            
        try:
            # maxiter=50 is usually plenty for brentq to converge to machine precision
            # xtol=1e-12 ensures high precision
            x_root = brentq(safe_f, a, b, xtol=1e-12, rtol=1e-12, maxiter=50)
            f_root = safe_f(x_root)
            
            if math.isfinite(x_root) and abs(f_root) <= TOL:
                return True, x_root, f_root
            else:
                # If brentq didn't hit TOL (rare, but possible due to float limits), 
                # check if it's close enough.
                if abs(f_root) <= 1e-7: 
                     # Try one more step of manual bisection if needed? 
                     # Or just accept if it's very close.
                     # Actually, brentq is very reliable. If it fails TOL, it's likely due to f being flat.
                     # Let's check the best point found during the process.
                     # safe_f updates best_x/best_f globally.
                     pass
                     
        except Exception:
            pass
             
        return False, None, None

    x_start = float(x0)
    
    # 1. Check starting point
    fx0 = safe_f(x_start)
    if abs(fx0) <= TOL:
        return x_start

    # 2. Adaptive Sign-Change Scan
    # Strategy: Scan outward from x0 with increasing step sizes.
    # We look for sign changes to bracket roots.
    # For oscillatory functions, roots are likely nearby.
    
    # Define scan parameters
    # Start with a small step, double it each time.
    # Range: from x0 - R to x0 + R, expanding R.
    
    # We will perform a sequence of scans with increasing density/range.
    # Scan 1: Dense local scan
    # Scan 2: Wider, sparser scan
    
    scan_configs = [
        # (num_points, range_multiplier)
        # range_multiplier is relative to max(1, |x0|)
        (50, 0.1),   # 50 points in [-0.1*scale, 0.1*scale]
        (100, 0.5),  # 100 points in [-0.5*scale, 0.5*scale]
        (100, 2.0),  # 100 points in [-2.0*scale, 2.0*scale]
        (100, 10.0), # 100 points in [-10.0*scale, 10.0*scale]
    ]
    
    scale = max(1.0, abs(x_start))
    
    for num_pts, mult in scan_configs:
        if calls > MAX_CALLS - 10:
            break
        if time.time() > DEADLINE - 5.0:
            break
            
        # Generate points
        # Use linspace for even spacing
        lower = x_start - mult * scale
        upper = x_start + mult * scale
        
        # Ensure we don't generate too many points if scale is huge? 
        # num_pts is fixed, so spacing adapts.
        
        # We want to include x0 in the grid to ensure continuity with previous checks
        # But linspace might miss x0 exactly. 
        # Let's construct points manually to ensure x0 is included and spacing is controlled.
        
        # Simple approach: use np.linspace
        points = np.linspace(lower, upper, num_pts)
        
        # Evaluate function at all points
        # We can do this in a batch to check for sign changes
        vals = []
        for p in points:
            if time.time() > DEADLINE - 2.0:
                break
            if calls > MAX_CALLS - 5:
                break
            fp = safe_f(p)
            vals.append((p, fp))
            
        if len(vals) < 2:
            continue
            
        # Check for sign changes
        # Iterate through consecutive pairs
        for i in range(len(vals) - 1):
            x1, f1 = vals[i]
            x2, f2 = vals[i+1]
            
            # If we found a root in previous iterations, we would have returned.
            # Check for exact root first (already done in safe_f updates, but explicit check)
            if abs(f1) <= TOL:
                return x1
            if abs(f2) <= TOL:
                return x2
                
            # Check for sign change
            if f1 * f2 < 0:
                # Bracket found!
                success, x_root, f_root = try_brent(x1, x2, f1, f2)
                if success:
                    return x_root
                # If brentq failed, we continue scanning. 
                # Maybe there are multiple roots in this interval? Unlikely for simple oscillatory,
                # but possible. We'll just let the scan continue.
                
            # Early exit if we've used too many calls
            if calls > MAX_CALLS - 20:
                break
                
    # 3. Fallback: If no sign change found, the function might be tangent to 0 (double root)
    # or we missed it. 
    # In this case, the "best_x" found during the scan is the closest point to 0.
    # We can try a local minimization of f(x)^2 around best_x to see if we can squeeze closer.
    
    if best_f > TOL and calls < MAX_CALLS - 50 and time.time() < DEADLINE - 10.0:
        # Use a very local Nelder-Mead around best_x
        # This is a hybrid fallback
        try:
            from scipy.optimize import minimize
            
            def objective(x):
                return safe_f(x[0]) ** 2
                
            # Small initial simplex
            h = 1e-5 * max(1.0, abs(best_x))
            x0_nm = np.array([best_x])
            
            res = minimize(
                objective, 
                x0_nm, 
                method='Nelder-Mead',
                options={
                    'maxiter': 20,
                    'xatol': 1e-10,
                    'fatol': TOL**2,
                    'adaptive': True
                }
            )
            
            x_res = float(res.x[0])
            f_res = safe_f(x_res)
            
            if math.isfinite(x_res) and abs(f_res) <= TOL:
                return x_res
        except Exception:
            pass

    return best_x
