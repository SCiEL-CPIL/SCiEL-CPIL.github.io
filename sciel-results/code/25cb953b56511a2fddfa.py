"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from scipy.optimize import brentq

def find_root(f, x0):
    TOL = 1e-8
    MAX_CALLS = 480
    START_TIME = time.time()
    DEADLINE = START_TIME + 115
    
    calls = 0
    best_x = float(x0)
    best_abs = float('inf')
    
    def safe_f(x):
        nonlocal calls, best_x, best_abs
        if calls >= MAX_CALLS or time.time() > DEADLINE:
            return float('inf')
        calls += 1
        try:
            val = f(x)
            if not math.isfinite(val):
                return float('inf')
            abs_val = abs(val)
            if abs_val < best_abs:
                best_abs = abs_val
                best_x = x
            return val
        except Exception:
            return float('inf')

    # Check x0
    f0 = safe_f(x0)
    if abs(f0) <= TOL:
        return x0

    # Phase 1: Spectral Period Estimation
    # We assume the function is oscillatory. We probe at a few scales to estimate the period T.
    # We look for sign changes or zero crossings at scales 1, 10, 100, 1000.
    # If we find a sign change between x0 and x0+L, the period is likely <= 2L.
    # We collect candidate periods.
    
    candidate_periods = []
    
    # Probe scales
    probe_scales = [1.0, 10.0, 100.0, 1000.0]
    
    for L in probe_scales:
        if calls >= MAX_CALLS or time.time() > DEADLINE:
            break
            
        # Check f(x0 + L)
        x_probe = x0 + L
        fp = safe_f(x_probe)
        
        if math.isfinite(fp):
            if abs(fp) <= TOL:
                return x_probe
            
            # If sign change, period is roughly in [L/10, 2*L]
            if f0 * fp < 0:
                # Estimate period as ~ L (roughly half a cycle or less)
                candidate_periods.append(L)
                # Also try to refine: check midpoint? No, save calls.
                # But we can try to find a tighter bracket immediately.
                # Let's just record L as a good scale.
        
        # Also check negative direction for the first few scales?
        # To save calls, stick to positive for now, or do both for small L.
        if L <= 10.0:
             x_probe_neg = x0 - L
             fp_neg = safe_f(x_probe_neg)
             if math.isfinite(fp_neg):
                 if abs(fp_neg) <= TOL:
                     return x_probe_neg
                 if f0 * fp_neg < 0:
                     candidate_periods.append(L)

    # Phase 2: Targeted Bracketing
    # If we found candidate periods, use the smallest one to define a dense grid.
    # If not, fall back to a geometric expansion.
    
    found_bracket = False
    a = None
    b = None
    
    if candidate_periods:
        # Use the smallest candidate period as the primary scale
        # This suggests the function oscillates on this scale.
        # We will scan +/- 2 * T_est with a fine step.
        T_est = min(candidate_periods)
        
        # Define a dense grid around x0 within [-2*T_est, 2*T_est]
        # Number of points: ~20 to be safe.
        num_points = 20
        range_size = 4 * T_est
        step = range_size / num_points
        
        prev_x = None
        prev_f = None
        
        for i in range(num_points + 1):
            if time.time() > DEADLINE or calls >= MAX_CALLS:
                break
                
            x = x0 - range_size/2 + i * step
            fx = safe_f(x)
            
            if not math.isfinite(fx):
                prev_x = None
                prev_f = None
                continue
                
            if abs(fx) <= TOL:
                return x
                
            if prev_x is not None and prev_f is not None:
                if prev_f * fx < 0:
                    a = min(prev_x, x)
                    b = max(prev_x, x)
                    found_bracket = True
                    break
            
            prev_x = x
            prev_f = fx
            
    if not found_bracket:
        # Fallback: Geometric expansion if spectral estimate failed or no bracket found in dense grid
        # This is similar to the original but only as a fallback.
        scales = [1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
        for scale in scales:
            if time.time() > DEADLINE or calls >= MAX_CALLS:
                break
            
            # Scan a small window with this scale
            # If we didn't find a bracket in the dense grid, maybe the root is further out
            # or the period estimate was wrong.
            # Try a few points at this scale.
            for k in [-2, -1, 1, 2]:
                if time.time() > DEADLINE or calls >= MAX_CALLS:
                    break
                x = x0 + k * scale
                fx = safe_f(x)
                
                if not math.isfinite(fx):
                    continue
                    
                if abs(fx) <= TOL:
                    return x
                
                # Check bracket against x0 or other points? 
                # This is less efficient. Let's just do a simple linear scan in this scale.
                # Actually, if the dense grid failed, the root might be just outside.
                # Let's just expand the grid search range.
            
            # If we are here, we might need a wider search.
            # Let's do a coarser grid over a larger range.
            num_points_coarse = 10
            range_coarse = 10 * scale
            step_coarse = range_coarse / num_points_coarse
            
            prev_x = None
            prev_f = None
            for i in range(num_points_coarse + 1):
                if time.time() > DEADLINE or calls >= MAX_CALLS:
                    break
                x = x0 - range_coarse/2 + i * step_coarse
                fx = safe_f(x)
                
                if not math.isfinite(fx):
                    prev_x = None
                    prev_f = None
                    continue
                    
                if abs(fx) <= TOL:
                    return x
                    
                if prev_x is not None and prev_f is not None:
                    if prev_f * fx < 0:
                        a = min(prev_x, x)
                        b = max(prev_x, x)
                        found_bracket = True
                        break
                
                prev_x = x
                prev_f = fx
                
            if found_bracket:
                break

    if found_bracket:
        try:
            # Refine using Brent's method
            # Use safe_f to track calls
            root = brentq(safe_f, a, b, maxiter=50, xtol=1e-12, rtol=1e-12)
            if math.isfinite(root):
                fr = safe_f(root)
                if abs(fr) <= TOL * 10:
                    return root
                return best_x
            else:
                return best_x
        except Exception:
            return best_x
    else:
        return best_x

# EVOLVE-BLOCK-END
