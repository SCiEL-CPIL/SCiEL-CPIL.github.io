"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from scipy.optimize import brentq

def find_root(f, x0):
    TOL = 1e-8
    MAX_CALLS = 480  # Safe margin under 500
    DEADLINE = time.time() + 115  # Safe margin under 120s limit
    
    calls = 0
    best_x = float(x0)
    best_abs = float('inf')
    
    def safe_f(x):
        nonlocal calls, best_x, best_abs
        if calls >= MAX_CALLS:
            return float('inf')
        if time.time() > DEADLINE:
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

    # Initial check
    f0 = safe_f(x0)
    if abs(f0) <= TOL:
        return x0

    # Multi-scale grid search to find a bracket [a, b] where f(a)*f(b) < 0
    # Start with a wide range around x0, then narrow down if needed
    # We use a logarithmic or geometric progression of step sizes to cover range efficiently
    # However, for oscillatory functions, a uniform grid with decreasing step size is safer
    # to catch sign changes.
    
    # Strategy:
    # 1. Try a few coarse steps to see if the root is nearby.
    # 2. If not, do a wider scan.
    # 3. Once a sign change is found, use brentq.
    
    # Let's try a sequence of step sizes.
    # We want to find a bracket.
    # Start with small steps around x0, then expand.
    
    base_step = 1e-3
    # Try steps: 1e-3, 1e-2, 1e-1, 1, 10, 100
    # But we also need to check in both directions.
    
    # To save calls, we can scan a grid.
    # Let's define a set of candidate intervals to scan.
    
    found_bracket = False
    a = None
    b = None
    
    # First, check immediate neighbors with a fine grid
    # This helps if x0 is very close to the root.
    steps_to_try = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
    
    # We will scan in a spiral-like or expanding box manner.
    # Actually, a simpler approach:
    # Scan a grid of points around x0.
    # If we find f(x_i) * f(x_{i+1}) < 0, we have a bracket.
    
    # Let's try a sequence of grids.
    # Grid 1: Fine, small range
    # Grid 2: Coarser, larger range
    
    # To be efficient, we can just iterate through a predefined list of (center, step, num_points)
    # But x0 is the center.
    
    # Let's try scanning with step h, from x0 - N*h to x0 + N*h
    # We need to be careful with call counts.
    
    # Approach:
    # 1. Try brentq with small brackets around x0.
    # 2. If that fails, expand the search.
    
    # Let's try a direct grid scan with adaptive step.
    
    # We'll maintain a list of points evaluated.
    # But safe_f already tracks best_x.
    
    # Let's try to find a sign change by stepping outwards.
    
    # Step 1: Check immediate vicinity with small steps
    # We'll do a binary search-like expansion or just a linear scan.
    # Linear scan is safer for oscillatory functions.
    
    # Let's try a few scales.
    scales = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]
    
    for scale in scales:
        if time.time() > DEADLINE or calls >= MAX_CALLS:
            break
            
        # We scan from x0 - 5*scale to x0 + 5*scale with step scale/10?
        # No, let's just take points at x0 + k*scale for k in -5..5
        # That's 11 points per scale. 9 scales = 99 points.
        # This is well within budget.
        
        prev_x = None
        prev_f = None
        
        for k in range(-5, 6):
            if time.time() > DEADLINE or calls >= MAX_CALLS:
                break
                
            x = x0 + k * scale
            fx = safe_f(x)
            
            if not math.isfinite(fx):
                prev_x = None
                prev_f = None
                continue
                
            if abs(fx) <= TOL:
                return x
                
            if prev_x is not None and prev_f is not None:
                if prev_f * fx < 0:
                    # Sign change found!
                    a = min(prev_x, x)
                    b = max(prev_x, x)
                    found_bracket = True
                    break
            else:
                # If this is the first point or we reset, just store
                if prev_x is None:
                    prev_x = x
                    prev_f = fx
                # If we had a prev but it was inf, we reset prev_x/prev_f to None in the inf block above
                # Wait, the logic above:
                # if not finite: prev_x=None, prev_f=None
                # else: 
                #   if prev_x is not None: check bracket
                #   else: set prev_x, prev_f
                # This works.
            
            # Update prev if we didn't break
            # Wait, if we don't break, we should update prev for the next iteration?
            # Yes.
            prev_x = x
            prev_f = fx
            
        if found_bracket:
            break
            
    if found_bracket:
        try:
            # Use brentq to refine
            # maxiter should be low to save time/calls, but brentq is efficient.
            # However, brentq calls f itself. We need to ensure we don't exceed MAX_CALLS.
            # brentq doesn't have a direct call limit, but it converges fast.
            # To be safe, we can implement a simple bisection if we are worried about calls,
            # but brentq is superlinear.
            # Let's use brentq with maxiter=50.
            root = brentq(safe_f, a, b, maxiter=50, xtol=1e-12, rtol=1e-12)
            # Check if it's valid
            if math.isfinite(root):
                # One final check to be sure
                fr = safe_f(root)
                if abs(fr) <= TOL * 10: # Allow slight tolerance for final check
                    return root
                # If not, return the best we found
                return best_x
            else:
                return best_x
        except Exception:
            return best_x
    else:
        # No bracket found. Return the best x found so far.
        return best_x

# EVOLVE-BLOCK-END
