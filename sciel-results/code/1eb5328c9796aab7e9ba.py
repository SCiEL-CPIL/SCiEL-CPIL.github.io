import math
import time
import numpy as np
from scipy.optimize import brentq

def find_root(f, x0):
    x0 = float(x0)
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin for 3 equations + overhead
    
    call_count = 0
    max_calls = 500
    
    def f_counted(x):
        nonlocal call_count
        if call_count >= max_calls:
            raise Exception("Max calls exceeded")
        call_count += 1
        val = f(x)
        return val
    
    # Check if x0 is already a root
    fx0 = f_counted(x0)
    if abs(fx0) <= 1e-8:
        return x0
    
    # Strategy: Global search via logarithmic-space grid to find sign changes,
    # then Brent's method for refinement.
    
    # Build a grid of candidate points in log-space around x0
    # This handles roots at various scales efficiently.
    
    # Determine a base scale
    base = max(abs(x0), 1.0)
    
    # Generate points: for each order of magnitude, sample a few points
    # We'll use points at: x0 * 10^k for k in range, plus x0 + offset
    # Actually, let's use a mixed approach:
    # 1. Points near x0: x0 + delta for small deltas
    # 2. Points at various scales: x0 * 10^k and -x0 * 10^k
    
    candidate_points = []
    
    # Near x0
    for delta in [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]:
        candidate_points.append(x0 + delta)
        candidate_points.append(x0 - delta)
    
    # Scale-based points
    for k in range(-4, 5):
        scale = base * (10.0 ** k)
        candidate_points.append(scale)
        candidate_points.append(-scale)
        candidate_points.append(x0 + scale)
        candidate_points.append(x0 - scale)
    
    # Remove duplicates and non-finite values
    seen = set()
    unique_points = []
    for p in candidate_points:
        if not math.isfinite(p):
            continue
        key = round(p, 10)
        if key not in seen:
            seen.add(key)
            unique_points.append(p)
    
    # Sort by distance from x0 to prioritize nearby points
    unique_points.sort(key=lambda p: abs(p - x0))
    
    # Evaluate points and look for sign changes
    prev_x = x0
    prev_f = fx0
    bracket = None
    
    for p in unique_points:
        if time.time() > deadline or call_count >= max_calls - 5:
            break
        
        try:
            fp = f_counted(p)
        except Exception:
            continue
        
        if abs(fp) <= 1e-8:
            return p
        
        # Check for sign change between prev_x and p
        if prev_f * fp < 0:
            bracket = (prev_x, p, prev_f, fp)
            break
        
        prev_x = p
        prev_f = fp
    
    # If we found a bracket, use Brent's method
    if bracket:
        try:
            root = brentq(f_counted, bracket[0], bracket[1], xtol=1e-12, rtol=1e-12, maxiter=100)
            return root
        except Exception:
            pass
    
    # If no bracket found, try a more targeted search
    # Use secant method from the two best points
    if call_count < max_calls - 1:
        # Pick two points with smallest |f| values
        # We need to track which points we've evaluated
        # Let's just try secant from x0 and the closest point we evaluated
        if unique_points:
            x1 = unique_points[0]
            if x1 != x0:
                try:
                    fx1 = f_counted(x1)
                except Exception:
                    fx1 = fx0
                
                # Secant method
                x_prev, x_curr = x0, x1
                f_prev, f_curr = fx0, fx1
                
                for _ in range(30):
                    if time.time() > deadline or call_count >= max_calls:
                        break
                    
                    if abs(f_curr) <= 1e-8:
                        return x_curr
                    if abs(f_prev) <= 1e-8:
                        return x_prev
                    
                    denom = f_curr - f_prev
                    if abs(denom) < 1e-15:
                        break
                    
                    x_new = x_curr - f_curr * (x_curr - x_prev) / denom
                    
                    if not math.isfinite(x_new):
                        break
                    
                    # Damping to prevent divergence
                    max_step = 10 * max(1.0, abs(x_curr))
                    if abs(x_new - x_curr) > max_step:
                        x_new = x_curr + max_step * (1 if x_new > x_curr else -1)
                    
                    x_prev, f_prev = x_curr, f_curr
                    x_curr = x_new
                    
                    try:
                        f_curr = f_counted(x_curr)
                    except Exception:
                        break
                    
                    if abs(f_curr) <= 1e-8:
                        return x_curr
                
                # Return the best estimate
                if abs(f_curr) < abs(f_prev):
                    return x_curr
                else:
                    return x_prev
    
    # Fallback: return x0
    return x0
