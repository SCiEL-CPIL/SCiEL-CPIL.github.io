"""Root finding for ill-conditioned equations: Brent's Method on f(x)^2 with adaptive bracketing."""
import time
import math
import numpy as np
from scipy.optimize import minimize_scalar

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin below 120s limit
    MAX_EVALS = 490   # Safety margin below 500 limit
    
    eval_count = 0
    best_x = float(x0)
    best_abs_f = float('inf')
    best_g = float('inf')

    def safe_eval(x):
        nonlocal eval_count, best_x, best_abs_f, best_g
        if eval_count >= MAX_EVALS or (time.time() - start_time) > DEADLINE:
            return None
        try:
            val = f(x)
            if not math.isfinite(val):
                return None
            eval_count += 1
            abs_val = abs(val)
            g_val = val * val
            if abs_val < best_abs_f:
                best_abs_f = abs_val
                best_x = x
                best_g = g_val
            return val
        except Exception:
            return None

    def objective(x):
        val = safe_eval(x)
        if val is None:
            return float('inf')
        return val * val

    def check_budget():
        return eval_count >= MAX_EVALS or (time.time() - start_time) > DEADLINE

    # 1. Initial Check
    f0 = safe_eval(float(x0))
    if f0 is not None and abs(f0) <= 1e-8:
        return best_x

    # 2. Adaptive Bracketing
    # We want to find a bracket [a, b] such that g(a) > g(x0) and g(b) > g(x0)
    # This indicates x0 is a local minimum of g(x) = f(x)^2.
    
    scale = max(1.0, abs(float(x0)))
    delta = 1e-3 * scale
    
    a = None
    b = None
    
    # Expand to find a bracket
    # We keep track of the last good points on left and right
    left_x = float(x0)
    left_g = best_g
    right_x = float(x0)
    right_g = best_g
    
    # We need to find points where g is higher than g(x0)
    # If g(x0) is already 0, we returned. So g(x0) > 0.
    
    for i in range(20):
        if check_budget():
            break
            
        # Try left
        if a is None:
            x_trial = left_x - delta
            g_trial = objective(x_trial)
            if g_trial < left_g:
                # Moved down, expand further
                left_x = x_trial
                left_g = g_trial
            else:
                # Found a higher point, bracket left side
                a = x_trial
                # We don't update left_x here, we keep the lower point for the center
                # The bracket is [x_trial, x0] roughly, but we need a 3 point bracket for Brent
                # Actually, for Brent, we can pass bracket=(a, x0, b)
                
        # Try right
        if b is None:
            x_trial = right_x + delta
            g_trial = objective(x_trial)
            if g_trial < right_g:
                # Moved down, expand further
                right_x = x_trial
                right_g = g_trial
            else:
                # Found a higher point, bracket right side
                b = x_trial
        
        if a is not None and b is not None:
            break
            
        # Increase step size
        delta *= 2.0
        
    # If we haven't found a full bracket, we might be on a monotonic slope or near a boundary
    # In that case, use bounded minimization on a larger interval
    
    x_result = best_x
    
    if a is not None and b is not None:
        # Use Brent's method with the 3-point bracket (a, x0, b)
        # Note: scipy's brent method expects bracket=(a, c, b) where c is the center?
        # Docs: "If bracket is provided, it must be a 3-tuple (a, c, b) such that a <= c <= b"
        # and f(c) < f(a) and f(c) < f(b).
        # Our x0 might not be exactly the center, but it is the lowest point we found in the middle.
        # We should ensure x0 is between a and b.
        # a < left_x <= x0 <= right_x < b ? 
        # Actually, a is the point where we stopped expanding left. So a < left_x.
        # And left_x was the lowest point on the left side we found.
        # Similarly for b.
        # The point with the lowest g in the bracket is likely x0 or left_x/right_x.
        # Let's just pass the bracket and let Brent find the minimum.
        # However, if x0 is not between a and b, we need to adjust.
        # Since we expanded from x0, a < x0 < b is generally true.
        
        try:
            res = minimize_scalar(
                objective, 
                method='brent', 
                bracket=(a, float(x0), b),
                options={'xtol': 1e-12, 'maxiter': 100}
            )
            if res.success and res.fun < float('inf'):
                x_result = res.x
                # Evaluate to update best_x
                safe_eval(x_result)
        except Exception:
            pass
    else:
        # No bracket found. Use bounded minimization on a wide interval
        # Centered at best_x
        center = best_x
        width = max(scale * 10.0, 10.0)
        try:
            res = minimize_scalar(
                objective, 
                method='bounded', 
                bounds=(center - width, center + width),
                options={'xtol': 1e-12, 'maxiter': 100}
            )
            if res.success and res.fun < float('inf'):
                x_result = res.x
                safe_eval(x_result)
        except Exception:
            pass

    # 3. Polish with Secant
    # If we have a good candidate, try to tighten it with secant steps
    # We need two points. Let's pick best_x and a nearby point.
    # If we have a and b, we can use them.
    # If not, we can perturb best_x.
    
    # Let's identify two points with low |f|
    # We have best_x. Let's find a second point.
    # If we have a and b, use them.
    # Otherwise, use best_x +/- small_delta
    
    points = []
    if a is not None:
        fa = safe_eval(a)
        if fa is not None:
            points.append((a, fa))
    if b is not None:
        fb = safe_eval(b)
        if fb is not None:
            points.append((b, fb))
    
    # Also add best_x
    fb_best = safe_eval(best_x)
    if fb_best is not None:
        points.append((best_x, fb_best))
    
    # Sort by |f|
    points.sort(key=lambda p: abs(p[1]))
    
    # Take the top 2 distinct points
    unique_points = []
    for x, fx in points:
        if not any(abs(x - ux) < 1e-12 for ux, _ in unique_points):
            unique_points.append((x, fx))
        if len(unique_points) == 2:
            break
            
    if len(unique_points) >= 2:
        x1, f1 = unique_points[0]
        x2, f2 = unique_points[1]
        
        # Secant iteration
        for _ in range(20):
            if check_budget():
                break
            if f1 == f2:
                break
            x_new = x1 - f1 * (x1 - x2) / (f1 - f2)
            
            # Safeguard
            max_jump = max(1.0, 0.5 * abs(x1))
            if abs(x_new - x1) > max_jump:
                x_new = x1 + math.copysign(max_jump, x_new - x1)
                
            f_new = safe_eval(x_new)
            if f_new is None:
                break
                
            if abs(f_new) <= 1e-8:
                return x_new
                
            if abs(f_new) < abs(f1):
                x2, f2 = x1, f1
                x1, f1 = x_new, f_new
            else:
                x2, f2 = x_new, f_new
                
            if abs(x_new - x2) < 1e-14:
                break

    return best_x

# EVOLVE-BLOCK-END
