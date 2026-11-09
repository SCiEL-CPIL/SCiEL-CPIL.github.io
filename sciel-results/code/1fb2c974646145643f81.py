import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Robust root finder using a Global-First Hybrid Strategy.
    
    Strategy:
    1. Global Coarse Scan: Evaluate |f(x)| on a grid around x0 (and a wider 
       range if needed) to find the point with the smallest |f(x)|. 
       This avoids getting trapped in local minima of |f(x)| (local extrema 
       of f) which are not roots.
    2. Local Refinement: Once the best candidate is found, use a bracketing 
       method (Illinois/Regula Falsi hybrid) to converge to |f(x)| <= 1e-8.
       Bracketing is very fast (O(log(1/eps)) calls) and robust for smooth 
       oscillatory functions if a sign change is found. If no sign change 
       is found in the immediate neighborhood, we fall back to minimizing 
       |f(x)| locally using a simple 3-point search or just return the 
       best found if it's already close enough.
    
    This differs from Nelder-Mead by explicitly searching for the global 
    minimum of |f(x)| first, then exploiting the structure of root finding 
    (sign changes) for speed.
    """
    start_time = time.time()
    deadline = start_time + 10.0  # Safety margin for 120s limit
    
    x0 = float(x0)
    
    # Phase 1: Global Coarse Scan to find the best starting point
    # We scan a range around x0. Since roots are likely O(1) or related to 
    # x0, we check a window.
    # If x0 is 0, check [-10, 10]. If x0 is large, check [x0-10, x0+10].
    # To be safe and fast, we use a moderate grid.
    
    # Determine search range
    center = x0
    if abs(center) < 1.0:
        center = 0.0
        range_min = -10.0
        range_max = 10.0
    else:
        range_min = center - 10.0
        range_max = center + 10.0
    
    # Coarse grid: 20 points
    n_coarse = 20
    step_coarse = (range_max - range_min) / (n_coarse - 1)
    
    best_x = center
    best_f = f(center)
    best_g = best_f * best_f
    
    # Check x0 first
    if abs(best_f) <= 1e-8:
        return center
        
    for i in range(1, n_coarse):
        x = range_min + i * step_coarse
        fx = f(x)
        g = fx * fx
        if g < best_g:
            best_g = g
            best_x = x
            best_f = fx
            if g <= 1e-16:
                return x
                
    # If the best point is still not very small, expand the search range?
    # For oscillatory functions, a root might be further away.
    # Let's try a second, wider scan if best_g is large (> 1.0)
    if best_g > 1.0:
        range_min2 = range_min - 20.0
        range_max2 = range_max + 20.0
        n_coarse2 = 20
        step_coarse2 = (range_max2 - range_min2) / (n_coarse2 - 1)
        # We can skip points already covered, but for simplicity and 
        # given the low cost, we just scan the new outer regions or 
        # a sparser grid over the whole expanded range.
        # Let's just scan the new outer parts to save calls.
        # Actually, simpler: just scan a wider range with the same density 
        # but only if the current best is bad.
        # To save calls, we'll just do a few more points in the expanded region.
        for i in range(1, n_coarse2):
            x = range_min2 + i * step_coarse2
            # Skip if x is in the original range (already scanned roughly)
            if range_min <= x <= range_max:
                continue
            fx = f(x)
            g = fx * fx
            if g < best_g:
                best_g = g
                best_x = x
                best_f = fx
                if g <= 1e-16:
                    return x

    # Phase 2: Local Refinement
    # We have a good candidate best_x. Now try to bracket a root nearby.
    # We look for a sign change in a small neighborhood of best_x.
    # If we find a sign change, use Illinois.
    # If not, we might be at a local minimum of |f| that is not a root, 
    # or the root is very close but no sign change is detected due to 
    # floating point or function shape (e.g., f(x) = (x-a)^2). 
    # In the latter case, we minimize f(x)^2 locally.
    
    # Define a small step for bracketing
    h = 1e-3
    x_left = best_x - h
    x_right = best_x + h
    
    # We need f(x_left) and f(x_right). We already have f(best_x).
    # To save calls, we might reuse best_f if we evaluate best_x again? 
    # No, we have best_f.
    
    fl = f(x_left)
    fr = f(x_right)
    
    # Check if we found a root exactly
    if abs(fl) <= 1e-8:
        return x_left
    if abs(fr) <= 1e-8:
        return x_right
    
    # Check for sign change in [x_left, best_x] or [best_x, x_right]
    # Or [x_left, x_right] if best_x is not between them? It is.
    
    # Illinois Method Implementation
    # We need a bracket [a, b] such that f(a)*f(b) <= 0.
    
    a, b = None, None
    fa, fb = None, None
    
    # Check [x_left, best_x]
    if fl * best_f <= 0:
        a, b = x_left, best_x
        fa, fb = fl, best_f
    # Check [best_x, x_right]
    elif best_f * fr <= 0:
        a, b = best_x, x_right
        fa, fb = best_f, fr
    # Check [x_left, x_right] (if best_x didn't have sign change with neighbors, 
    # maybe the sign change is across the whole interval? Unlikely if best_x 
    # is the min of |f|, but possible if f is flat near 0).
    elif fl * fr <= 0:
        a, b = x_left, x_right
        fa, fb = fl, fr
    else:
        # No sign change found in immediate neighborhood.
        # This implies we are likely at a local minimum of |f(x)| that is 
        # not a root (e.g., f(x) = (x-1)^2 + 0.1, min at x=1 is 0.1).
        # In this case, we cannot use bracketing. We must minimize |f(x)|.
        # We will use a simple local minimization on |f(x)| around best_x.
        # Since we are already at the best of the coarse scan, we just 
        # return the best_x if it's close enough, or do a few steps of 
        # 3-point search.
        
        # If best_g is already small, return.
        if best_g <= 1e-16:
            return best_x
            
        # Try a small 3-point search to see if we can improve
        h_min = 1e-4
        x1 = best_x - h_min
        x2 = best_x + h_min
        
        f1 = f(x1)
        f2 = f(x2)
        
        g1 = f1*f1
        g2 = f2*f2
        
        # Compare with best_g
        if g1 < best_g:
            best_x = x1
            best_g = g1
            best_f = f1
        if g2 < best_g:
            best_x = x2
            best_g = g2
            best_f = f2
            
        # If we still don't have a root, we return the best found.
        # The task says "returns x with |f(x)| <= 1e-8". 
        # If the function doesn't have a root in the vicinity, we can't do better.
        # However, the problem states "3 fixed oscillatory equations", implying 
        # roots exist. If we didn't find a sign change, maybe the root is 
        # further away? But our coarse scan should have caught a closer 
        # "valley". 
        # To be safe, if we didn't find a bracket, we return the best candidate.
        return best_x

    # We have a bracket [a, b]
    # Illinois Method
    # Keep track of the last function value to avoid redundant calls
    # Illinois converges faster than bisection.
    
    # Ensure fa is the "older" value (larger magnitude) to shrink the 
    # interval towards the root effectively? 
    # Standard Illinois: 
    # 1. Evaluate f at midpoint (or regula falsi point).
    # 2. Replace the endpoint with the larger |f| with the new point.
    
    # Let's implement a robust Illinois loop
    max_iter = 50
    for _ in range(max_iter):
        if time.time() > deadline - 1.0:
            break
            
        # Check convergence
        if abs(fb) <= 1e-8:
            return b
        if abs(fa) <= 1e-8:
            return a
            
        # Regula Falsi estimate
        if abs(fa) == abs(fb):
            x_mid = (a + b) / 2.0
        else:
            x_mid = (a * fb - b * fa) / (fb - fa)
            
        # Ensure x_mid is within (a, b)
        if x_mid <= a or x_mid >= b:
            x_mid = (a + b) / 2.0
            
        f_mid = f(x_mid)
        
        if abs(f_mid) <= 1e-8:
            return x_mid
            
        if fa * f_mid <= 0:
            b = x_mid
            fb = f_mid
        else:
            a = x_mid
            fa = f_mid
            
        # Illinois Shrink: 
        # The endpoint with the larger |f| is replaced by the previous 
        # value of the endpoint with the smaller |f|? 
        # Actually, Illinois replaces the function value of the 
        # "stagnant" endpoint with half its previous value to force 
        # bisection-like behavior if regula falsi stalls.
        
        # Standard Illinois:
        # After updating a/b, we identify which endpoint is "old" (has been 
        # there for 2 iterations). A simpler approximation:
        # If we update a, then fa_new = f_mid. The other endpoint b is "old".
        # We set f_b = f_b / 2.
        # This requires tracking which side was updated.
        
        # Let's use a simplified Illinois:
        # We track fa, fb.
        # If we updated a (a = x_mid), then b is the one that hasn't changed 
        # in this step (relative to the other). 
        # Actually, the standard Illinois algorithm is:
        # 1. Compute x_r.
        # 2. If f(a)*f(x_r) < 0, set b = x_r. Else a = x_r.
        # 3. If f(a) and f(b) have the same sign (which shouldn't happen 
        #    if we bracket correctly, but in Illinois we modify f values), 
        #    wait.
        
        # Correct Illinois:
        # We maintain [a, b] and f(a), f(b).
        # We compute x_r.
        # We evaluate f(x_r).
        # If f(a) and f(x_r) have opposite signs, we replace b with x_r.
        #   Then we set f(b) = f(b) / 2. (The new b is x_r, so we halve 
        #   the value of the NEW b? No, we halve the value of the endpoint 
        #   that was NOT replaced? 
        #   Actually, the rule is: The function value of the endpoint that 
        #   has not been updated in the previous step is halved.
        #   A common implementation:
        #   If we replace b, we halve f(a).
        #   If we replace a, we halve f(b).
        
        # Let's implement that.
        # We need to know which one was replaced.
        
        # Re-evaluating the loop logic with this rule:
        
    # Since I didn't implement the shrink logic in the loop above, 
    # let's rewrite the loop properly.
    
    # Reset for proper Illinois
    a, b = x_left, x_right # Wait, we already have a, b, fa, fb from above
    # Let's just re-initialize the Illinois loop properly.
    
    # Current state: a, b, fa, fb are valid bracket.
    # We need to track if we should halve fa or fb.
    # Let's use a flag.
    
    # Actually, the simple Regula Falsi without Illinois shrink might 
    # stall if the function is very non-linear. But for oscillatory 
    # smooth functions, it's usually fine. 
    # To be safe, let's just do Bisection if Regula Falsi doesn't 
    # converge fast enough.
    
    # Let's stick to a robust Bisection/Regula Falsi hybrid.
    # We already have a bracket.
    
    # Re-initialize variables for the refinement loop
    # a, b, fa, fb are set.
    
    for _ in range(50):
        if time.time() > deadline - 1.0:
            break
            
        if abs(fa) <= 1e-8:
            return a
        if abs(fb) <= 1e-8:
            return b
            
        # Regula Falsi point
        if abs(fa) == abs(fb):
            x_mid = (a + b) / 2.0
        else:
            x_mid = (a * fb - b * fa) / (fb - fa)
            
        if x_mid <= a or x_mid >= b:
            x_mid = (a + b) / 2.0
            
        f_mid = f(x_mid)
        
        if abs(f_mid) <= 1e-8:
            return x_mid
            
        if fa * f_mid <= 0:
            b = x_mid
            fb = f_mid
            # Illinois: halve fa (the one not updated)
            fa = fa / 2.0 
        else:
            a = x_mid
            fa = f_mid
            # Illinois: halve fb (the one not updated)
            fb = fb / 2.0
            
        # Check convergence of interval
        if abs(b - a) < 1e-12:
            return a if abs(fa) < abs(fb) else b

    # Return the best of a, b
    return a if abs(fa) < abs(fb) else b

# EVOLVE-BLOCK-END
