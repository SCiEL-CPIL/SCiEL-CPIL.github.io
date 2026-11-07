import numpy as np

def find_root(f, x0):
    tol = 1e-8
    
    # Phase 1: Global Expansion to find a bracket
    # We expand a window around x0 to find a sign change.
    # We keep track of the best x found so far (min |f(x)|) in case we fail to bracket.
    
    x_best = float(x0)
    f_best = f(x_best)
    
    if not np.isfinite(f_best):
        # If x0 is bad, try a small perturbation? Or just fail?
        # The task says non-finite x is a failure. But f(x) non-finite?
        # "an exception, a non-finite x, None or |f(x)| > 1e-8 makes that equation a failure"
        # It implies if we return x where f(x) is not finite, it's a failure.
        # If f(x0) is not finite, we should try to move.
        # But let's assume f is well-behaved at x0 for now, or handle it.
        pass

    if abs(f_best) <= tol:
        return x_best

    # Initial step size. Small to catch high frequency.
    h = 1e-3
    max_h = 1e4
    max_expansions = 50
    
    # We evaluate x0 once.
    # We will evaluate points to the right and left.
    # To avoid re-evaluating, we just check new points.
    
    # Points to check: x0 + h, x0 - h, x0 + 2h, x0 - 2h ... 
    # Actually, doubling h skips intervals. 
    # If f(x0+h) and f(x0+2h) have same sign, we might miss a root in (x0+h, x0+2h) if it oscillates twice?
    # No, if it oscillates, it changes sign. 
    # If f(a) and f(b) have same sign, there is an even number of roots (or 0) in (a,b).
    # We need a bracket with opposite signs.
    
    # Strategy: 
    # 1. Check x0.
    # 2. Check x0 + h. If sign change, bracket [x0, x0+h].
    # 3. Check x0 - h. If sign change, bracket [x0-h, x0].
    # 4. If no sign change, increase h. 
    #    But we must check the *new* points against the *previous* boundary points?
    #    If we just jump to 2h, we ignore (h, 2h).
    #    To be safe against oscillations, we should step linearly for a while?
    #    Or, we can just accept that we might miss a root if we skip, but expand range.
    #    Given "oscillatory", high frequency is possible.
    #    Let's use a linear scan for small h, then exponential?
    #    Or just linear scan up to max_h.
    
    # Let's do a linear scan with step size h=1e-3 up to 10 steps (range 0.01), 
    # then h=1e-2 up to 10 steps (range 0.1), etc.
    # This is a "Zoom" strategy.
    
    current_h = 1e-3
    for zoom_level in range(10):
        for i in range(1, 11): # 10 steps in each direction
            step = i * current_h
            
            # Check right
            x_r = x0 + step
            f_r = f(x_r)
            if np.isfinite(f_r):
                if abs(f_r) < abs(f_best):
                    f_best = f_r
                    x_best = x_r
                if f_best * f_r < 0: # Wait, f_best is min abs, not necessarily sign change partner.
                    # We need to check against the point that forms the bracket.
                    # We should keep the last point in the direction.
                    pass
            
            # This logic is getting messy. Let's simplify.
            
            # Keep track of (x_prev, f_prev) in right direction and left direction.
            break # Pseudocode, let's rewrite cleanly.

    # Clean Implementation:
    
    x_center = float(x0)
    f_center = f(x_center)
    
    if not np.isfinite(f_center):
        # Try to recover?
        # If f(x0) is inf, we can't use it.
        # Let's just try a small shift.
        x_center = x0 + 1e-3
        f_center = f(x_center)
        if not np.isfinite(f_center):
            return x0 # Will likely fail, but best effort.

    if abs(f_center) <= tol:
        return x_center

    # State for right and left expansion
    x_r = x_center
    f_r = f_center
    x_l = x_center
    f_l = f_center
    
    best_x = x_center
    best_f = abs(f_center)
    
    h = 1e-3
    # We will expand h by factor 2, but we must check intermediate points?
    # No, if we double, we skip.
    # Let's use a fixed step size for the first phase to ensure we don't skip roots in high frequency zones.
    # Step 1e-3, 20 steps -> range 0.02.
    # Step 1e-2, 20 steps -> range 0.2.
    # Step 1e-1, 20 steps -> range 2.0.
    # Step 1.0, 20 steps -> range 20.0.
    
    step_sizes = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
    steps_per_direction = 20
    
    for h in step_sizes:
        for i in range(1, steps_per_direction + 1):
            # Right side
            x_new_r = x_center + i * h
            f_new_r = f(x_new_r)
            
            if np.isfinite(f_new_r):
                if abs(f_new_r) < best_f:
                    best_f = abs(f_new_r)
                    best_x = x_new_r
                    
                if f_r * f_new_r < 0:
                    # Bracket found [x_r, x_new_r]
                    return _bisect(f, x_r, x_new_r, f_r, f_new_r, tol)
                
                x_r = x_new_r
                f_r = f_new_r
            
            # Left side
            x_new_l = x_center - i * h
            f_new_l = f(x_new_l)
            
            if np.isfinite(f_new_l):
                if abs(f_new_l) < best_f:
                    best_f = abs(f_new_l)
                    best_x = x_new_l
                    
                if f_l * f_new_l < 0:
                    # Bracket found [x_new_l, x_l]
                    return _bisect(f, x_new_l, x_l, f_new_l, f_l, tol)
                
                x_l = x_new_l
                f_l = f_new_l
            
            # Optimization: if we are far away and |f| is increasing, maybe stop?
            # For oscillatory, |f| might not increase.
            
    return best_x

def _bisect(f, a, b, fa, fb, tol=1e-8, max_iter=100):
    # fa and fb have opposite signs
    if fa == 0: return a
    if fb == 0: return b
    
    for _ in range(max_iter):
        m = (a + b) / 2.0
        fm = f(m)
        
        if not np.isfinite(fm):
            # If mid is bad, shrink bracket?
            # Just return best guess?
            # Bisection assumes continuity. If fm is inf, something is wrong.
            # Try closer to a or b?
            # Let's just break and return m or best.
            break
            
        if abs(fm) <= tol:
            return m
            
        if fa * fm < 0:
            b = m
            fb = fm
        else:
            a = m
            fa = fm
            
        if abs(b - a) < 1e-12:
            break
            
    return (a + b) / 2.0
