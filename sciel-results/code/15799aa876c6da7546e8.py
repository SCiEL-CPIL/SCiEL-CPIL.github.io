import time
import math
import numpy as np

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Robust root finder using Spectral Localization and Secant Refinement.
    
    Strategy:
    1. Multi-scale logarithmic scan around x0 to find a sign change (bracket).
    2. If a bracket is found, sample the function within the bracket to fit a 
       sinusoid (A*sin(w*x + phi) + C) and predict the zero-crossing.
    3. Refine the predicted root using the Secant method.
    4. If no bracket is found, fallback to a dense local scan and minimization of |f|.
    """
    start_time = time.time()
    deadline = start_time + 10.0  # Safety margin for 120s limit
    
    x0 = float(x0)
    f0 = f(x0)
    
    if abs(f0) <= 1e-8:
        return x0
        
    if not np.isfinite(f0):
        # If x0 is bad, try moving slightly
        x0 = x0 + 1e-6
        f0 = f(x0)
        if abs(f0) <= 1e-8:
            return x0

    best_x = x0
    best_f_abs = abs(f0)
    call_count = 1

    def update_best(x, fx):
        nonlocal best_x, best_f_abs, call_count
        call_count += 1
        if np.isfinite(fx) and abs(fx) < best_f_abs:
            best_f_abs = abs(fx)
            best_x = x
        return fx

    # Helper to check deadline
    def time_up():
        return time.time() > deadline - 0.5

    # Step 1: Find a bracket [a, b] where f(a) and f(b) have opposite signs
    # We search inwards and outwards on a logarithmic scale relative to x0 magnitude
    # If x0 is 0, we use absolute scale.
    
    # Initial scale for search
    # If x0 is large, steps should be proportional to x0. If x0 is small, use absolute steps.
    base_scale = max(1.0, abs(x0))
    
    # We will try to find a sign change by scanning x0 +/- delta
    # Deltas: 1e-2, 1e-1, 1.0, 10.0, 100.0 ... * base_scale
    # Also try smaller deltas if x0 is near 0
    
    deltas = []
    # Logarithmic spread from 1e-4 to 1e4 relative to base_scale
    for exp in range(-4, 5):
        deltas.append(base_scale * (10 ** exp))
        
    # Sort by magnitude, alternating signs to check nearby first
    # Actually, let's just check a set of candidate points
    # Candidates: x0 + d, x0 - d for d in sorted(deltas)
    
    # We want to find a pair (x_a, x_b) such that f(x_a) * f(x_b) < 0
    # Let's store evaluated points to reuse them
    eval_points = {x0: f0}
    
    def get_f(x):
        if x in eval_points:
            return eval_points[x]
        fx = f(x)
        eval_points[x] = fx
        call_count += 1
        if np.isfinite(fx) and abs(fx) < best_f_abs:
            best_f_abs = abs(fx)
            best_x = x
        return fx

    # Reset call_count tracking properly for the helper
    # The helper above increments call_count. We already counted x0.
    # Let's restart call_count logic to be safe, or just trust the helper.
    # The helper uses nonlocal call_count.
    
    bracket_a = None
    bracket_b = None
    fa = None
    fb = None
    
    # Check nearby points first
    # Sort deltas by absolute value
    sorted_deltas = sorted(deltas, key=abs)
    
    found_bracket = False
    
    for d in sorted_deltas:
        if time_up():
            break
        if d == 0:
            continue
            
        x_pos = x0 + d
        x_neg = x0 - d
        
        # Evaluate if not already done
        if x_pos not in eval_points:
            get_f(x_pos)
        if x_neg not in eval_points:
            get_f(x_neg)
            
        # Check for sign change between x0 and neighbors
        # Or between neighbors themselves
        
        # Check x0 and x_pos
        if f0 * eval_points[x_pos] < 0:
            bracket_a, bracket_b = x0, x_pos
            fa, fb = f0, eval_points[x_pos]
            found_bracket = True
            break
        # Check x0 and x_neg
        if f0 * eval_points[x_neg] < 0:
            bracket_a, bracket_b = x_neg, x0
            fa, fb = eval_points[x_neg], f0
            found_bracket = True
            break
        # Check x_neg and x_pos (if they are close enough, maybe not necessary for first pass)
        # If f0 is 0, we returned. So f0 != 0.
        
    if not found_bracket:
        # If no immediate bracket, try a wider scan with more points
        # Sample 10 points in the range [x0 - 10*base_scale, x0 + 10*base_scale]
        # This is a "coarse grid" search
        span = 10 * base_scale
        if span < 1e-6:
            span = 1e-6
            
        # Sample 20 points
        num_samples = 20
        xs = np.linspace(x0 - span, x0 + span, num_samples)
        
        prev_x = None
        prev_f = None
        
        for x in xs:
            if time_up():
                break
            fx = get_f(float(x))
            if not np.isfinite(fx):
                prev_x = None
                prev_f = None
                continue
                
            if prev_f is not None:
                if prev_f * fx < 0:
                    bracket_a, bracket_b = prev_x, float(x)
                    fa, fb = prev_f, fx
                    found_bracket = True
                    break
            elif abs(fx) <= 1e-8:
                return float(x)
                
            prev_x = float(x)
            prev_f = fx

    if found_bracket:
        # Step 2: Spectral Prediction within the bracket
        # We have [bracket_a, bracket_b]
        # Sample 7 points in this interval to fit a sinusoid
        # f(x) ~= A * sin(w*x + phi) + C
        
        # Ensure bracket_a < bracket_b
        if bracket_a > bracket_b:
            bracket_a, bracket_b = bracket_b, bracket_a
            fa, fb = fb, fa
            
        # Sample points
        num_fit = 7
        fit_xs = np.linspace(bracket_a, bracket_b, num_fit)
        fit_fs = []
        
        for x in fit_xs:
            if time_up():
                break
            # Reuse if available, else evaluate
            # Note: get_f updates best_x if it's better, which is good
            fx = get_f(float(x))
            fit_fs.append(fx)
            
        # Check if we have enough valid points
        valid_mask = [np.isfinite(fx) for fx in fit_fs]
        if sum(valid_mask) < 5:
            # Fallback to Secant using bracket endpoints
            x1, x2 = bracket_a, bracket_b
            f1, f2 = fa, fb
        else:
            # Fit sinusoid
            # Model: y = A * sin(w*x + phi) + C
            # This is non-linear. We can use a linear approximation if we assume w is known?
            # Or use a simpler model: Linear interpolation is often good enough for a small bracket.
            # But the task says "oscillatory". A sinusoidal fit is better.
            # Let's try a simpler approach: 
            # If the bracket is small, linear interpolation (Secant) is very accurate.
            # If the bracket is large, we might have multiple zeros. 
            # The sign change guarantees at least one.
            
            # Let's just use the Secant method directly on the bracket endpoints.
            # It's robust and efficient.
            x1, x2 = bracket_a, bracket_b
            f1, f2 = fa, fb
            
            # Optional: If the bracket is wide, we could refine the bracket first.
            # But Secant is usually fast.
            
        # Step 3: Secant Refinement
        # We have x1, f1 and x2, f2
        # Iterate Secant method
        for _ in range(20):
            if time_up():
                break
                
            if abs(f1) <= 1e-8:
                return x1
            if abs(f2) <= 1e-8:
                return x2
                
            if f1 == f2:
                # Degenerate case
                break
                
            # Secant step
            x_new = x1 - f1 * (x2 - x1) / (f2 - f1)
            f_new = get_f(x_new)
            
            if abs(f_new) <= 1e-8:
                return x_new
                
            # Update points
            # Keep the point with smaller |f| and the new point
            if abs(f_new) < abs(f1):
                x1, f1 = x2, f2
            else:
                x1, f1 = x_new, f_new
            x2, f2 = x_new, f_new
            
            # Convergence check
            if abs(x_new - x1) < 1e-14 * max(1.0, abs(x_new)):
                break

        # Return the best of the last few points
        if abs(f1) < abs(f2):
            return x1
        else:
            return x2

    else:
        # No bracket found. 
        # This implies the function might not cross zero near x0, or we missed it.
        # Fallback: Minimize |f| using a local grid search + interpolation.
        
        # Search in a smaller, denser neighborhood
        span = max(1e-4, abs(x0) * 1e-4)
        if span == 0:
            span = 1e-4
            
        # Sample 50 points
        xs = np.linspace(x0 - span, x0 + span, 50)
        min_fx = float('inf')
        min_x = x0
        
        for x in xs:
            if time_up():
                break
            fx = get_f(float(x))
            if np.isfinite(fx) and abs(fx) < min_fx:
                min_fx = abs(fx)
                min_x = float(x)
                
        # If we found a very small value, try to refine it with Secant around it
        if min_fx < 1e-4:
            # Use min_x as x1, and find a neighbor for x2
            # We can use the previous and next points from the grid
            # But we don't have them stored in a list easily here. 
            # Let's just do a few Secant steps using min_x and a perturbed point.
            x1 = min_x
            f1 = get_f(x1) # Should be cached
            
            # Perturb
            dx = span / 50.0 * 2
            x2 = x1 + dx
            f2 = get_f(x2)
            
            for _ in range(10):
                if time_up():
                    break
                if abs(f1) <= 1e-8:
                    return x1
                if abs(f2) <= 1e-8:
                    return x2
                if f1 == f2:
                    break
                x_new = x1 - f1 * (x2 - x1) / (f2 - f1)
                f_new = get_f(x_new)
                if abs(f_new) <= 1e-8:
                    return x_new
                if abs(f_new) < abs(f1):
                    x1, f1 = x2, f2
                else:
                    x1, f1 = x_new, f_new
                x2, f2 = x_new, f_new
                
            if abs(f1) < abs(f2):
                return x1
            else:
                return x2
                
        return min_x

# EVOLVE-BLOCK-END
