"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find a root of f starting near x0.
    Strategy: Optimized Global Secant Search with adaptive scale ordering 
    and bisection fallback. Prioritizes scales likely to bracket roots 
    in oscillatory functions, interleaving local and global probes.
    """
    # Use a safety margin to avoid timeouts (limit is 120s, kill at 150s)
    deadline = time.time() + 115.0 
    tol = 1e-9
    x0 = float(x0)
    
    # Global best tracking
    best_x = x0
    best_f = None
    best_abs = float('inf')
    
    def update_best(x_val, f_val):
        nonlocal best_x, best_f, best_abs
        if x_val is None or f_val is None:
            return
        if not math.isfinite(x_val) or not math.isfinite(f_val):
            return
        abs_f = abs(f_val)
        if abs_f < best_abs:
            best_x = x_val
            best_f = f_val
            best_abs = abs_f
        return abs_f

    # 1. Initial Probing
    # Check x0 itself
    try:
        f_x0 = f(x0)
        update_best(x0, f_x0)
        if best_abs <= tol:
            return best_x
    except Exception:
        f_x0 = None
        best_abs = float('inf')

    # Define search scales. 
    # For oscillatory functions, roots are often spaced by periods (e.g., 2pi).
    # We want to try scales that are likely to straddle a zero.
    # Original: 1e-5 ... 1e4 sequentially.
    # New: Interleaved and optimized.
    # Scales: 
    # 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000
    # Plus some "sub-scale" and "super-scale" variations to avoid symmetry.
    
    # We create a list of (scale, mode) where mode determines how points are picked.
    # Mode 0: Symmetric bracket (x0-s, x0+s)
    # Mode 1: Asymmetric forward (x0+s, x0+2s)
    # Mode 2: Asymmetric backward (x0-s, x0-2s)
    # Mode 3: Single step from x0 (x0, x0+s) - effectively secant with x0
    
    search_scales = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 1e4]
    
    # We will iterate through scales. To improve robustness, we don't just do one mode per scale.
    # We will try a few different geometric configurations per scale if time permits.
    
    # Sort scales by "promisingness"? Hard to know. 
    # Let's try a specific order: 1, 10, 0.1, 100, 0.01, 1000, 0.001, 10000.
    # This covers local, medium, and global rapidly.
    ordered_scales = [1, 10, 0.1, 100, 0.01, 1000, 0.001, 10000]
    
    for scale in ordered_scales:
        if time.time() > deadline:
            break
            
        # For each scale, try a few different initial pairs.
        # We limit the number of pairs per scale to conserve calls if early ones fail.
        
        # Pair 1: Symmetric bracket
        # x_a = x0 - scale, x_b = x0 + scale
        x_a = x0 - scale
        x_b = x0 + scale
        
        # Pair 2: Forward step
        # x_c = x0 + scale, x_d = x0 + 2*scale
        x_c = x0 + scale
        x_d = x0 + 2.0 * scale
        
        # Pair 3: Backward step
        # x_e = x0 - scale, x_f = x0 - 2*scale
        x_e = x0 - scale
        x_f = x0 - 2.0 * scale

        # We will test these pairs sequentially.
        # We can break early if we find a root in the first pair.
        
        pairs_to_test = [
            (x_a, x_b),
            (x_c, x_d),
            (x_e, x_f)
        ]
        
        for x_start, x_end in pairs_to_test:
            if time.time() > deadline:
                break
                
            # Skip if points are too close or identical
            if abs(x_end - x_start) < 1e-12 * max(1.0, abs(x0)):
                continue
                
            try:
                f_start = f(x_start)
                f_end = f(x_end)
            except Exception:
                continue
                
            update_best(x_start, f_start)
            update_best(x_end, f_end)
            
            if best_abs <= tol:
                return best_x
                
            # Check if we found a root immediately
            if abs(f_start) <= tol:
                return x_start
            if abs(f_end) <= tol:
                return x_end
                
            # Run Secant Method
            x_curr_a = x_start
            x_curr_b = x_end
            f_curr_a = f_start
            f_curr_b = f_end
            
            # Check for bracket
            has_bracket = (f_curr_a * f_curr_b < 0)
            bracket_lo = min(x_curr_a, x_curr_b)
            bracket_hi = max(x_curr_a, x_curr_b)
            
            # Secant iterations
            for _ in range(30): # Reduced max iterations slightly to allow more trials if needed, secant is fast
                if time.time() > deadline:
                    break
                    
                if abs(f_curr_a) <= tol:
                    return x_curr_a
                if abs(f_curr_b) <= tol:
                    return x_curr_b
                
                denom = f_curr_b - f_curr_a
                if denom == 0 or not math.isfinite(denom):
                    break
                    
                x_new = x_curr_b - f_curr_b * (x_curr_b - x_curr_a) / denom
                
                if not math.isfinite(x_new):
                    break
                
                # Safety: if the step is huge, it might be diverging. 
                # But for oscillatory f, big jumps can be good.
                # We'll just proceed but check time.
                
                x_curr_a = x_curr_b
                f_curr_a = f_curr_b
                x_curr_b = x_new
                
                try:
                    f_curr_b = f(x_new)
                except Exception:
                    break
                    
                update_best(x_curr_b, f_curr_b)
                
                if best_abs <= tol:
                    return best_x
                
                # Update bracket logic
                if f_curr_a * f_curr_b < 0:
                    has_bracket = True
                    bracket_lo = min(x_curr_a, x_curr_b)
                    bracket_hi = max(x_curr_a, x_curr_b)
                elif has_bracket:
                    # If we lose the bracket (signs same) and new point is outside old bracket,
                    # the secant likely jumped over the root or diverged.
                    if not (bracket_lo <= x_new <= bracket_hi):
                        has_bracket = False
                    # If inside but no sign change, f is likely not continuous or we missed it.
            
            # Bisection Fallback
            if has_bracket and (bracket_hi - bracket_lo) > 0:
                # We need f values at the bracket ends. 
                # We might have them from the last secant step, but they might be stale 
                # if the bracket was defined earlier. 
                # Re-evaluating is safer but costs 2 calls.
                # Given the 500 call limit per equation, 2 calls for a guaranteed 
                # convergence path is a good tradeoff.
                
                try:
                    f_lo = f(bracket_lo)
                    f_hi = f(bracket_hi)
                except Exception:
                    continue
                    
                update_best(bracket_lo, f_lo)
                update_best(bracket_hi, f_hi)
                
                if best_abs <= tol:
                    return best_x
                
                # Ensure bracket is valid
                if f_lo * f_hi > 0:
                    continue
                
                # Bisection loop
                # Bisection halves the interval. 
                # To get 1e-9 precision from an interval of size S, we need log2(S/1e-9) iterations.
                # If S ~ 100, log2(1e11) ~ 36.
                # Let's allow up to 50 bisection steps.
                for _ in range(50):
                    if time.time() > deadline:
                        break
                        
                    x_mid = (bracket_lo + bracket_hi) / 2.0
                    try:
                        f_mid = f(x_mid)
                    except Exception:
                        break
                        
                    update_best(x_mid, f_mid)
                    
                    if best_abs <= tol:
                        return best_x
                        
                    if abs(f_mid) <= tol:
                        return x_mid
                        
                    if f_lo * f_mid <= 0:
                        bracket_hi = x_mid
                        f_hi = f_mid
                    else:
                        bracket_lo = x_mid
                        f_lo = f_mid

    # Return the best result found
    return best_x
# EVOLVE-BLOCK-END
