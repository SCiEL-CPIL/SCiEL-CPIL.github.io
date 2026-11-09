import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Robust root finder using Multi-Scale Adaptive Bracketing + Illinois Refinement.
    
    Strategy:
    1. Adaptive Bracketing: Expand search radius geometrically from x0 to find 
       a sign change [a, b]. This handles cases where the root is far from x0 
       or x0 is in a region of constant sign.
    2. Illinois Algorithm: Once a bracket is found, use the Illinois method 
       (a variant of Regula Falsi) to converge. It is derivative-free, 
       super-linear (faster than bisection), and robust against oscillations 
       compared to Newton's method.
    3. Fallback: If no sign change is found (e.g., tangent root), use a local 
       minimization of |f(x)| via a fine grid search near the best point found, 
       followed by a tight local Illinois-like refinement if possible, or simply 
       return the best point if it meets the tolerance.
    """
    start_time = time.time()
    deadline = start_time + 10.0  # Safety margin for 120s limit
    
    x0 = float(x0)
    
    # Cache to avoid redundant calls
    cache = {}
    
    def f_cached(x):
        if x in cache:
            return cache[x]
        val = f(x)
        cache[x] = val
        return val
        
    # Initial check
    fx0 = f_cached(x0)
    if abs(fx0) <= 1e-8:
        return x0
        
    # Phase 1: Adaptive Bracketing
    # We search in expanding intervals.
    # Start with a small step, double it until we find a sign change or reach a max radius.
    
    a = x0
    fa = fx0
    
    # Initial step size
    h = 1e-3
    max_h = 1e6
    
    b = None
    fb = None
    
    # Track best point found so far (in case no bracket is found)
    best_x = x0
    best_f = abs(fx0)
    
    # We will try steps in both directions if necessary, but usually 
    # expanding from x0 is enough. Let's do a simple expansion.
    
    # To be robust, we check a few points in a small local area first to see 
    # if the function is "flat" or if we are near a root.
    
    # Simple expanding search
    # We alternate directions or just expand radius?
    # Let's expand radius: x0 - h, x0 + h, x0 - 2h, x0 + 2h ...
    # Or just x0 + h, x0 - h, x0 + 2h, x0 - 2h...
    
    current_h = 1e-2
    direction = 1
    steps_taken = 0
    max_steps = 100
    
    found_bracket = False
    
    while current_h <= max_h and steps_taken < max_steps:
        if time.time() > deadline - 5.0:
            break
            
        # Try x0 + direction * current_h
        x_try = x0 + direction * current_h
        f_try = f_cached(x_try)
        
        if abs(f_try) < best_f:
            best_f = abs(f_try)
            best_x = x_try
            
        if abs(f_try) <= 1e-8:
            return x_try
            
        # Check for sign change between x0 (or previous point) and x_try?
        # Actually, checking between x0 and x_try is good, but if x0 is far, 
        # we might miss a bracket between x0+h and x0+2h.
        # Better: Keep a 'last' point and check sign change between last and current.
        # But 'last' should be the previous point in the sequence.
        
        # Let's restructure:
        # Point 0: x0
        # Point 1: x0 + h
        # Point 2: x0 - h
        # Point 3: x0 + 2h
        # Point 4: x0 - 2h
        # ...
        
        # We need to track the previous point in the sequence to check for bracket.
        # However, the sequence jumps back and forth.
        # A simpler robust strategy:
        # 1. Check x0.
        # 2. Check x0 + h. If sign change, bracket [x0, x0+h].
        # 3. Check x0 - h. If sign change with x0, bracket [x0-h, x0].
        # 4. If no bracket, increase h. Check x0 + 2h. 
        #    Check sign change between x0+h and x0+2h? Or x0 and x0+2h?
        #    Checking x0 and x0+2h is safer to ensure we don't skip over a 
        #    single sign change if the function oscillates fast, but we might 
        #    have multiple roots. Any root is fine.
        #    If f(x0) and f(x0+2h) have opposite signs, there is at least one 
        #    root in between. We can bisect that interval.
        
        # Let's stick to checking against x0 first, as it's the anchor.
        if fa * f_try < 0:
            a = x0
            b = x_try
            fa = fx0
            fb = f_try
            found_bracket = True
            break
            
        # Also check if we have a previous point in the same direction?
        # For simplicity, let's just expand h and check against x0.
        # If the root is very close to x0, we'll catch it.
        # If the root is far, we'll catch it when h is large enough.
        
        # Alternate direction
        if direction == 1:
            # Try negative side
            x_try_neg = x0 - current_h
            f_try_neg = f_cached(x_try_neg)
            
            if abs(f_try_neg) < best_f:
                best_f = abs(f_try_neg)
                best_x = x_try_neg
                
            if abs(f_try_neg) <= 1e-8:
                return x_try_neg
                
            if fa * f_try_neg < 0:
                a = x0
                b = x_try_neg
                fa = fx0
                fb = f_try_neg
                found_bracket = True
                break
        else:
            # We already tried negative in the previous iteration's logic? 
            # No, my logic above only tried positive then negative in the same 'step'.
            # Let's simplify the loop structure.
            pass
        
        # Update h and direction for next iteration
        # If we just tried positive, next try negative? 
        # Or just increase h?
        # Let's do:
        # Iter 1: +1h, -1h
        # Iter 2: +2h, -2h
        # Iter 3: +4h, -4h
        
        current_h *= 2.0
        steps_taken += 1
        
    # If no bracket found, try a local minimization strategy?
    # Or just return best_x if it's good enough?
    # If best_f > 1e-8, we might fail.
    # Let's try a few more points around best_x if no bracket.
    
    if not found_bracket:
        # Local refinement around best_x
        # Use a small grid to see if we can find a sign change nearby
        local_h = max(1e-6, abs(best_x) * 1e-6)
        for i in range(10):
            if time.time() > deadline - 2.0:
                break
            x_try = best_x + (i - 5) * local_h
            f_try = f_cached(x_try)
            if abs(f_try) <= 1e-8:
                return x_try
            if abs(f_try) < best_f:
                best_f = abs(f_try)
                best_x = x_try
        
        # If still no bracket and best_f > 1e-8, we have to return best_x 
        # and hope for the best, or try a very tight local search.
        # Given the task is "oscillatory equations", it's highly likely 
        # a sign change exists if we search far enough.
        # If we didn't find one, maybe the root is a tangent (double root).
        # In that case, Newton or Secant might work better, but they are risky.
        # Let's return best_x.
        return best_x

    # Phase 2: Illinois Algorithm
    # We have a bracket [a, b] with fa * fb < 0.
    # Illinois method:
    # 1. Compute x_r (regula falsi step).
    # 2. If x_r is closer to a than b, shrink b by half (b = (a+b)/2).
    #    Else shrink a by half.
    # 3. Evaluate f at the new point.
    # 4. Update bracket.
    # This ensures linear convergence but faster than bisection.
    
    a, b = a, b
    fa, fb = fa, fb
    
    # Ensure a < b for clarity, though not strictly necessary for the math
    if a > b:
        a, b = b, a
        fa, fb = fb, fa
        
    max_iter = 100
    
    for _ in range(max_iter):
        if time.time() > deadline - 0.5:
            break
            
        # Regula Falsi step
        # x_r = a - fa * (b - a) / (fb - fa)
        # To avoid division by zero or instability, check denominator
        denom = fb - fa
        if abs(denom) < 1e-15:
            # Fallback to bisection if denominator is tiny
            x_mid = (a + b) / 2
            f_mid = f_cached(x_mid)
            if abs(f_mid) <= 1e-8:
                return x_mid
            if fa * f_mid < 0:
                b = x_mid
                fb = f_mid
            else:
                a = x_mid
                fa = f_mid
            continue
            
        x_r = a - fa * (b - a) / denom
        
        # Check if x_r is within bracket (it should be, but floating point?)
        if x_r <= a or x_r >= b:
            x_r = (a + b) / 2
            
        f_r = f_cached(x_r)
        
        if abs(f_r) <= 1e-8:
            return x_r
            
        # Illinois update
        if fa * f_r < 0:
            # Root is in [a, x_r]
            b = x_r
            fb = f_r
            # If we just replaced b, shrink the OTHER end (a) by half?
            # Standard Illinois: 
            # If f(a)*f(x_r) < 0, then b=x_r. 
            # Then we need to decide which side to shrink.
            # The side that has been kept longer (or is larger) should be shrunk.
            # In Illinois, we shrink the side that did NOT change.
            # Since b changed, we shrink a.
            a = (a + b) / 2
            fa = f_cached(a)
        else:
            # Root is in [x_r, b]
            a = x_r
            fa = f_r
            # Since a changed, we shrink b.
            b = (a + b) / 2
            fb = f_cached(b)
            
        # Convergence check
        if abs(b - a) < 1e-15:
            break
            
    # Return the midpoint or the point with smaller |f|
    x_mid = (a + b) / 2
    f_mid = f_cached(x_mid)
    
    # Final check
    if abs(fa) <= abs(fb):
        if abs(fa) <= 1e-8:
            return a
        if abs(f_mid) <= 1e-8:
            return x_mid
        return a
    else:
        if abs(fb) <= 1e-8:
            return b
        if abs(f_mid) <= 1e-8:
            return x_mid
        return b

# EVOLVE-BLOCK-END
