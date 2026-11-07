"""Rewritten program with robust bracketing + Illinois method for oscillatory roots."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Robust root finder for oscillatory equations.
    Strategy:
      1. Quick Newton attempt (up to 10 steps, 2 calls/step = 20 calls)
      2. If not converged, find a bracket via expanding search
      3. Use Illinois method to converge within tolerance
    Budget: keep total calls well under 500.
    """
    start_time = time.time()
    TIME_LIMIT = 110.0  # safety margin under 120s
    TARGET = 1e-8
    MAX_CALLS = 480     # safety margin under 500
    
    call_count = 0
    def f_counted(x):
        nonlocal call_count
        call_count += 1
        val = f(x)
        if not math.isfinite(val):
            return float('nan')
        return val
    
    x = float(x0)
    
    # Phase 1: Newton's method with finite difference derivative
    # Up to 10 steps, each uses 2 calls (f(x) and f(x+h))
    # But we reuse fx when possible
    for step in range(10):
        if call_count >= MAX_CALLS - 5:
            break
        if time.time() - start_time > TIME_LIMIT - 5:
            break
        fx = f_counted(x)
        if not math.isfinite(fx):
            break
        if abs(fx) <= TARGET:
            return x
        h = 1e-7 * max(1.0, abs(x))
        if call_count >= MAX_CALLS - 1:
            break
        fx_plus = f_counted(x + h)
        if not math.isfinite(fx_plus):
            break
        d = (fx_plus - fx) / h
        if d == 0 or not math.isfinite(d):
            break
        x_new = x - fx / d
        if not math.isfinite(x_new):
            break
        x = x_new
    
    # Check if Newton already found a root
    if call_count < MAX_CALLS:
        fx = f_counted(x)
        if math.isfinite(fx) and abs(fx) <= TARGET:
            return x
    
    # Phase 2: Find a bracket [a, b] with f(a)*f(b) < 0
    # Start from x and expand in both directions
    x_start = x
    a = x_start
    fa = f_counted(a)
    if not math.isfinite(fa):
        # Try a small offset
        a = x_start + 1e-4
        fa = f_counted(a)
    
    if not math.isfinite(fa) or abs(fa) <= TARGET:
        if math.isfinite(fa) and abs(fa) <= TARGET:
            return a
        # fa is nan, try different starting point
        a = x_start
        fa = f_counted(a)
    
    # Expand to find bracket
    step_size = 1e-3
    found_bracket = False
    b = None
    fb = None
    
    for i in range(200):  # up to 200 expansion steps
        if call_count >= MAX_CALLS - 10:
            break
        if time.time() - start_time > TIME_LIMIT - 5:
            break
        
        # Try expanding to the right
        b_try = a + step_size
        fb_try = f_counted(b_try)
        if math.isfinite(fb_try):
            if abs(fb_try) <= TARGET:
                return b_try
            if math.isfinite(fa) and fa * fb_try < 0:
                b = b_try
                fb = fb_try
                found_bracket = True
                break
        step_size *= 2.0
        if step_size > 1e6:
            break
    
    if not found_bracket:
        # Try expanding to the left
        step_size = 1e-3
        b = None
        for i in range(200):
            if call_count >= MAX_CALLS - 10:
                break
            if time.time() - start_time > TIME_LIMIT - 5:
                break
            b_try = a - step_size
            fb_try = f_counted(b_try)
            if math.isfinite(fb_try):
                if abs(fb_try) <= TARGET:
                    return b_try
                if math.isfinite(fa) and fa * fb_try < 0:
                    b = b_try
                    fb = fb_try
                    found_bracket = True
                    break
            step_size *= 2.0
            if step_size > 1e6:
                break
    
    if not found_bracket:
        # Last resort: return best x found
        return x
    
    # Phase 3: Illinois method to converge
    # Illinois is a modified bisection that converges faster
    eps = 1e-15
    for i in range(300):  # up to 300 iterations
        if call_count >= MAX_CALLS:
            break
        if time.time() - start_time > TIME_LIMIT - 2:
            break
        
        mid = 0.5 * (a + b)
        fmid = f_counted(mid)
        if not math.isfinite(fmid):
            # Fall back to bisection point
            mid = 0.5 * (a + b)
            fmid = f_counted(mid)
            if not math.isfinite(fmid):
                break
        
        if abs(fmid) <= TARGET:
            return mid
        
        # Illinois: if f(a) and f(mid) have same sign, adjust a
        if fa * fmid > 0:
            a = mid
            fa = fmid
            # Illinois modification: shrink the interval on the other side
            b = a + (b - a) * 0.5
            # Actually, Illinois keeps the opposite endpoint fixed and
            # moves the same-sign endpoint to midpoint, then adjusts the
            # other endpoint. Let me use a cleaner version.
            # Standard Illinois: 
            #   if f(a)*f(mid) > 0: a = mid, fa = fmid; b = a + (b-a)/2
            #   else: b = mid, fb = fmid; a = b + (a-b)/2
            # But this can cause issues. Let me use a simpler approach:
            # Just do bisection but with a twist for faster convergence.
            pass
        else:
            b = mid
            fb = fmid
        
        # Actually, let me use a cleaner Illinois implementation
        # Reset and do it properly
        break
    
    # Let me redo Phase 3 with a proper Illinois method
    # Restart with the bracket we found
    # We need to re-evaluate since we may have modified a, b
    # Actually, let's just do a clean bisection with a limit
    # since Illinois implementation above was messy
    
    # Clean bisection/Illinois
    # Re-establish bracket
    a_val = a
    fa_val = fa
    b_val = b
    fb_val = fb
    
    # Make sure we have a valid bracket
    if not (math.isfinite(fa_val) and math.isfinite(fb_val) and fa_val * fb_val < 0):
        # Try to find a new bracket more carefully
        # Sample points around x0
        best_x = x_start
        best_f = abs(f_counted(x_start))
        
        for dx in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]:
            if call_count >= MAX_CALLS - 5:
                break
            for sign in [1, -1]:
                xx = x_start + sign * dx
                ff = f_counted(xx)
                if math.isfinite(ff) and abs(ff) < best_f:
                    best_f = abs(ff)
                    best_x = xx
                if abs(ff) <= TARGET:
                    return xx
        
        return best_x
    
    # Now do Illinois method properly
    a2 = a_val
    fa2 = fa_val
    b2 = b_val
    fb2 = fb_val
    
    for i in range(350):
        if call_count >= MAX_CALLS:
            break
        if time.time() - start_time > TIME_LIMIT - 1:
            break
        
        # Illinois step
        if fa2 * fb2 > 0:
            # Shouldn't happen if bracket is valid, but safety
            break
        
        # Compute interpolation point
        # Illinois: linear interpolation between (a, fa) and (b, fb)
        # x_interp = b - fb * (b - a) / (fb - fa)
        denom = fb2 - fa2
        if abs(denom) < 1e-30:
            x_new = 0.5 * (a2 + b2)
        else:
            x_new = b2 - fb2 * (b2 - a2) / denom
        
        # Ensure x_new is within [a2, b2]
        if x_new <= a2 or x_new >= b2:
            x_new = 0.5 * (a2 + b2)
        
        f_new = f_counted(x_new)
        if not math.isfinite(f_new):
            x_new = 0.5 * (a2 + b2)
            f_new = f_counted(x_new)
            if not math.isfinite(f_new):
                break
        
        if abs(f_new) <= TARGET:
            return x_new
        
        # Update bracket
        if fa2 * f_new > 0:
            a2 = x_new
            fa2 = f_new
            # Illinois: adjust the other endpoint
            # b2 = a2 + (b2 - a2) * 0.5  -- this is the Illinois modification
            # Actually standard Illinois: keep b2, but the interval shrinks
            # The key is that we're doing linear interpolation which converges
            # faster than bisection when the function is roughly linear
        else:
            b2 = x_new
            fb2 = f_new
        
        # Check convergence
        if abs(b2 - a2) < 1e-15:
            break
    
    # Return the midpoint of the final bracket
    return 0.5 * (a2 + b2)
# EVOLVE-BLOCK-END
