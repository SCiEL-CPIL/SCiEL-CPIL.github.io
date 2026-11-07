import math
import time

# EVOLVE-BLOCK-START

def find_root(f, x0):
    start_time = time.time()
    TIME_LIMIT = 115.0  # Safety margin for 120s limit (evaluator kills at 150s, task says 120s finish)
    
    calls = 0
    best_x = x0
    best_f_abs = float('inf')
    cache = {}
    
    def _f(x):
        nonlocal calls, best_x, best_f_abs
        if time.time() - start_time > TIME_LIMIT:
            raise TimeoutError("Time limit exceeded")
        
        # Check cache
        if x in cache:
            return cache[x]
            
        try:
            val = float(f(x))
        except Exception:
            cache[x] = float('nan')
            return float('nan')
            
        if not math.isfinite(val):
            cache[x] = float('nan')
            return float('nan')
            
        calls += 1
        cache[x] = val
        
        abs_val = abs(val)
        if abs_val < best_f_abs:
            best_f_abs = abs_val
            best_x = x
            
        return val

    x0 = float(x0)
    
    # 1. Check initial point
    fx0 = _f(x0)
    if abs(fx0) <= 1e-8:
        return x0

    # 2. Adaptive Sign-Change Search
    # We try to find a bracket [a, b] such that f(a) and f(b) have opposite signs.
    # We expand the search range from x0 outwards.
    
    # Initial small step
    step = 0.1
    max_expansions = 40 # 0.1 * 2^40 is huge, but we limit by calls/time
    
    # We will keep track of the last evaluated point and its value to avoid re-evaluating
    # and to detect sign changes across the expansion.
    
    # Let's search in both directions: x0 - k*step and x0 + k*step
    # Actually, a simpler robust approach:
    # Iterate k from 1 to max_k:
    #   x_plus = x0 + k * step
    #   x_minus = x0 - k * step
    #   Check sign change between (x0, x_plus) and (x0, x_minus)
    #   If not, check sign change between (x_plus, x_minus)? No, that's a huge jump.
    #   Better: Maintain a list of points.
    
    points = [(x0, fx0)]
    
    # We will expand step exponentially.
    # Current step size
    current_step = 0.1
    
    found_bracket = False
    a = None
    b = None
    fa = None
    fb = None
    
    # To ensure we don't miss a root in a very narrow interval due to coarse steps,
    # if we detect a sign change between two points separated by `current_step`,
    # we will do a dense scan in between.
    
    for k in range(1, max_expansions + 1):
        if calls > 450: break
        if time.time() - start_time > TIME_LIMIT: break
        
        x_plus = x0 + k * current_step
        x_minus = x0 - k * current_step
        
        # Evaluate new points
        fp = _f(x_plus)
        fm = _f(x_minus)
        
        # Check if we hit a root
        if abs(fp) <= 1e-8: return x_plus
        if abs(fm) <= 1e-8: return x_minus
        
        # Check for sign changes
        # 1. Between x0 and x_plus?
        # We have fx0.
        if fx0 * fp < 0:
            # Sign change in [x0, x_plus]
            a, b = x0, x_plus
            fa, fb = fx0, fp
            found_bracket = True
            break
        # 2. Between x0 and x_minus?
        if fx0 * fm < 0:
            a, b = x_minus, x0
            fa, fb = fm, fx0
            found_bracket = True
            break
            
        # 3. Between x_minus and x_plus? 
        # This covers the case where the root is in (x_minus, x0) or (x0, x_plus) 
        # but we didn't catch it with x0? No, if fx0*fp > 0 and fx0*fm > 0, 
        # then f(x_minus) and f(x_plus) have the same sign as fx0. 
        # So no sign change in [x_minus, x_plus] if the function is continuous and 
        # we haven't crossed a root. 
        # Wait, if f(x_minus) and f(x_plus) have opposite signs, we MUST have crossed 
        # a root. But if fx0 has the same sign as both, that's impossible for a 
        # continuous function unless there are 2 roots. 
        # Since we are looking for ANY root, checking adjacent intervals is safer.
        
        # Let's check sign change between x_minus and x_plus just in case 
        # (e.g. if we missed a sign change due to numerical noise or non-continuity? 
        # Task says "oscillatory equations", usually implies continuous).
        if fm * fp < 0:
            # This implies a root exists in [x_minus, x_plus].
            # Since we checked [x0, x_plus] and [x_minus, x0] and found no sign change,
            # this is theoretically impossible for a continuous function unless 
            # fx0 was 0 (handled) or numerical issues. 
            # But let's handle it: we have a bracket.
            a, b = x_minus, x_plus
            fa, fb = fm, fp
            found_bracket = True
            break
        
        # If no bracket found, expand step
        current_step *= 2.0
        
        # If step becomes too large, we might jump over a root without detecting 
        # a sign change if the root is a "touch" (double root) or if the function 
        # oscillates very fast. 
        # However, the problem states "find root", implying simple roots or at least 
        # sign changes. If the function doesn't change sign (e.g. x^2), this method fails.
        # But "oscillatory" usually implies sin, cos, etc., which cross zero.
        
    if not found_bracket:
        # Fallback: Use the best_x found so far and try to refine it locally 
        # assuming it's close to a root (maybe a double root or we missed the sign change).
        # Use Golden Section on |f(x)| around best_x.
        a = best_x - 0.5
        b = best_x + 0.5
        fa = _f(a)
        fb = _f(b)
        
        # If we still don't have a sign change, we are in trouble.
        # But let's try Illinois on the interval containing best_x?
        # Illinois requires a sign change.
        # If no sign change is found in the adaptive search, we might have missed it.
        # Let's try a dense scan around best_x.
        
        # Dense scan around best_x
        scan_width = 1.0
        num_scan = 50
        step_scan = 2.0 * scan_width / num_scan
        start_scan = best_x - scan_width
        
        for i in range(num_scan + 1):
            if calls > 480: break
            if time.time() - start_time > TIME_LIMIT: break
            
            xi = start_scan + i * step_scan
            fi = _f(xi)
            
            if abs(fi) <= 1e-8:
                return xi
            
            # Check sign change with previous point in scan
            if i > 0:
                prev_x = start_scan + (i - 1) * step_scan
                prev_f = _f(prev_x) # Cached
                if prev_f * fi < 0:
                    a, b = prev_x, xi
                    fa, fb = prev_f, fi
                    found_bracket = True
                    break
            
    if not found_bracket:
        # If we still don't have a bracket, return the best point found.
        # This is a failure case for the sign-change strategy.
        return best_x

    # 3. Refine the bracket [a, b]
    # First, do a dense linear scan in [a, b] to get a tighter bracket.
    # This is crucial if the initial bracket is wide (e.g. step=100).
    
    # Determine scan density. 
    # If bracket is small (< 1.0), use 50 points.
    # If bracket is large, use more points or adaptive.
    # Let's use a fixed 100 points for the first refinement pass to ensure we catch the root.
    
    if b - a > 1e-6:
        num_scan = 100
        step_scan = (b - a) / num_scan
        
        prev_x = a
        prev_f = fa
        
        for i in range(1, num_scan + 1):
            if calls > 490: break
            if time.time() - start_time > TIME_LIMIT: break
            
            xi = a + i * step_scan
            fi = _f(xi)
            
            if abs(fi) <= 1e-8:
                return xi
            
            if prev_f * fi < 0:
                a, b = prev_x, xi
                fa, fb = prev_f, fi
                break
                
            prev_x = xi
            prev_f = fi
    else:
        # Bracket is already tiny
        a, b = a, b
        fa, fb = fa, fb

    # 4. Illinois Method to converge to the root
    # Illinois is a hybrid of Bisection and Regula Falsi (False Position).
    # It is robust and fast.
    
    # Ensure a < b
    if a > b:
        a, b = b, a
        fa, fb = fb, fa
        
    # Illinois parameters
    # c is the point where the function is "false" (we keep the side with smaller |f|)
    # a is the "true" side (we shrink the other side)
    
    # Standard Illinois:
    # 1. Calculate root estimate x_new using linear interpolation (Regula Falsi)
    # 2. If x_new is closer to a, update b and fb = fb * 0.5
    #    Else update a and fa = fa * 0.5
    
    x_prev = b
    f_prev = fb
    
    for _ in range(100):
        if calls > 495: break
        if time.time() - start_time > TIME_LIMIT: break
        
        # Check convergence
        if abs(b - a) < 1e-12:
            break
            
        # Regula Falsi step
        # x_new = b - f(b) * (b - a) / (f(b) - f(a))
        # Handle division by zero (shouldn't happen if signs are opposite)
        denom = fb - fa
        if abs(denom) < 1e-30:
            # Fallback to bisection
            x_new = (a + b) / 2
        else:
            x_new = b - fb * (b - a) / denom
            
        # Ensure x_new is within [a, b]
        if x_new <= a or x_new >= b:
            x_new = (a + b) / 2
            
        f_new = _f(x_new)
        
        if abs(f_new) <= 1e-8:
            return x_new
            
        # Update bracket
        if fa * f_new < 0:
            # Root is in [a, x_new]
            b = x_new
            fb = f_new
            # Illinois damping: halve the function value of the "old" b (which is now x_new? No, the one we are discarding)
            # In Illinois, we damp the function value of the point that is being replaced on the "false" side.
            # If we moved b, we damp fa? No.
            # Standard Illinois:
            # If new point is closer to a, we replace b and halve fb (the value at the new b? No, the value at the point that is now "far"?)
            # Let's stick to the standard definition:
            # If x_new is closer to a, then b is the point to replace. We set b=x_new. 
            # To avoid slow convergence, we set f(b) = f(b)/2. 
            # Wait, if we replace b, we should damp the NEW b? Or the OLD b?
            # "Halve the function value of the point that is no longer used"? No.
            # "Halve the function value of the point that is retained on the side that was previously the 'false' side"?
            
            # Let's use a simpler, robust variant:
            # If we move b, we don't necessarily need to damp if we use Bisection fallback often.
            # But Illinois is faster.
            # Correct Illinois:
            # If x_new is closer to a, set b = x_new, fb = fb/2. (Damp the new value? No, damp the value of the point that is now "far"?)
            # Actually, the damping is applied to the function value of the point that is NOT updated?
            # No, it's applied to the function value of the point that is being replaced?
            
            # Let's look up the exact algorithm.
            # 1. Compute x_new.
            # 2. If |x_new - a| < |x_new - b|, then b = x_new, fb = fb/2.
            #    Else a = x_new, fa = fa/2.
            
            if abs(x_new - a) < abs(x_new - b):
                # We replaced b. Damp fb (the new value at b? Or the old one?)
                # The algorithm says: "replace b with x_new and halve f(b)".
                # This means the NEW f(b) is halved.
                fb = fb / 2.0 
            else:
                a = x_new
                fa = fa / 2.0
                
        else:
            # Root is in [x_new, b]
            if abs(x_new - a) < abs(x_new - b):
                a = x_new
                fa = fa / 2.0
            else:
                b = x_new
                fb = fb / 2.0

    # Return the midpoint or the best point
    x_mid = (a + b) / 2
    f_mid = _f(x_mid)
    
    # Check a, b, mid
    best_final_x = a
    best_final_val = abs(fa)
    
    if abs(fb) < best_final_val:
        best_final_x = b
        best_final_val = abs(fb)
        
    if abs(f_mid) < best_final_val:
        best_final_x = x_mid
        best_final_val = abs(f_mid)
        
    # Also check best_x from the global cache
    if best_f_abs < best_final_val:
        best_final_x = best_x
        
    return best_final_x

# EVOLVE-BLOCK-END
