"""Root finder for 3 fixed oscillatory equations with optimized call budget."""
import math
import time

# EVOLVE-BLOCK-START

def find_root(f, x0):
    """
    Find x with |f(x)| <= 1e-8.
    
    Strategy:
    1. Local bracketing search with adaptive geometric steps.
    2. Illinois-style hybrid Secant/Bisection convergence.
    3. Final high-precision polish pass with multiple attempts.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    tol = 1e-8
    
    x0 = float(x0)
    
    # Function call cache to avoid redundant evaluations
    cache = {}
    call_count = 0
    
    def eval_f(x):
        nonlocal call_count
        # Use a rounded key for cache to handle floating point issues
        key = round(x, 15)
        if key in cache:
            return cache[key]
        call_count += 1
        try:
            result = f(x)
        except Exception:
            cache[key] = float('nan')
            return float('nan')
        if not math.isfinite(result):
            cache[key] = result
        else:
            cache[key] = result
        return result
    
    # 1. Initial Evaluation
    f0 = eval_f(x0)
    
    if not math.isfinite(f0):
        # Try a slight offset
        x_try = x0 + 1e-6
        f0 = eval_f(x_try)
        if not math.isfinite(f0):
            return x_try
        x0 = x_try
        
    if abs(f0) <= tol:
        return x0

    # 2. Bracketing Search
    # Adaptive geometric steps: start small, grow if needed
    # Try both directions with increasing magnitudes
    deltas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0, 5.0, 10.0, 50.0]
    
    a = None
    b = None
    fa = None
    fb = None
    
    # Track best point found
    best_x = x0
    best_f = f0
    best_abs = abs(f0)
    
    def update_best(x, fx):
        nonlocal best_x, best_f, best_abs
        if math.isfinite(x) and math.isfinite(fx):
            af = abs(fx)
            if af < best_abs:
                best_x = x
                best_f = fx
                best_abs = af

    # Also track points with sign changes for later use
    sign_points = [(x0, f0)]
    
    for d in deltas:
        if time.time() > deadline:
            break
            
        x_left = x0 - d
        x_right = x0 + d
        
        # Evaluate left
        fl = eval_f(x_left)
        if math.isfinite(fl):
            update_best(x_left, fl)
            sign_points.append((x_left, fl))
            if abs(fl) <= tol:
                return x_left
            # Check sign change with x0
            if f0 * fl < 0:
                a, fa = x_left, fl
                b, fb = x0, f0
                break
        
        # Evaluate right
        fr = eval_f(x_right)
        if math.isfinite(fr):
            update_best(x_right, fr)
            sign_points.append((x_right, fr))
            if abs(fr) <= tol:
                return x_right
            # Check sign change with x0
            if f0 * fr < 0:
                a, fa = x0, f0
                b, fb = x_right, fr
                break

    # If we found a bracket, use Illinois-style Hybrid Secant/Bisection
    if a is not None and b is not None:
        # Ensure a < b
        if a > b:
            a, b = b, a
            fa, fb = fb, fa
            
        # Illinois Hybrid Iteration
        # When secant goes outside bracket, halve the step (Illinois algorithm)
        for _ in range(60):
            if time.time() > deadline:
                break
                
            # Check current endpoints
            if abs(fa) <= tol: return a
            if abs(fb) <= tol: return b
            
            # Calculate Secant Point
            denom = fb - fa
            if abs(denom) < 1e-300 or not math.isfinite(denom):
                # Fallback to Bisection
                x_new = (a + b) / 2
            else:
                x_secant = b - fb * (b - a) / denom
                
                # Check if Secant point is inside bracket
                if a < x_secant < b:
                    x_new = x_secant
                else:
                    # Illinois: use bisection but also adjust the function value
                    # to dampen the next secant step
                    x_new = (a + b) / 2
                    # Illinois damping: replace the function value at the endpoint
                    # that was just replaced with half its value
                    if fa * fb > 0:
                        # This shouldn't happen if bracket is valid, but safety
                        pass
                    # The actual Illinois method: when we bisect, we set
                    # f(new_endpoint) = f(old_endpoint) / 2 for the side we're keeping
                    # We'll implement this by tracking a "damped" flag
                    # For simplicity, just use bisection here
                    pass
                    
            if not math.isfinite(x_new):
                break
                
            f_new = eval_f(x_new)
            
            if not math.isfinite(f_new):
                break
                
            update_best(x_new, f_new)
            
            if abs(f_new) <= tol:
                return x_new
                
            # Update Bracket
            if fa * f_new < 0:
                b, fb = x_new, f_new
            else:
                a, fa = x_new, f_new
                
            # Convergence check
            if abs(b - a) < 1e-14 * max(1.0, abs(a), abs(b)):
                break
        
        # Polish pass: Multiple attempts to tighten the solution
        # Try polishing with different bracket sizes
        polish_brackets = []
        
        # If we have a valid bracket, use it
        if a is not None and b is not None:
            polish_brackets.append((a, b, fa, fb))
        
        # Also try a tiny bracket around best_x
        for eps in [1e-10, 1e-9, 1e-8]:
            pa, pb = best_x - eps, best_x + eps
            pfa = eval_f(pa)
            pfb = eval_f(pb)
            if math.isfinite(pfa) and math.isfinite(pfb) and pfa * pfb < 0:
                polish_brackets.append((pa, pb, pfa, pfb))
                break
        
        # Also try bracket around the midpoint of a,b if we have one
        if a is not None and b is not None:
            mid = (a + b) / 2
            for eps in [1e-10, 1e-9]:
                pa, pb = mid - eps, mid + eps
                pfa = eval_f(pa)
                pfb = eval_f(pb)
                if math.isfinite(pfa) and math.isfinite(pfb) and pfa * pfb < 0:
                    polish_brackets.append((pa, pb, pfa, pfb))
                    break
        
        # Run polish bisection on the best bracket
        for (pa, pb, pfa, pfb) in polish_brackets:
            if not (math.isfinite(pa) and math.isfinite(pb) and 
                    math.isfinite(pfa) and math.isfinite(pfb) and 
                    pfa * pfb < 0):
                continue
                
            for _ in range(40):
                if time.time() > deadline:
                    break
                if abs(pfa) <= tol: return pa
                if abs(pfb) <= tol: return pb
                    
                x_mid = (pa + pb) / 2
                f_mid = eval_f(x_mid)
                
                if not math.isfinite(f_mid):
                    break
                    
                update_best(x_mid, f_mid)
                if abs(f_mid) <= tol:
                    return x_mid
                    
                if pfa * f_mid < 0:
                    pb, pfb = x_mid, f_mid
                else:
                    pa, pfa = x_mid, f_mid
                
                if abs(pb - pa) < 1e-15 * max(1.0, abs(pa)):
                    break
        
        return best_x if best_abs <= tol else (a + b) / 2

    else:
        # No bracket found.
        # Try Secant with the two best points
        x1 = x0
        f1 = f0
        
        # Find a second point from sign_points
        x2 = None
        f2 = None
        min_abs = float('inf')
        
        for px, pf in sign_points:
            if px != x0 and math.isfinite(pf):
                if abs(pf) < min_abs:
                    min_abs = abs(pf)
                    x2 = px
                    f2 = pf
        
        if x2 is None:
            # Fallback: try specific offsets
            for offset in [1e-2, -1e-2, 1.0, -1.0, 10.0, -10.0]:
                x_try = x0 + offset
                f_try = eval_f(x_try)
                if math.isfinite(f_try):
                    update_best(x_try, f_try)
                    x2 = x_try
                    f2 = f_try
                    break
        
        if x2 is None:
            return best_x

        # Run Secant with fallback to bisection if it diverges
        secant_x1, secant_f1 = x1, f1
        secant_x2, secant_f2 = x2, f2
        
        for _ in range(40):
            if time.time() > deadline:
                break
                
            if abs(secant_f1) <= tol: return secant_x1
            if abs(secant_f2) <= tol: return secant_x2
            
            denom = secant_f2 - secant_f1
            if abs(denom) < 1e-300 or not math.isfinite(denom):
                break
                
            x_new = secant_x2 - secant_f2 * (secant_x2 - secant_x1) / denom
            
            if not math.isfinite(x_new):
                break
                
            f_new = eval_f(x_new)
            
            if not math.isfinite(f_new):
                break
                
            update_best(x_new, f_new)
            
            if abs(f_new) <= tol:
                return x_new
            
            # Check for divergence: if x_new is too far from both points, reset
            dist1 = abs(x_new - secant_x1)
            dist2 = abs(x_new - secant_x2)
            if dist1 > 100 * max(1.0, abs(secant_x1)) and dist2 > 100 * max(1.0, abs(secant_x2)):
                # Diverging, try to find a bracket from cached points
                break
                
            secant_x1, secant_f1 = secant_x2, secant_f2
            secant_x2, secant_f2 = x_new, f_new
        
        # Try to find a bracket from all cached points
        # Sort points by x and look for sign changes
        cached_points = [(k, v) for k, v in cache.items() if math.isfinite(v)]
        if len(cached_points) >= 2:
            cached_points.sort(key=lambda p: p[0])
            for i in range(len(cached_points) - 1):
                xa, fa_cached = cached_points[i]
                xb, fb_cached = cached_points[i + 1]
                if fa_cached * fb_cached < 0:
                    # Found a bracket from cache
                    pa, pb = xa, xb
                    pfa, pfb = fa_cached, fb_cached
                    for _ in range(40):
                        if time.time() > deadline:
                            break
                        if abs(pfa) <= tol: return pa
                        if abs(pfb) <= tol: return pb
                        
                        x_mid = (pa + pb) / 2
                        f_mid = eval_f(x_mid)
                        
                        if not math.isfinite(f_mid):
                            break
                        
                        update_best(x_mid, f_mid)
                        if abs(f_mid) <= tol:
                            return x_mid
                        
                        if pfa * f_mid < 0:
                            pb, pfb = x_mid, f_mid
                        else:
                            pa, pfa = x_mid, f_mid
                        
                        if abs(pb - pa) < 1e-15 * max(1.0, abs(pa)):
                            break
                    break
        
        # Final polish attempt
        for eps in [1e-10, 1e-9, 1e-8]:
            pa, pb = best_x - eps, best_x + eps
            pfa = eval_f(pa)
            pfb = eval_f(pb)
            if math.isfinite(pfa) and math.isfinite(pfb) and pfa * pfb < 0:
                for _ in range(40):
                    if time.time() > deadline:
                        break
                    if abs(pfa) <= tol: return pa
                    if abs(pfb) <= tol: return pb
                    
                    x_mid = (pa + pb) / 2
                    f_mid = eval_f(x_mid)
                    
                    if not math.isfinite(f_mid):
                        break
                    
                    update_best(x_mid, f_mid)
                    if abs(f_mid) <= tol:
                        return x_mid
                    
                    if pfa * f_mid < 0:
                        pb, pfb = x_mid, f_mid
                    else:
                        pa, pfa = x_mid, f_mid
                    
                    if abs(pb - pa) < 1e-15 * max(1.0, abs(pa)):
                        break
                break

        return best_x

# EVOLVE-BLOCK-END
