"""Root finder for 3 fixed oscillatory equations."""
# EVOLVE-BLOCK-START
import math
import time

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8.
    Uses a combination of:
    1. Initial scan to bracket a root
    2. Bisection for guaranteed convergence
    3. Secant/Newton refinement for speed
    """
    start_time = time.time()
    deadline = start_time + 30.0  # Safety margin before 150s kill
    
    x0 = float(x0)
    best_x = x0
    best_fx = f(x0)
    best_abs = abs(best_fx)
    
    if best_abs <= 1e-8:
        return best_x
    
    # Phase 1: Scan to find a bracket [a, b] where f(a)*f(b) < 0
    # Try multiple scales and directions
    def time_ok():
        return time.time() < deadline
    
    # Generate candidate points around x0
    # Use logarithmic spacing to cover a wide range
    base_scale = max(1.0, abs(x0))
    
    # Try steps of various magnitudes
    step_scales = [1e-8, 1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0]
    
    bracket = None
    for scale in step_scales:
        if not time_ok():
            break
        step = scale * base_scale
        if step == 0:
            step = 1e-10
        
        # Try a few points in both directions
        for direction in [1, -1]:
            for i in range(1, 50):
                if not time_ok():
                    break
                x = x0 + direction * i * step
                fx = f(x)
                
                if not math.isfinite(fx):
                    continue
                
                abs_fx = abs(fx)
                if abs_fx < best_abs:
                    best_abs = abs_fx
                    best_x = x
                    best_fx = fx
                    if best_abs <= 1e-8:
                        return best_x
                
                # Check if we've bracketed a root
                if best_fx * fx < 0:
                    # We have a sign change between best_x and x
                    bracket = (best_x, x, best_fx, fx)
                    break
                elif fx * best_fx < 0:
                    bracket = (x, best_x, fx, best_fx)
                    break
            
            if bracket:
                break
        
        if bracket:
            break
    
    # If no bracket found, try a broader search
    if not bracket:
        # Try expanding the search range
        for scale in [1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0, 1e4, 1e5]:
            if not time_ok():
                break
            step = scale * base_scale
            if step == 0:
                step = 1e-10
            
            for direction in [1, -1]:
                for i in range(1, 200):
                    if not time_ok():
                        break
                    x = x0 + direction * i * step
                    fx = f(x)
                    
                    if not math.isfinite(fx):
                        continue
                    
                    abs_fx = abs(fx)
                    if abs_fx < best_abs:
                        best_abs = abs_fx
                        best_x = x
                        best_fx = fx
                        if best_abs <= 1e-8:
                            return best_x
                    
                    # Check sign change with any previously seen point
                    # Use x0 as reference
                    f_x0 = f(x0) if not hasattr(find_root, '_cached_f_x0') else find_root._cached_f_x0
                    if f_x0 * fx < 0:
                        bracket = (x0, x, f_x0, fx)
                        break
                    else:
                        # Check with best_x
                        if best_fx * fx < 0:
                            bracket = (best_x, x, best_fx, fx)
                            break
                
                if bracket:
                    break
            
            if bracket:
                break
    
    if not time_ok():
        return best_x
    
    # Phase 2: If we have a bracket, use bisection
    if bracket:
        a, b = bracket[0], bracket[1]
        fa, fb = bracket[2], bracket[3]
        
        # Bisection
        for _ in range(100):
            if not time_ok():
                break
            mid = 0.5 * (a + b)
            fmid = f(mid)
            
            if not math.isfinite(fmid):
                break
            
            abs_fmid = abs(fmid)
            if abs_fmid < best_abs:
                best_abs = abs_fmid
                best_x = mid
                best_fx = fmid
                if best_abs <= 1e-8:
                    return best_x
            
            if fa * fmid <= 0:
                b = mid
                fb = fmid
            else:
                a = mid
                fa = fmid
            
            if abs(b - a) < 1e-14 * max(1.0, abs(a), abs(b)):
                break
        
        # After bisection, try secant/Newton for faster convergence
        # Use the two endpoints of the final bracket
        a_final = a
        b_final = b
        fa_final = f(a_final)
        fb_final = f(b_final)
        
        # Secant method iterations
        x1, x2 = a_final, b_final
        f1, f2 = fa_final, fb_final
        
        for _ in range(50):
            if not time_ok():
                break
            if f2 == f1:
                break
            x3 = x2 - f2 * (x2 - x1) / (f2 - f1)
            f3 = f(x3)
            
            if not math.isfinite(f3) or not math.isfinite(x3):
                break
            
            abs_f3 = abs(f3)
            if abs_f3 < best_abs:
                best_abs = abs_f3
                best_x = x3
                best_fx = f3
                if best_abs <= 1e-8:
                    return best_x
            
            x1, f1 = x2, f2
            x2, f2 = x3, f3
    
    else:
        # No bracket found; use a local minimization approach
        # Try to minimize |f(x)| using a simple approach
        # Use golden section search on |f(x)| in a local neighborhood
        
        # Try several local neighborhoods
        for scale in [1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0]:
            if not time_ok():
                break
            width = scale * base_scale
            if width == 0:
                width = 1e-10
            
            a = x0 - width
            b = x0 + width
            
            # Golden section search
            gr = (math.sqrt(5) - 1) / 2
            c = b - gr * (b - a)
            d = a + gr * (b - a)
            
            fc = abs(f(c))
            fd = abs(f(d))
            
            if fc < best_abs:
                best_abs = fc
                best_x = c
            if fd < best_abs:
                best_abs = fd
                best_x = d
            
            for _ in range(80):
                if not time_ok():
                    break
                if fc < fd:
                    b = d
                    d = c
                    fd = fc
                    c = b - gr * (b - a)
                    fc = abs(f(c))
                    if fc < best_abs:
                        best_abs = fc
                        best_x = c
                else:
                    a = c
                    c = d
                    fc = fd
                    d = a + gr * (b - a)
                    fd = abs(f(d))
                    if fd < best_abs:
                        best_abs = fd
                        best_x = d
                
                if best_abs <= 1e-8:
                    return best_x
                
                if abs(b - a) < 1e-14 * max(1.0, abs(a), abs(b)):
                    break
        
        # If still not good enough, try Newton's method from best_x
        x = best_x
        fx = best_fx
        for _ in range(30):
            if not time_ok():
                break
            h = 1e-7 * max(1.0, abs(x))
            d = (f(x + h) - fx) / h
            if not math.isfinite(d) or d == 0:
                break
            x_new = x - fx / d
            if not math.isfinite(x_new):
                break
            fx_new = f(x_new)
            if not math.isfinite(fx_new):
                break
            if abs(fx_new) < best_abs:
                best_abs = abs(fx_new)
                best_x = x_new
                best_fx = fx_new
                if best_abs <= 1e-8:
                    return best_x
            x, fx = x_new, fx_new
    
    return best_x
# EVOLVE-BLOCK-END
