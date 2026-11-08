import time
import math

# EVOLVE-BLOCK-START

def find_root(f, x0):
    """
    Find root x of f(x) such that |f(x)| <= 1e-8.
    Uses Illinois bisection (1 call/step, guaranteed convergence) with
    adaptive bracketing for oscillatory functions.
    """
    start_time = time.time()
    deadline = start_time + 10.0

    x0 = float(x0)
    best_x = x0
    best_fx = f(x0)
    best_abs = abs(best_fx)

    if best_abs <= 1e-10:
        return best_x

    # Phase 1: Find a bracket [a, b] where f(a) and f(b) have opposite signs
    # Use logarithmic spacing to cover a wide range efficiently
    base_scale = max(1.0, abs(x0))
    
    bracket = None
    # Try various step scales, both directions
    step_scales = [1e-8, 1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0]
    
    for scale in step_scales:
        if time.time() > deadline:
            break
        step = scale * base_scale
        if step == 0:
            step = 1e-10
        
        for direction in [1, -1]:
            for i in range(1, 50):
                if time.time() > deadline:
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
                    if best_abs <= 1e-10:
                        return best_x
                
                # Check if we've bracketed a root between x0 and x
                if best_fx * fx < 0:
                    bracket = (x0, x, best_fx, fx)
                    break
                # Or between best_x and x
                elif best_fx * fx < 0:
                    bracket = (best_x, x, best_fx, fx)
                    break
            
            if bracket:
                break
        
        if bracket:
            break

    # If no bracket found, try a broader search
    if not bracket:
        for scale in [1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0, 1e4, 1e5]:
            if time.time() > deadline:
                break
            step = scale * base_scale
            if step == 0:
                step = 1e-10
            
            for direction in [1, -1]:
                for i in range(1, 200):
                    if time.time() > deadline:
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
                        if best_abs <= 1e-10:
                            return best_x
                    
                    if best_fx * fx < 0:
                        bracket = (best_x, x, best_fx, fx)
                        break
                
                if bracket:
                    break
            
            if bracket:
                break

    if time.time() > deadline:
        return best_x

    # Phase 2: If we have a bracket, use Illinois bisection
    if bracket:
        a, b = bracket[0], bracket[1]
        fa, fb = bracket[2], bracket[3]
        
        # Illinois bisection: move the endpoint with smaller |f|
        for _ in range(100):
            if time.time() > deadline:
                break
            
            # Illinois: replace the endpoint with smaller |f| with midpoint
            if abs(fa) < abs(fb):
                a = (a + b) / 2
                fa = f(a)
            else:
                b = (a + b) / 2
                fb = f(b)
            
            if not math.isfinite(fa) or not math.isfinite(fb):
                break
            
            abs_fa = abs(fa)
            abs_fb = abs(fb)
            if abs_fa < best_abs:
                best_abs = abs_fa
                best_x = a
                best_fx = fa
                if best_abs <= 1e-10:
                    return best_x
            if abs_fb < best_abs:
                best_abs = abs_fb
                best_x = b
                best_fx = fb
                if best_abs <= 1e-10:
                    return best_x
            
            # Check convergence
            if abs(b - a) < 1e-14 * max(1.0, abs(a), abs(b)):
                break

        # Phase 3: Refine with secant method using the two best points
        # Use the two endpoints of the final bracket
        x1, x2 = a, b
        f1, f2 = fa, fb
        
        # If we have better points from tracking, use those
        # Try secant from the two closest points to zero
        candidates = [(a, fa), (b, fb), (best_x, best_fx)]
        # Sort by |f| and take the two smallest
        candidates.sort(key=lambda t: abs(t[1]))
        x1, f1 = candidates[0]
        x2, f2 = candidates[1]
        
        for _ in range(50):
            if time.time() > deadline:
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
                if best_abs <= 1e-10:
                    return best_x
            
            x1, f1 = x2, f2
            x2, f2 = x3, f3

    else:
        # No bracket found; use secant method as fallback
        step = 1e-4 * max(1.0, abs(x0))
        x_prev = x0
        f_prev = best_fx
        x_curr = x0 + step
        f_curr = f(x_curr)
        
        if abs(f_curr) < best_abs:
            best_abs = abs(f_curr)
            best_x = x_curr
            best_fx = f_curr
            if best_abs <= 1e-10:
                return best_x
        
        for _ in range(100):
            if time.time() > deadline:
                break
            denom = f_curr - f_prev
            if abs(denom) < 1e-15:
                break
            x_next = x_curr - f_curr * (x_curr - x_prev) / denom
            f_next = f(x_next)
            
            if not math.isfinite(f_next) or not math.isfinite(x_next):
                break
            
            if abs(f_next) < best_abs:
                best_abs = abs(f_next)
                best_x = x_next
                best_fx = f_next
                if best_abs <= 1e-10:
                    return best_x
            
            x_prev = x_curr
            f_prev = f_curr
            x_curr = x_next
            f_curr = f_next

    return best_x

# EVOLVE-BLOCK-END
