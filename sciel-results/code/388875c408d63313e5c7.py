import time
import math
import numpy as np

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin under 120s
    
    x = float(x0)
    
    # Helper to safely evaluate f
    def safe_eval(z):
        try:
            val = f(z)
            if not math.isfinite(val):
                return None
            return val
        except Exception:
            return None

    # Get first evaluation to establish baseline
    fx = safe_eval(x)
    if fx is None:
        return x
    if abs(fx) <= 1e-8:
        return x
        
    best_x = x
    best_fx = fx
    call_count = 1

    # Strategy: Cubic Interpolation
    # Choose 4 points around x0
    # Adaptive step size
    h = 1e-4 * max(1.0, abs(x))
    
    # Symmetric points for better numerical conditioning of the interpolation
    xs = [
        x - 1.5 * h,
        x - 0.5 * h,
        x + 0.5 * h,
        x + 1.5 * h
    ]
    
    ys = []
    for xi in xs:
        if time.time() > deadline:
            return best_x
        yi = safe_eval(xi)
        if yi is None:
            # If any evaluation fails, fall back to simple secant from x
            break
        ys.append(yi)
        call_count += 1
        if abs(yi) < abs(best_fx):
            best_x = xi
            best_fx = yi
        if abs(yi) <= 1e-8:
            return xi

    if len(ys) == 4:
        # All 4 evaluations successful
        try:
            # Fit a cubic polynomial
            coeffs = np.polyfit(xs, ys, 3)
            # Find roots
            roots = np.roots(coeffs)
            
            # Filter real roots
            real_roots = [r.real for r in roots if abs(r.imag) < 1e-10]
            
            if real_roots:
                # Pick the real root closest to x0
                root_candidate = min(real_roots, key=lambda r: abs(r - x))
                
                # Verify the candidate
                if time.time() > deadline:
                    return best_x
                    
                fx_cand = safe_eval(root_candidate)
                call_count += 1
                
                if fx_cand is not None:
                    if abs(fx_cand) < abs(best_fx):
                        best_x = root_candidate
                        best_fx = fx_cand
                    if abs(fx_cand) <= 1e-8:
                        return root_candidate
                    else:
                        # Cubic approximation was not accurate enough.
                        # Fall back to Secant method starting from best_x
                        # We have used 5 calls. We have plenty left.
                        # Use root_candidate as the new starting point if it's closer, 
                        # or just use best_x.
                        # Let's start secant from best_x and a neighbor.
                        
                        # We can reuse one of the previous points for secant to save a call?
                        # Secant needs (x1, f1), (x2, f2).
                        # We have (best_x, best_fx).
                        # We need another point.
                        # Let's pick the point from xs with the smallest |f| that is not best_x.
                        # Or just evaluate a new point.
                        
                        # To keep it simple and robust, let's just run a standard secant loop
                        # starting with two points.
                        # Point 1: best_x, best_fx
                        # Point 2: Let's use x + h if not already evaluated, or just evaluate x + h.
                        
                        # Actually, we already evaluated x +/- 0.5h and 1.5h.
                        # Let's find the second best point from our evaluations to use as the second secant point.
                        # This saves a call.
                        
                        candidates = [(x, fx)] + list(zip(xs, ys))
                        # Sort by |f|
                        candidates.sort(key=lambda t: abs(t[1]))
                        
                        # best_x is candidates[0][0]
                        # We need a second point. Let's take candidates[1] if it's distinct.
                        x2 = candidates[1][0]
                        fx2 = candidates[1][1]
                        
                        if x2 == best_x:
                             # Fallback if duplicates, just use x + h
                             x2 = x + h
                             fx2 = safe_eval(x2)
                             call_count += 1
                             if fx2 is None:
                                 return best_x
                        else:
                             # We already have fx2, no new call needed for this setup
                             pass

                        # Secant Loop
                        # Current points: (best_x, best_fx) and (x2, fx2)
                        # Note: best_fx is f(best_x)
                        
                        for i in range(400): # Max 400 more calls
                            if time.time() > deadline:
                                return best_x
                            
                            denom = fx2 - best_fx
                            if abs(denom) < 1e-30:
                                # Stuck, perturb
                                h_perturb = 1e-3 * max(1.0, abs(best_x))
                                x_new = best_x + h_perturb
                                fx_new = safe_eval(x_new)
                                call_count += 1
                                if fx_new is None:
                                    return best_x
                                if abs(fx_new) < abs(best_fx):
                                    best_x = x_new
                                    best_fx = fx_new
                                if abs(fx_new) <= 1e-8:
                                    return x_new
                                x2, fx2 = x_new, fx_new
                                continue
                                
                            x_new = x2 - fx2 * (x2 - best_x) / denom
                            if not math.isfinite(x_new):
                                return best_x
                                
                            fx_new = safe_eval(x_new)
                            call_count += 1
                            if fx_new is None:
                                return best_x
                                
                            if abs(fx_new) < abs(best_fx):
                                best_x = x_new
                                best_fx = fx_new
                            if abs(fx_new) <= 1e-8:
                                return x_new
                                
                            # Shift points
                            best_x, best_fx = x2, fx2
                            x2, fx2 = x_new, fx_new
                            
        except Exception:
            # If interpolation fails, fall back to simple secant
            pass

    # If we reached here, either interpolation failed or we didn't get 4 points.
    # Fall back to a simple secant method from x0
    # We have best_x, best_fx.
    # Get a second point
    h2 = 1e-4 * max(1.0, abs(best_x))
    x2 = best_x + h2
    fx2 = safe_eval(x2)
    call_count += 1
    if fx2 is None:
        return best_x
    if abs(fx2) < abs(best_fx):
        best_x = x2
        best_fx = fx2
    if abs(fx2) <= 1e-8:
        return x2

    # Secant loop
    for i in range(400):
        if time.time() > deadline:
            return best_x
        
        denom = fx2 - best_fx
        if abs(denom) < 1e-30:
            h_perturb = 1e-3 * max(1.0, abs(best_x))
            x_new = best_x + h_perturb
            fx_new = safe_eval(x_new)
            call_count += 1
            if fx_new is None:
                return best_x
            if abs(fx_new) < abs(best_fx):
                best_x = x_new
                best_fx = fx_new
            if abs(fx_new) <= 1e-8:
                return x_new
            x2, fx2 = x_new, fx_new
            continue
            
        x_new = x2 - fx2 * (x2 - best_x) / denom
        if not math.isfinite(x_new):
            return best_x
            
        fx_new = safe_eval(x_new)
        call_count += 1
        if fx_new is None:
            return best_x
            
        if abs(fx_new) < abs(best_fx):
            best_x = x_new
            best_fx = fx_new
        if abs(fx_new) <= 1e-8:
            return x_new
            
        # Shift
        best_x, best_fx = x2, fx2
        x2, fx2 = x_new, fx_new

    return best_x
