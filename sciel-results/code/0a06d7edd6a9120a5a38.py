import numpy as np
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 110.0
    
    x0 = float(x0)
    
    # 1. Cubic Reconstruction
    h = max(1.0, abs(x0) * 0.1)
    if h == 0: h = 1.0
    
    # Symmetric points
    points = [x0 - 1.5*h, x0 - 0.5*h, x0 + 0.5*h, x0 + 1.5*h]
    
    try:
        vals = [f(p) for p in points]
        
        if not all(np.isfinite(v) for v in vals):
            raise ValueError("Non-finite")
            
        # Fit cubic
        coeffs = np.polyfit(points, vals, 3)
        roots = np.roots(coeffs)
        
        real_roots = [r.real for r in roots if abs(r.imag) < 1e-9]
        
        if real_roots:
            # Pick root closest to x0
            best_root = min(real_roots, key=lambda r: abs(r - x0))
            
            # Verify
            fx = f(best_root)
            if np.isfinite(fx) and abs(fx) <= 1e-8:
                return best_root
            
            # If close, use as start for Secant? 
            # Or just proceed to Secant with existing points.
            
    except Exception:
        pass

    # 2. Secant Method using the 4 points evaluated above
    # We have (points, vals). 
    # If the try block failed early (e.g. non-finite), we might not have all 4.
    # Let's assume we have at least 2 valid points.
    
    # If we didn't get to evaluate all 4, we need to handle that.
    # But the try block evaluates all 4 at once. If one fails, the list comprehension fails?
    # No, `vals = [f(p) for p in points]` will raise an exception if f raises.
    # If f returns non-finite, the check `if not all...` raises.
    # So if we reach here, we likely have `points` and `vals` defined and valid, 
    # OR the exception happened before `vals` was fully populated?
    # No, list comprehension is atomic. If it fails, `vals` is not assigned or partial?
    # In Python, if an exception is raised in a list comprehension, the variable is not assigned.
    # So we need to be careful.
    
    # Let's restructure to ensure we have at least 2 points for Secant.
    
    # If we are here, either:
    # a) Cubic root found but not accurate enough. We have 4 points + 1 verification call.
    # b) Cubic root not real. We have 4 points.
    # c) Exception in cubic logic. We might have 0-4 points.
    
    # To be safe, let's collect points in a list.
    # Actually, the current code structure is risky if `f` raises.
    # Let's assume `f` returns floats. The task says "f takes and returns a float".
    
    # Let's just use the `points` and `vals` from the try block if available.
    # If not, we need to generate new points.
    
    # To simplify, let's just restart the search with a standard Secant from x0 if the cubic block fails completely.
    # But we want to reuse calls.
    
    # Let's assume the try block succeeded in evaluating `vals`.
    # If it didn't, we don't have `vals`.
    
    # Let's use a helper to get 2 points.
    
    # If `vals` is not defined, we need to evaluate.
    # Let's just do a robust Secant from scratch if the cubic block didn't produce usable data.
    
    # However, the most likely case is that `vals` IS defined.
    
    # Let's check if `vals` is in scope.
    # In Python, variables defined in a try block are accessible in the except/else/subsequent code if the assignment happened.
    # If `vals = ...` raised, `vals` is undefined.
    
    # Let's just initialize a list of known points.
    known_points = []
    
    # Re-evaluate logic:
    # The try block above might have failed.
    # Let's just do a fresh Secant start if we don't have points.
    # But we want to minimize calls.
    
    # Let's assume the cubic block ran and we have `points` and `vals`.
    # If `vals` is not defined, we define it empty.
    
    if 'vals' not in locals() or not vals:
        # Fallback: generate 2 points
        x1 = x0
        y1 = f(x1)
        x2 = x0 + max(1.0, abs(x0)*0.1)
        y2 = f(x2)
        known_points = [(x1, y1), (x2, y2)]
    else:
        # Use the 4 points from cubic attempt
        # Filter out any non-finite
        valid_pts = [(p, v) for p, v in zip(points, vals) if np.isfinite(v)]
        if len(valid_pts) >= 2:
            known_points = valid_pts
        else:
            # Not enough valid points, generate 2
            x1 = x0
            y1 = f(x1)
            x2 = x0 + max(1.0, abs(x0)*0.1)
            y2 = f(x2)
            known_points = [(x1, y1), (x2, y2)]

    # Secant Method
    # Sort known_points by |y| to pick the best two
    known_points.sort(key=lambda kv: abs(kv[1]))
    
    # Take the two best
    (xa, ya), (xb, yb) = known_points[0], known_points[1]
    
    # If they are the same point, we need a new one
    if xa == xb:
         xb = xa + max(1.0, abs(xa)*0.1)
         yb = f(xb)
         
    for _ in range(50):
        if time.time() > deadline:
            break
            
        # Secant step
        if yb == ya:
            break
            
        xc = xb - yb * (xb - xa) / (yb - ya)
        
        if not np.isfinite(xc):
            break
            
        yc = f(xc)
        
        if not np.isfinite(yc):
            break
            
        if abs(yc) <= 1e-8:
            return xc
            
        # Update points: keep the two with smallest |y|
        # Points: (xa, ya), (xb, yb), (xc, yc)
        # We want to discard the worst one.
        # Usually, we discard xa (the older one) if the function is monotonic, 
        # but for general polynomials, keeping the two smallest |y| is a good heuristic (Illinois method variant).
        
        new_pts = [(xa, ya), (xb, yb), (xc, yc)]
        new_pts.sort(key=lambda kv: abs(kv[1]))
        
        # If xc is the best, we keep it.
        # We discard the worst (index 2).
        # The remaining two become the new xa, xb.
        # Note: We need to ensure xa and xb are distinct.
        
        # Let's just pick the two smallest.
        (xa, ya), (xb, yb) = new_pts[0], new_pts[1]
        
        # If they are identical, nudge xb
        if xa == xb:
             xb = xa + max(1.0, abs(xa)*0.1)
             yb = f(xb)
             
    # Return the best candidate found
    # Check all known points + xc?
    # Just return the one with smallest |f| from the last iteration
    # We have xa, xb, xc (last evaluated)
    # Let's check xa, xb, and the last xc if it exists.
    
    # To be safe, just return the best of the last few.
    # For simplicity, return xb (which is one of the best two).
    # Or check f(xa), f(xb) and return the min.
    
    # Since we might have exited loop early, let's just return the best of xa, xb.
    # But we might have a better xc that didn't meet 1e-8 but is better than xa/xb.
    # Let's just return xc if it was the last one and it's finite.
    
    # Actually, the loop returns if it finds a solution.
    # If it finishes, it returns the last xb.
    # Let's refine the return.
    
    # Collect all candidates: xa, xb, and the last xc if available.
    # Since we don't store history, let's just return xb.
    # But xb is one of the two smallest |y| in the last step.
    
    return xb

# EVOLVE-BLOCK-END
