"""Scalar root finder for 3 fixed smooth polynomial equations."""
import math
import time
import numpy as np

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find a root of f starting near x0, returning x with |f(x)| <= 1e-8.
    Uses single-shot polynomial reconstruction: evaluate f at 4 points,
    build cubic via Lagrange interpolation, solve analytically.
    """
    DEADLINE = 140.0  # Safety margin before 150s kill
    start_time = time.time()
    
    def check_deadline():
        return (time.time() - start_time) < DEADLINE
    
    def is_finite(x):
        return math.isfinite(x)
    
    def safe_f(x):
        try:
            val = f(x)
            if not is_finite(val):
                return None
            return float(val)
        except:
            return None
    
    # Track best solution found
    best_x = None
    best_fx = float('inf')
    
    def update_best(x, fx):
        nonlocal best_x, best_fx
        if is_finite(x) and is_finite(fx) and abs(fx) < abs(best_fx):
            best_x = x
            best_fx = fx
        if best_x is not None and abs(best_fx) <= 1e-8:
            return True
        return False
    
    # First try at x0
    x0 = float(x0)
    fx0 = safe_f(x0)
    if fx0 is not None:
        update_best(x0, fx0)
        if abs(fx0) <= 1e-8:
            return x0
    
    # Strategy 1: Single-shot cubic reconstruction
    # Evaluate f at 4 points near x0 to build a cubic polynomial
    if check_deadline():
        # Choose 4 distinct points around x0
        scale = max(1.0, abs(x0))
        # Use well-separated points for numerical stability
        points = [x0 - scale, x0 - 0.5*scale, x0 + 0.5*scale, x0 + scale]
        
        # Ensure x0 is included if not already (it's at the center)
        # Actually let's use x0 and 3 nearby points for better local accuracy
        offsets = [0.0, 1.0, -1.0, 2.0]
        points = [x0 + o * scale * 0.1 for o in offsets]
        
        # Make sure points are distinct
        for i in range(1, 4):
            if abs(points[i] - points[i-1]) < 1e-10:
                points[i] = points[i-1] + 1e-6
        
        fvals = []
        valid = True
        for px in points:
            fv = safe_f(px)
            if fv is None:
                valid = False
                break
            fvals.append(fv)
        
        if valid:
            # Build cubic polynomial via Lagrange interpolation
            # p(x) = sum_j f(x_j) * L_j(x)
            # L_j(x) = prod_{k != j} (x - x_k) / (x_j - x_k)
            # For 4 points, this gives a cubic
            try:
                xs = np.array(points)
                fs = np.array(fvals)
                
                # Compute polynomial coefficients using numpy
                # np.polyfit gives least squares, but for exact interpolation use:
                # Build the polynomial directly
                coeffs = np.zeros(4)
                for j in range(4):
                    # L_j(x) = prod_{k != j} (x - x_k) / (x_j - x_k)
                    lj_coeffs = np.array([1.0])
                    for k in range(4):
                        if k != j:
                            lj_coeffs = np.convolve(lj_coeffs, np.array([1.0, -xs[k]]))
                    lj_coeffs = lj_coeffs / np.prod(xs[j] - np.delete(xs, j))
                    coeffs += fs[j] * lj_coeffs
                
                # coeffs is [a3, a2, a1, a0] for cubic a3*x^3 + a2*x^2 + a1*x + a0
                # Find roots
                roots = np.roots(coeffs)
                
                # Find real roots
                real_roots = []
                for r in roots:
                    if abs(r.imag) < 1e-10:
                        real_roots.append(r.real)
                
                if real_roots:
                    # Find root nearest to x0
                    best_root = min(real_roots, key=lambda r: abs(r - x0))
                    f_best = safe_f(best_root)
                    if f_best is not None:
                        update_best(best_root, f_best)
                        if abs(f_best) <= 1e-8:
                            return best_root
                        
                        # If the cubic root isn't accurate enough, refine with Newton/secant
                        # Use the cubic as a guide but do a few Newton steps on f
                        x_refine = best_root
                        for _ in range(10):
                            if not check_deadline():
                                break
                            fx_ref = safe_f(x_refine)
                            if fx_ref is None:
                                break
                            if abs(fx_ref) <= 1e-8:
                                return x_refine
                            update_best(x_refine, fx_ref)
                            
                            # Estimate derivative from cubic
                            # p'(x) = 3*a3*x^2 + 2*a2*x + a1
                            dfx = 3*coeffs[0]*x_refine**2 + 2*coeffs[1]*x_refine + coeffs[2]
                            if abs(dfx) < 1e-15:
                                break
                            x_new = x_refine - fx_ref / dfx
                            if not is_finite(x_new):
                                break
                            if abs(x_new - x_refine) < 1e-14:
                                break
                            x_refine = x_new
                        
                        # Also try secant refinement from two nearby points
                        if check_deadline() and best_x is not None and abs(best_fx) > 1e-8:
                            x1 = x0
                            x2 = x0 + 1e-4 * max(1.0, abs(x0))
                            fx1 = safe_f(x1)
                            fx2 = safe_f(x2)
                            if fx1 is not None and fx2 is not None:
                                if abs(fx1) <= 1e-8:
                                    return x1
                                if abs(fx2) <= 1e-8:
                                    return x2
                                for _ in range(50):
                                    if not check_deadline():
                                        break
                                    if abs(fx2) <= 1e-8:
                                        return x2
                                    update_best(x2, fx2)
                                    if abs(fx1) <= 1e-8:
                                        return x1
                                    update_best(x1, fx1)
                                    denom = fx2 - fx1
                                    if denom == 0 or not is_finite(denom):
                                        break
                                    x_new = x2 - fx2 * (x2 - x1) / denom
                                    if not is_finite(x_new):
                                        break
                                    if abs(x_new - x1) < 1e-14:
                                        x_new = x1 + 1e-10
                                    if abs(x_new - x2) < 1e-14:
                                        x_new = x2 + 1e-10
                                    x1, fx1 = x2, fx2
                                    x2 = x_new
                                    fx2 = safe_f(x2)
                                    if fx2 is None:
                                        break
            except Exception:
                pass
    
    # Strategy 2: Fallback secant if cubic reconstruction failed
    if check_deadline() and best_x is not None and abs(best_fx) > 1e-8:
        x1 = x0
        offset = 1e-4 * max(1.0, abs(x0))
        x2 = x0 + offset
        fx1 = safe_f(x1)
        fx2 = safe_f(x2)
        
        if fx1 is not None and fx2 is not None:
            if abs(fx1) <= 1e-8:
                return x1
            if abs(fx2) <= 1e-8:
                return x2
            
            for _ in range(100):
                if not check_deadline():
                    break
                if abs(fx2) <= 1e-8:
                    return x2
                update_best(x2, fx2)
                if abs(fx1) <= 1e-8:
                    return x1
                update_best(x1, fx1)
                
                denom = fx2 - fx1
                if denom == 0 or not is_finite(denom):
                    break
                x_new = x2 - fx2 * (x2 - x1) / denom
                if not is_finite(x_new):
                    break
                if abs(x_new - x1) < 1e-14:
                    x_new = x1 + 1e-10
                if abs(x_new - x2) < 1e-14:
                    x_new = x2 + 1e-10
                
                x1, fx1 = x2, fx2
                x2 = x_new
                fx2 = safe_f(x2)
                if fx2 is None:
                    break
    
    # Strategy 3: Bisection with bracketing search
    if check_deadline() and best_x is not None and abs(best_fx) > 1e-8:
        a, b = None, None
        fa, fb = None, None
        
        scale = 1.0
        for _ in range(50):
            if not check_deadline():
                break
            a_try = x0 - scale
            b_try = x0 + scale
            
            fa_try = safe_f(a_try)
            fb_try = safe_f(b_try)
            
            if fa_try is not None and fb_try is not None:
                if fa_try * fb_try <= 0:
                    a, b, fa, fb = a_try, b_try, fa_try, fb_try
                    break
                if abs(fa_try) <= 1e-8:
                    return a_try
                if abs(fb_try) <= 1e-8:
                    return b_try
            
            scale *= 2
            if scale > 1e6:
                break
        
        if a is not None and b is not None:
            for _ in range(100):
                if not check_deadline():
                    break
                mid = (a + b) / 2.0
                fmid = safe_f(mid)
                if fmid is None:
                    break
                if abs(fmid) <= 1e-8:
                    return mid
                update_best(mid, fmid)
                
                if fa * fmid <= 0:
                    b = mid
                    fb = fmid
                else:
                    a = mid
                    fa = fmid
    
    # Return best found
    if best_x is not None:
        return best_x
    return x0
# EVOLVE-BLOCK-END
