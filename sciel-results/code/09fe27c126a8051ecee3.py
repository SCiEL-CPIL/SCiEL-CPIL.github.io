"""Scalar root finder for 3 fixed smooth polynomial equations."""
import math
import time
import numpy as np

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find a root of f starting near x0, returning x with |f(x)| <= 1e-8.
    Uses polynomial reconstruction as primary strategy with fallbacks.
    """
    DEADLINE = 145.0
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

    x0 = float(x0)
    fx0 = safe_f(x0)
    if fx0 is not None:
        update_best(x0, fx0)
        if abs(fx0) <= 1e-8:
            return x0

    # Strategy 1: Polynomial reconstruction - evaluate at several points, fit polynomial, find roots
    if check_deadline() and best_x is not None and abs(best_fx) > 1e-8:
        scale = max(1.0, abs(x0))
        
        point_sets = [
            [x0 - 2*scale, x0 - scale, x0, x0 + scale, x0 + 2*scale],
            [x0 - 4*scale, x0 - 2*scale, x0, x0 + 2*scale, x0 + 4*scale],
            [x0 - 8*scale, x0 - 4*scale, x0, x0 + 4*scale, x0 + 8*scale],
            [x0 - 3*scale, x0 - scale, x0 + scale, x0 + 2*scale, x0 + 3*scale],
        ]
        
        for pts in point_sets:
            if not check_deadline():
                break
            
            vals = []
            valid = True
            for p in pts:
                v = safe_f(p)
                if v is None:
                    valid = False
                    break
                vals.append(v)
            
            if not valid:
                continue
            
            for deg in [2, 3, 4]:
                if deg + 1 > len(pts):
                    continue
                fit_pts = pts[:deg+1]
                fit_vals = vals[:deg+1]
                
                try:
                    coeffs = np.polyfit(fit_pts, fit_vals, deg)
                    roots = np.roots(coeffs)
                    
                    real_roots = [r.real for r in roots if abs(r.imag) < 1e-10]
                    
                    for r in real_roots:
                        if not is_finite(r):
                            continue
                        fr = safe_f(r)
                        if fr is None:
                            continue
                        if abs(fr) <= 1e-8:
                            return r
                        update_best(r, fr)
                        if abs(fr) <= 1e-8:
                            return r
                except:
                    continue
        
        if best_x is not None and abs(best_fx) > 1e-8 and check_deadline():
            x1 = best_x
            fx1 = best_fx
            x2 = x1 + 0.1 * max(1.0, abs(x1))
            fx2 = safe_f(x2)
            
            if fx2 is not None:
                update_best(x2, fx2)
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
                    if abs(x_new - x1) < 1e-12:
                        x_new = x1 + 1e-6
                    if abs(x_new - x2) < 1e-12:
                        x_new = x2 + 1e-6
                    
                    x1, fx1 = x2, fx2
                    x2 = x_new
                    fx2 = safe_f(x2)
                    if fx2 is None:
                        break

    # Strategy 2: Secant method from x0
    if check_deadline() and best_x is not None and abs(best_fx) > 1e-8:
        x1 = x0
        fx1 = fx0
        x2 = x0 + 0.1 * max(1.0, abs(x0))
        fx2 = safe_f(x2)
        
        if fx2 is not None:
            update_best(x2, fx2)
            if abs(fx2) <= 1e-8:
                return x2
            
            for _ in range(150):
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
                if abs(x_new - x1) < 1e-12:
                    x_new = x1 + 1e-6
                if abs(x_new - x2) < 1e-12:
                    x_new = x2 + 1e-6
                
                x1, fx1 = x2, fx2
                x2 = x_new
                fx2 = safe_f(x2)
                if fx2 is None:
                    break

    # Strategy 3: Newton's method with finite-difference derivative
    if check_deadline() and best_x is not None and abs(best_fx) > 1e-8:
        x = x0
        for _ in range(100):
            if not check_deadline():
                break
            fx = safe_f(x)
            if fx is None:
                break
            if abs(fx) <= 1e-8:
                return x
            update_best(x, fx)
            
            h = 1e-7 * max(1.0, abs(x))
            fx_plus = safe_f(x + h)
            if fx_plus is None:
                break
            d = (fx_plus - fx) / h
            if d == 0 or not is_finite(d):
                break
            x_new = x - fx / d
            if not is_finite(x_new):
                break
            if abs(x_new - x) > 10 * max(1.0, abs(x)):
                x_new = x + 0.5 * (x_new - x)
            x = x_new

    # Strategy 4: Bisection with bracketing search
    if check_deadline() and best_x is not None and abs(best_fx) > 1e-8:
        a, b = None, None
        fa, fb = None, None
        
        scale = 0.1 * max(1.0, abs(x0))
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
            if scale > 1e8:
                break
        
        if a is not None and b is not None:
            for _ in range(150):
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

    # Strategy 5: Global search with multiple starting points
    if check_deadline() and best_x is not None and abs(best_fx) > 1e-8:
        offsets = [0.01, -0.01, 0.1, -0.1, 1.0, -1.0, 10.0, -10.0,
                   0.001, -0.001, 100.0, -100.0, 0.5, -0.5, 5.0, -5.0]
        for offset in offsets:
            if not check_deadline():
                break
            x_start = x0 + offset * max(1.0, abs(x0))
            fx_start = safe_f(x_start)
            if fx_start is None:
                continue
            if abs(fx_start) <= 1e-8:
                return x_start
            update_best(x_start, fx_start)
            
            x1 = x_start
            fx1 = fx_start
            x2 = x_start + 0.1 * max(1.0, abs(x_start))
            fx2 = safe_f(x2)
            if fx2 is None:
                continue
            update_best(x2, fx2)
            if abs(fx2) <= 1e-8:
                return x2
            
            for _ in range(60):
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
                if abs(x_new - x1) < 1e-12:
                    x_new = x1 + 1e-6
                if abs(x_new - x2) < 1e-12:
                    x_new = x2 + 1e-6
                
                x1, fx1 = x2, fx2
                x2 = x_new
                fx2 = safe_f(x2)
                if fx2 is None:
                    break

    if best_x is not None:
        return best_x
    return x0
# EVOLVE-BLOCK-END
