import math
import time
import numpy as np
from scipy.optimize import brentq

def find_root(f, x0):
    x0 = float(x0)
    start_time = time.time()
    deadline = start_time + 118.0
    
    call_count = 0
    max_calls = 500
    best_x = x0
    best_f = float('inf')
    
    def f_counted(x):
        nonlocal call_count, best_x, best_f
        if call_count >= max_calls:
            raise Exception("Max calls exceeded")
        call_count += 1
        val = f(x)
        if abs(val) < abs(best_f):
            best_f = val
            best_x = x
        return val
    
    def check_root(x, fx):
        nonlocal best_x, best_f
        if abs(fx) < abs(best_f):
            best_f = fx
            best_x = x
        if abs(fx) <= 1e-8:
            return True
        return False

    # 1. Check x0
    try:
        fx0 = f_counted(x0)
    except Exception:
        return x0
        
    if check_root(x0, fx0):
        return x0

    # 2. Try to determine polynomial degree and coefficients
    # We will sample points x0, x0+1, x0+2, ...
    # and use divided differences.
    
    points = [x0]
    values = [fx0]
    
    # We'll try to fit a polynomial of increasing degree
    # If the divided differences stabilize, we assume we have the degree.
    
    max_degree_to_check = 10 # Polynomials are likely low degree
    
    for d in range(1, max_degree_to_check + 1):
        if time.time() > deadline or call_count >= max_calls - 5:
            break
            
        # Need one more point to determine degree d
        x_new = x0 + d
        if not math.isfinite(x_new):
            break
            
        try:
            f_new = f_counted(x_new)
        except Exception:
            break
            
        if check_root(x_new, f_new):
            return x_new
            
        points.append(x_new)
        values.append(f_new)
        
        # Calculate divided differences for the first d+1 points
        # If the d-th divided difference is small relative to previous,
        # and (d+1)-th would be even smaller (or we just assume d is enough),
        # we can try to fit.
        
        # Actually, let's just fit a polynomial of degree d using the d+1 points
        # and check if the residual on the next point (if we had one) would be small.
        # But we don't have the next point yet.
        
        # Alternative: Check if the divided differences of order d are roughly constant.
        # For a polynomial of degree d, the d-th divided difference is constant.
        
        # Let's compute the divided differences table for the current points.
        # If the last row (highest order) is "small" compared to the scale,
        # it might indicate the degree is lower, or we are close to the root of the interpolant.
        
        # Simpler: Just fit the polynomial of degree d and find its roots.
        # Then verify.
        
        try:
            coeffs = np.polyfit(points[:d+1], values[:d+1], d)
            roots = np.roots(coeffs)
            real_roots = [r.real for r in roots if abs(r.imag) < 1e-10 and math.isfinite(r.real)]
            
            if not real_roots:
                continue
                
            # Sort roots by distance to x0
            real_roots.sort(key=lambda r: abs(r - x0))
            
            found = False
            for root in real_roots:
                if time.time() > deadline or call_count >= max_calls - 2:
                    break
                    
                try:
                    fr = f_counted(root)
                except Exception:
                    continue
                    
                if check_root(root, fr):
                    return root
                    
                # If it's very close, we might be able to polish
                if abs(fr) < 1e-6:
                    # Use Newton's method for polishing
                    # Need derivative. For polynomial, we can compute it from coeffs.
                    # Or just use secant with a tiny step.
                    h = 1e-8
                    x1 = root - h
                    x2 = root + h
                    if call_count < max_calls - 2:
                        try:
                            f1 = f_counted(x1)
                            if check_root(x1, f1):
                                return x1
                            f2 = f_counted(x2)
                            if check_root(x2, f2):
                                return x2
                            
                            # Secant step
                            denom = f2 - f1
                            if abs(denom) > 1e-15:
                                x_new = x2 - f2 * (x2 - x1) / denom
                                if math.isfinite(x_new) and call_count < max_calls:
                                    f_new = f_counted(x_new)
                                    if check_root(x_new, f_new):
                                        return x_new
                        except Exception:
                            pass
                            
        except Exception:
            continue

    # 3. Fallback: Brent's Method with Bracketing
    # We need a bracket [a, b] where f(a)*f(b) < 0.
    # We have points and values from the previous loop.
    # Let's check if any pair of our sampled points brackets a root.
    
    found_bracket = False
    bracket_a = None
    bracket_b = None
    bracket_fa = None
    bracket_fb = None
    
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            if points[i] == points[j]:
                continue
            if values[i] * values[j] < 0:
                found_bracket = True
                bracket_a = points[i]
                bracket_b = points[j]
                bracket_fa = values[i]
                bracket_fb = values[j]
                break
        if found_bracket:
            break
            
    if not found_bracket:
        # Try expanding from x0
        fa = fx0
        step = 1.0
        for direction in [1, -1]:
            if time.time() > deadline or call_count >= max_calls - 10:
                break
                
            for i in range(20):
                b = x0 + direction * step
                if not math.isfinite(b):
                    break
                try:
                    fb = f_counted(b)
                except Exception:
                    break
                    
                if check_root(b, fb):
                    return b
                    
                if fa * fb < 0:
                    found_bracket = True
                    bracket_a = x0
                    bracket_b = b
                    bracket_fa = fa
                    bracket_fb = fb
                    break
                    
                step *= 2.0
                
            if found_bracket:
                break

    if found_bracket:
        try:
            root = brentq(f_counted, bracket_a, bracket_b, xtol=1e-13, rtol=1e-13, maxiter=50)
            if math.isfinite(root):
                fr = f_counted(root)
                if abs(fr) <= 1e-8:
                    return root
                # Polish if close
                if abs(fr) < 1e-6 and call_count < max_calls - 2:
                    h = 1e-8
                    x1 = root - h
                    x2 = root + h
                    try:
                        f1 = f_counted(x1)
                        if check_root(x1, f1):
                            return x1
                        f2 = f_counted(x2)
                        if check_root(x2, f2):
                            return x2
                        if abs(f2 - f1) > 1e-15:
                            x_new = x2 - f2 * (x2 - x1) / (f2 - f1)
                            if math.isfinite(x_new) and call_count < max_calls:
                                f_new = f_counted(x_new)
                                if abs(f_new) <= 1e-8:
                                    return x_new
                    except Exception:
                        pass
        except Exception:
            pass

    # 4. Fallback: Secant Method
    # Use the two points with smallest |f(x)|
    if len(points) >= 2:
        # Sort points by |f(x)|
        sorted_pts = sorted(zip(values, points), key=lambda x: abs(x[0]))
        x1 = sorted_pts[0][1]
        x2 = sorted_pts[1][1]
        f1 = sorted_pts[0][0]
        f2 = sorted_pts[1][0]
    else:
        x1 = x0
        x2 = x0 + 1.0
        f1 = fx0
        try:
            f2 = f_counted(x2)
        except Exception:
            f2 = fx0
            
    for _ in range(50):
        if time.time() > deadline or call_count > max_calls - 1:
            break
            
        if abs(f1) <= 1e-8:
            return x1
        if abs(f2) <= 1e-8:
            return x2
            
        denom = f2 - f1
        if abs(denom) < 1e-15:
            break
            
        x_new = x2 - f2 * (x2 - x1) / denom
        
        if not math.isfinite(x_new):
            break
            
        # Clamp
        if abs(x_new - x2) > 100 * max(1.0, abs(x2)):
            x_new = x2 + 100 * max(1.0, abs(x2)) * (1 if x_new > x2 else -1)
            
        x1, x2 = x2, x_new
        f1, f2 = f2, 0.0 # Placeholder
        
        if call_count < max_calls:
            try:
                f2 = f_counted(x2)
            except Exception:
                break
                
        if abs(f2) <= 1e-8:
            return x2

    return best_x
