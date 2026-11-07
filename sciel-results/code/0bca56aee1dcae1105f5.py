import numpy as np
from scipy.optimize import brentq

def find_root(f, x0):
    x0 = float(x0)
    tol = 1e-8
    
    # Initial evaluation
    try:
        f0 = f(x0)
        if not np.isfinite(f0):
            return x0
        if abs(f0) <= tol:
            return x0
    except:
        return x0

    # Phase 1: Secant Method
    # x1 = x0 + eps
    eps = 1e-6 * max(1.0, abs(x0))
    x1 = x0 + eps
    try:
        f1 = f(x1)
        if not np.isfinite(f1):
            # Fallback to bracketing
            pass
        elif abs(f1) <= tol:
            return x1
        else:
            # Iterate Secant
            x_prev = x0
            x_curr = x1
            f_prev = f0
            f_curr = f1
            
            for i in range(10):
                if abs(f_curr - f_prev) < 1e-15:
                    break
                x_new = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
                
                if not np.isfinite(x_new):
                    break
                
                # Safeguard: if step is too large, break to bracketing
                if abs(x_new - x_curr) > 10.0 * max(1.0, abs(x_curr)):
                    break
                
                f_new = f(x_new)
                if not np.isfinite(f_new):
                    break
                    
                if abs(f_new) <= tol:
                    return x_new
                    
                x_prev = x_curr
                f_prev = f_curr
                x_curr = x_new
                f_curr = f_new
            else:
                # Converged or finished iterations
                if abs(f_curr) <= tol:
                    return x_curr
                # Check if we have a bracket from the last two points
                if f_prev * f_curr < 0:
                    try:
                        root = brentq(f, x_prev, x_curr, xtol=1e-12, rtol=1e-12, maxiter=100)
                        return root
                    except:
                        pass
    except:
        pass

    # Phase 2: Robust Bracketing from x0
    # Exponential expansion
    step = 1e-4 * max(1.0, abs(x0))
    a = None
    b = None
    fa = None
    fb = None
    
    for i in range(50):
        x_left = x0 - step
        x_right = x0 + step
        
        try:
            f_left = f(x_left)
            f_right = f(x_right)
        except:
            break
            
        if not np.isfinite(f_left) or not np.isfinite(f_right):
            step *= 2.0
            continue
            
        if abs(f_left) <= tol: return x_left
        if abs(f_right) <= tol: return x_right
        
        if f_left * f0 < 0:
            a, b = x_left, x0
            fa, fb = f_left, f0
            break
        elif f_right * f0 < 0:
            a, b = x0, x_right
            fa, fb = f0, f_right
            break
            
        step *= 2.0
        if step > 1e10:
            break

    if a is not None and b is not None:
        try:
            root = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
            return root
        except:
            pass

    # Phase 3: Fallback
    best_x = x0
    best_val = abs(f0)
    for scale in [1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
        for sign in [1, -1]:
            x_try = x0 + sign * scale
            try:
                fx_try = f(x_try)
                if np.isfinite(fx_try) and abs(fx_try) < best_val:
                    best_val = abs(fx_try)
                    best_x = x_try
            except:
                pass
    return best_x
