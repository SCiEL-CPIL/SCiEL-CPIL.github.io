"""Scalar root finder for 3 fixed smooth polynomial equations."""
import math
import time
import numpy as np

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find a root of f starting near x0, returning x with |f(x)| <= 1e-8.
    Uses a combination of:
    1. Check x0 first
    2. Try to find a bracket via adaptive expansion
    3. Use Brent's method (scipy) for robust convergence
    4. Fallback to secant method if no bracket found
    5. Polish with Newton-like steps using finite-difference derivative
    """
    DEADLINE = 145.0  # Safety margin before 150s kill
    start_time = time.time()
    
    def check_deadline():
        return (time.time() - start_time) < DEADLINE
    
    def is_finite(x):
        return math.isfinite(x)
    
    call_count = 0
    
    def safe_f(x):
        nonlocal call_count
        if call_count >= 490:  # Leave buffer for safety
            return None
        try:
            val = f(x)
            call_count += 1
            if not is_finite(val):
                return None
            return float(val)
        except:
            call_count += 1
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
    
    # Strategy 1: Try to find a bracket, then use Brent's method
    if check_deadline():
        # Adaptive bracket expansion
        a, b = None, None
        fa, fb = None, None
        
        scale = 1e-3
        max_scale = 1e6
        
        # First try small brackets near x0
        for _ in range(20):
            if not check_deadline():
                break
            if call_count >= 480:
                break
            
            a_try = x0 - scale
            b_try = x0 + scale
            
            fa_try = safe_f(a_try)
            fb_try = safe_f(b_try)
            
            if fa_try is not None and fb_try is not None:
                if abs(fa_try) <= 1e-8:
                    return a_try
                if abs(fb_try) <= 1e-8:
                    return b_try
                if fa_try * fb_try < 0:
                    a, b, fa, fb = a_try, b_try, fa_try, fb_try
                    break
            
            scale *= 2.0
            if scale > max_scale:
                break
        
        # If bracket found, use Brent's method
        if a is not None and b is not None and check_deadline():
            try:
                from scipy.optimize import brentq
                x_root = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=50)
                if is_finite(x_root):
                    fx_root = safe_f(x_root)
                    if fx_root is not None:
                        update_best(x_root, fx_root)
                        if abs(fx_root) <= 1e-8:
                            return x_root
            except Exception:
                pass
        
        # If no bracket or Brent failed, try secant method
        if check_deadline() and (best_x is None or abs(best_fx) > 1e-8):
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
                    if call_count >= 480:
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
    
    # Strategy 2: If we have a good candidate, polish with Newton using finite-diff derivative
    if check_deadline() and best_x is not None and abs(best_fx) > 1e-8:
        x_polish = best_x
        for _ in range(20):
            if not check_deadline():
                break
            if call_count >= 480:
                break
            
            fx_p = safe_f(x_polish)
            if fx_p is None:
                break
            update_best(x_polish, fx_p)
            if abs(fx_p) <= 1e-8:
                return x_polish
            
            # Finite difference derivative
            h = 1e-8 * max(1.0, abs(x_polish))
            fx_plus = safe_f(x_polish + h)
            if fx_plus is None:
                break
            dfx = (fx_plus - fx_p) / h
            if abs(dfx) < 1e-15:
                break
            
            x_new = x_polish - fx_p / dfx
            if not is_finite(x_new):
                break
            if abs(x_new - x_polish) < 1e-14:
                break
            x_polish = x_new
    
    # Strategy 3: If still not converged, try a few more secant iterations from different starting points
    if check_deadline() and (best_x is None or abs(best_fx) > 1e-8):
        # Try different offsets
        for offset_factor in [1e-3, 1e-2, 0.1, 1.0, 10.0]:
            if not check_deadline():
                break
            if call_count >= 470:
                break
            
            x1 = x0
            x2 = x0 + offset_factor * max(1.0, abs(x0))
            fx1 = safe_f(x1)
            fx2 = safe_f(x2)
            
            if fx1 is None or fx2 is None:
                continue
            if abs(fx1) <= 1e-8:
                return x1
            if abs(fx2) <= 1e-8:
                return x2
            
            for _ in range(20):
                if not check_deadline():
                    break
                if call_count >= 480:
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
    
    # Return best found
    if best_x is not None:
        return best_x
    return x0
# EVOLVE-BLOCK-END
