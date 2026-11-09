import math
import time

def find_root(f, x0):
    """
    Finds a root x for f(x) such that |f(x)| <= 1e-8.
    Uses a combination of bracketing search and secant/bisection refinement.
    """
    tol = 1e-8
    max_calls = 480  # Safety margin below 500
    call_count = 0
    cache = {}
    
    # Deadline guard: 110 seconds
    start_time = time.time()
    
    def safe_f(x):
        nonlocal call_count
        if call_count >= max_calls:
            raise RuntimeError("Call limit exceeded")
        if time.time() - start_time > 110:
            raise RuntimeError("Time limit exceeded")
            
        if x in cache:
            return cache[x]
        
        fx = f(x)
        call_count += 1
        cache[x] = fx
        
        if not math.isfinite(fx):
            raise ValueError("Non-finite value")
            
        return fx

    x = float(x0)
    
    # 1. Check initial point
    try:
        fx = safe_f(x)
    except Exception:
        return x
        
    if abs(fx) <= tol:
        return x

    best_x = x
    best_f = fx

    # 2. Try Steffensen's Method from x0 (quadratic convergence, no bracket needed)
    # x_{n+1} = x_n - f(x_n)^2 / (f(x_n + f(x_n)) - f(x_n))
    # This is equivalent to Newton's method but uses finite differences for the derivative.
    # Limit to ~30 calls to preserve budget for fallback methods.
    steff_calls_used = 0
    x_ste = x
    f_ste = fx
    for i in range(15):
        if call_count > max_calls - 40:
            break
        # f(x + f(x))
        x_probe = x_ste + f_ste
        if not math.isfinite(x_probe):
            break
        try:
            f_probe = safe_f(x_probe)
        except Exception:
            break
        steff_calls_used += 1
        denom = f_probe - f_ste
        if denom == 0:
            break
        x_new = x_ste - f_ste * f_ste / denom
        if not math.isfinite(x_new):
            break
        try:
            f_new = safe_f(x_new)
        except Exception:
            break
        steff_calls_used += 1
        if abs(f_new) <= tol:
            return x_new
        if abs(f_new) < abs(best_f):
            best_f = f_new
            best_x = x_new
        # Check for convergence
        if abs(x_new - x_ste) < 1e-12 * max(1.0, abs(x_new)):
            if abs(f_new) <= 1e-7:
                break
        x_ste = x_new
        f_ste = f_new

    # 3. Find a bracket [a, b] with f(a)*f(b) < 0
    # We search outwards from x0
    a = x
    fa = fx
    b = None
    fb = None
    
    # Steps to try for bracketing
    steps = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0, 1e4, 1e5, 1e6]
    
    directions = [1, -1]
    
    found_bracket = False
    
    for step in steps:
        for d in directions:
            x_new = x + d * step
            try:
                fx_new = safe_f(x_new)
            except Exception:
                continue
            
            if abs(fx_new) <= tol:
                return x_new
            
            if abs(fx_new) < abs(best_f):
                best_f = fx_new
                best_x = x_new
                
            if fa * fx_new < 0:
                a, b = x, x_new
                fa, fb = fx, fx_new
                found_bracket = True
                break
            
        if found_bracket:
            break
        
        if call_count > 120:
            break

    if not found_bracket:
        return best_x

    # 4. Refine the root using Secant Method, falling back to Bisection
    lo = min(a, b)
    hi = max(a, b)
    
    x_prev = a
    f_prev = fa
    x_curr = b
    f_curr = fb
    
    # Track best bracket during secant for fallback
    best_lo, best_hi = lo, hi
    best_flo, best_fhi = f_prev, f_curr
    if abs(f_prev) < abs(f_curr):
        best_lo, best_hi = a, b
    else:
        best_lo, best_hi = b, a
        best_flo, best_fhi = fb, fa

    for i in range(30):
        if call_count > max_calls - 10:
            break
            
        denom = f_curr - f_prev
        if denom == 0:
            break
            
        x_next = x_curr - f_curr * (x_curr - x_prev) / denom
        
        if not math.isfinite(x_next):
            break
            
        # Allow secant to stay within a reasonable range
        bracket_width = hi - lo
        if x_next < lo - 2 * bracket_width or x_next > hi + 2 * bracket_width:
            break
            
        try:
            f_next = safe_f(x_next)
        except Exception:
            break
            
        if abs(f_next) <= tol:
            return x_next
            
        if abs(f_next) < abs(best_f):
            best_f = f_next
            best_x = x_next
        
        # Update best bracket if we have a sign change
        if f_prev * f_next < 0:
            if abs(x_next - x_prev) < abs(best_hi - best_lo):
                best_lo, best_hi = min(x_prev, x_next), max(x_prev, x_next)
                best_flo, best_fhi = (f_prev, f_next) if x_prev < x_next else (f_next, f_prev)
        elif f_curr * f_next < 0:
            if abs(x_next - x_curr) < abs(best_hi - best_lo):
                best_lo, best_hi = min(x_curr, x_next), max(x_curr, x_next)
                best_flo, best_fhi = (f_curr, f_next) if x_curr < x_next else (f_next, f_curr)
        
        x_prev = x_curr
        f_prev = f_curr
        x_curr = x_next
        f_curr = f_next
        
        if abs(x_next - x_prev) < 1e-12 * max(1.0, abs(x_next)):
            if abs(f_next) <= 1e-7:
                break
                 
    # 5. Bisection on the best bracket found
    a_bis = best_lo
    b_bis = best_hi
    fa_bis = best_flo
    fb_bis = best_fhi
    
    if fa_bis * fb_bis > 0:
        return best_x
        
    for i in range(50):
        if call_count > max_calls - 1:
            break
            
        mid = (a_bis + b_bis) / 2.0
        try:
            f_mid = safe_f(mid)
        except Exception:
            break
        
        if abs(f_mid) <= tol:
            return mid
            
        if fa_bis * f_mid < 0:
            b_bis = mid
            fb_bis = f_mid
        else:
            a_bis = mid
            fa_bis = f_mid
            
        if abs(b_bis - a_bis) < 1e-13:
            break
            
    # Final polish: return midpoint or best_x
    mid_final = (a_bis + b_bis) / 2.0
    if abs(f_mid) < abs(best_f):
        return mid_final
    return best_x
