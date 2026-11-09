import math
import time
import scipy.optimize

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    max_calls = 480
    call_count = 0
    
    def time_ok():
        return (time.time() - start_time) < deadline and call_count < max_calls

    # Helper to call f and track count
    def f_call(x):
        nonlocal call_count
        call_count += 1
        return f(x)

    x = float(x0)
    fx = f_call(x)
    
    if not math.isfinite(fx):
        return x

    if abs(fx) <= 1e-8:
        return x

    # --- Phase 1: Find a Bracket ---
    a = x
    fa = fx
    
    # Try right expansion first (most common case for well-behaved starting points)
    h = 1.0
    b = None
    fb = None
    
    for i in range(30):
        if not time_ok():
            break
        
        b_candidate = a + h
        fb_candidate = f_call(b_candidate)
        
        if not math.isfinite(fb_candidate):
            h *= 2
            continue
            
        if fa * fb_candidate < 0:
            b = b_candidate
            fb = fb_candidate
            break
        
        h *= 2.0
    
    # If right expansion failed, try left
    if b is None:
        h = 1.0
        for i in range(30):
            if not time_ok():
                break
            
            b_candidate = a - h
            fb_candidate = f_call(b_candidate)
            
            if not math.isfinite(fb_candidate):
                h *= 2
                continue
                
            if fa * fb_candidate < 0:
                b = b_candidate
                fb = fb_candidate
                break
            
            h *= 2.0
    
    if b is None:
        # Extreme fallback: Secant without bracket
        x1 = x0 + 1e-6
        fx1 = f_call(x1)
        for _ in range(20):
            if not time_ok(): break
            if abs(fx) <= 1e-8: return x0
            if abs(fx1) <= 1e-8: return x1
            if fx1 == fx: break
            x_new = x0 - fx * (x0 - x1) / (fx - fx1)
            if not math.isfinite(x_new): break
            f_new = f_call(x_new)
            if not math.isfinite(f_new): break
            x0, fx = x1, fx1
            x1, fx1 = x_new, f_new
        return x0 if abs(fx) < abs(fx1) else x1

    # --- Phase 2: Refine using SciPy Brent's Method ---
    if a > b:
        a, b = b, a
        fa, fb = fb, fa
        
    try:
        root = scipy.optimize.brentq(f_call, a, b, xtol=1e-14, rtol=1e-14, maxiter=100)
        if math.isfinite(root):
            f_root = f_call(root)
            if math.isfinite(f_root) and abs(f_root) <= 1e-8:
                return root
            # If brentq returned a point but |f| > 1e-8, try a few Newton steps
            x_new = root
            fx_new = f_root
            for _ in range(5):
                if not time_ok():
                    break
                # Estimate derivative using central difference
                dx = max(1e-8, abs(x_new) * 1e-8)
                fp = f_call(x_new + dx)
                fm = f_call(x_new - dx)
                if not (math.isfinite(fp) and math.isfinite(fm)):
                    break
                df = (fp - fm) / (2 * dx)
                if abs(df) < 1e-30:
                    break
                x_new -= fx_new / df
                if not math.isfinite(x_new):
                    break
                fx_new = f_call(x_new)
                if not math.isfinite(fx_new):
                    break
                if abs(fx_new) <= 1e-8:
                    return x_new
            return x_new if abs(fx_new) < abs(f_root) else root
        else:
            return a
    except Exception:
        # Fallback to manual Illinois/Secant-Bisection
        x_prev = a
        fx_prev = fa
        x_curr = b
        fx_curr = fb
        
        for _ in range(100):
            if not time_ok():
                break
            if abs(fx_curr) <= 1e-8:
                return x_curr
            if abs(fx_prev) <= 1e-8:
                return x_prev
                
            if abs(x_curr - x_prev) < 1e-12 * max(1.0, abs(x_curr), abs(x_prev)):
                return 0.5 * (x_curr + x_prev)
                
            if fx_curr != fx_prev:
                x_secant = x_curr - fx_curr * (x_curr - x_prev) / (fx_curr - fx_prev)
            else:
                x_secant = 0.5 * (x_curr + x_prev)
                
            if min(x_prev, x_curr) < x_secant < max(x_prev, x_curr):
                x_next = x_secant
            else:
                x_next = 0.5 * (x_curr + x_prev)
                
            f_next = f_call(x_next)
            
            if not math.isfinite(f_next):
                 x_next = 0.5 * (x_curr + x_prev)
                 f_next = f_call(x_next)
                 
            if abs(f_next) <= 1e-8:
                return x_next
                 
            if fx_prev * f_next < 0:
                x_curr = x_next
                fx_curr = f_next
            else:
                x_prev = x_next
                fx_prev = f_next
                
        return x_curr if abs(fx_curr) < abs(fx_prev) else x_prev
