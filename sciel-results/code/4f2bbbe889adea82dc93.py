import math
import time

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    max_calls = 480
    call_count = 0
    best_x = float(x0)
    best_f = float('inf')
    
    def time_ok():
        return (time.time() - start_time) < deadline and call_count < max_calls

    def f_call(x):
        nonlocal call_count, best_x, best_f
        call_count += 1
        fx = f(x)
        if math.isfinite(fx) and abs(fx) < abs(best_f):
            best_f = fx
            best_x = x
        return fx

    x = float(x0)
    if not math.isfinite(x):
        return best_x

    fx = f_call(x)
    
    if not math.isfinite(fx):
        return best_x

    if abs(fx) <= 1e-8:
        return x

    # Try to find a bracket
    a = x
    fa = fx
    h = 1.0
    b = None
    fb = None
    
    for _ in range(20):
        if not time_ok():
            break
        b = a + h
        fb = f_call(b)
        if not math.isfinite(fb):
            h *= 2
            continue
        if fa * fb < 0:
            break
        a = b
        fa = fb
        h *= 2
        
    if b is not None and fa * fb < 0:
        # Use Brent's method via scipy if available, otherwise manual Illinois/Secant-Bisection
        try:
            from scipy.optimize import brentq
            root = brentq(f_call, a, b, xtol=1e-14, rtol=1e-14, maxiter=100)
            if math.isfinite(root):
                return root
        except Exception:
            pass
        
        # Fallback if scipy fails or raises
        # Manual Secant-Bisection
        p = 0.5 * (a + b)
        fp = f_call(p)
        
        if abs(fp) <= 1e-8:
            return p
            
        x_prev = a
        fx_prev = fa
        x_curr = p
        fx_curr = fp
        
        if fx_prev * fx_curr < 0:
            b = x_curr
            fb = fx_curr
        else:
            a = x_curr
            fa = fx_curr
            
        for _ in range(100):
            if not time_ok():
                break
                
            if abs(fx_curr) <= 1e-8:
                return x_curr
            if abs(fx_prev) <= 1e-8:
                return x_prev
                
            if abs(b - a) < 1e-12 * max(1.0, abs(a), abs(b)):
                return 0.5 * (a + b)

            if fx_curr != fx_prev:
                x_secant = x_curr - fx_curr * (x_curr - x_prev) / (fx_curr - fx_prev)
            else:
                x_secant = 0.5 * (a + b)
                
            if a < x_secant < b:
                x_next = x_secant
            else:
                x_next = 0.5 * (a + b)
                
            f_next = f_call(x_next)
            
            if not math.isfinite(f_next):
                x_next = 0.5 * (a + b)
                f_next = f_call(x_next)
                
            if abs(f_next) <= 1e-8:
                return x_next
                
            x_prev = x_curr
            fx_prev = fx_curr
            x_curr = x_next
            fx_curr = f_next
            
            if fa * f_next < 0:
                b = x_next
                fb = f_next
            else:
                a = x_next
                fa = f_next
                
        return x_curr if abs(fx_curr) < abs(fx_prev) else x_prev

    else:
        # No bracket found. Fall back to Newton's Method with Finite Difference
        x = float(x0)
        fx = f_call(x)
        
        for _ in range(60):
            if not time_ok():
                break
            if abs(fx) <= 1e-8:
                return x
                
            h = 1e-6 * max(1.0, abs(x))
            fx_plus = f_call(x + h)
            d = (fx_plus - fx) / h
            
            if d == 0 or not math.isfinite(d):
                break
                
            x_new = x - fx / d
            
            if not math.isfinite(x_new):
                break
                
            if abs(x_new - x) > 10.0:
                x_new = x + 10.0 * math.copysign(1.0, x_new - x)
                
            x = x_new
            fx = f_call(x)
            
        return x
