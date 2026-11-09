"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    def time_ok():
        return (time.time() - start_time) < deadline
    
    x = float(x0)
    if not math.isfinite(x):
        x = 0.0
    
    # Evaluate initial point
    try:
        fx = f(x)
    except:
        fx = float('inf')
    
    if not math.isfinite(fx):
        # Try some other points to get a finite value
        for dx in [1.0, -1.0, 0.1, -0.1, 0.01, -0.01, 0.5, -0.5]:
            x_try = x + dx
            try:
                fx_try = f(x_try)
                if math.isfinite(fx_try):
                    x, fx = x_try, fx_try
                    break
            except:
                continue
    
    if abs(fx) <= 1e-12:
        return x
    
    # Try to find a bracket [a, b] where f(a) and f(b) have opposite signs
    a, b = None, None
    fa, fb = None, None
    
    # First, try expanding from x0
    step = 1.0
    for _ in range(100):
        if not time_ok():
            break
        # Try x0 + step
        x_plus = x + step
        try:
            f_plus = f(x_plus)
        except:
            f_plus = float('inf')
        
        if math.isfinite(f_plus):
            if fa is None or abs(f_plus) < abs(fa):
                a, fa = x_plus, f_plus
            if fa is not None and fa * f_plus < 0:
                b, fb = x_plus, f_plus
                break
        
        # Try x0 - step
        x_minus = x - step
        try:
            f_minus = f(x_minus)
        except:
            f_minus = float('inf')
        
        if math.isfinite(f_minus):
            if fa is None or abs(f_minus) < abs(fa):
                a, fa = x_minus, f_minus
            if fa is not None and fa * f_minus < 0:
                b, fb = x_minus, f_minus
                break
        
        step *= 2.0
    
    # If we have a bracket, use Illinois method
    if a is not None and b is not None and fa * fb < 0:
        # Ensure a < b
        if a > b:
            a, b = b, a
            fa, fb = fb, fa
        
        # Illinois method: bisection with weighted update
        for _ in range(200):
            if not time_ok():
                break
            mid = (a + b) / 2.0
            try:
                f_mid = f(mid)
            except:
                f_mid = float('inf')
            
            if not math.isfinite(f_mid):
                break
            
            if abs(f_mid) <= 1e-12:
                return mid
            
            if fa * f_mid <= 0:
                b = mid
                fb = f_mid
            else:
                # Illinois: replace the endpoint with larger |f|
                if abs(fa) > abs(fb):
                    a = (a + b) / 2.0
                    try:
                        fa = f(a)
                    except:
                        fa = float('inf')
                    if not math.isfinite(fa):
                        break
                else:
                    b = (a + b) / 2.0
                    try:
                        fb = f(b)
                    except:
                        fb = float('inf')
                    if not math.isfinite(fb):
                        break
            
            # Check if interval is small enough
            if abs(b - a) < 1e-14 * max(1.0, abs(a), abs(b)):
                # Return the point with smaller |f|
                if abs(fa) <= abs(fb):
                    return a
                else:
                    return b
        
        # Return the point with smaller |f|
        if abs(fa) <= abs(fb):
            return a
        else:
            return b
    
    # If no bracket found, try Newton's method from x0
    x_newton = float(x0)
    if not math.isfinite(x_newton):
        x_newton = 0.0
    
    for _ in range(60):
        if not time_ok():
            break
        try:
            fx_n = f(x_newton)
        except:
            break
        
        if not math.isfinite(fx_n):
            break
        
        if abs(fx_n) <= 1e-12:
            return x_newton
        
        h = 1e-6 * max(1.0, abs(x_newton))
        try:
            fx_n_h = f(x_newton + h)
        except:
            break
        
        if not math.isfinite(fx_n_h):
            break
        
        d = (fx_n_h - fx_n) / h
        if d == 0 or not math.isfinite(d):
            break
        
        x_new = x_newton - fx_n / d
        if not math.isfinite(x_new):
            break
        
        # Damping if step is too large
        if abs(x_new - x_newton) > 10.0 * max(1.0, abs(x_newton)):
            x_new = x_newton + 10.0 * max(1.0, abs(x_newton)) * (1 if x_new > x_newton else -1)
        
        x_newton = x_new
    
    # Return the best point found
    best_x = x
    best_fx = fx
    if math.isfinite(fx_n):
        if abs(fx_n) < abs(best_fx):
            best_x = x_newton
    
    return best_x
# EVOLVE-BLOCK-END
