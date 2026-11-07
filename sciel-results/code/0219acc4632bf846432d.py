import time
import numpy as np
from scipy.optimize import brentq

def find_root(f, x0):
    start_time = time.time()
    TIME_LIMIT = 140.0  # Safety margin
    x0 = float(x0)
    
    call_count = 0
    MAX_CALLS = 480  # Leave buffer for safety
    
    def time_ok():
        return (time.time() - start_time) < TIME_LIMIT

    def safe_f(x):
        nonlocal call_count
        if not time_ok():
            raise TimeoutError("Time limit exceeded")
        if call_count >= MAX_CALLS:
            raise Exception("Call limit exceeded")
        call_count += 1
        val = f(float(x))
        if not np.isfinite(val):
            return float('inf')
        return float(val)

    best_x = x0
    best_f_val = float('inf')
    
    try:
        f0 = safe_f(x0)
        if abs(f0) <= 1e-8:
            return x0
        best_f_val = abs(f0)
        
        # Phase 1: Newton's Method with Damping
        x = x0
        fx = f0
        for _ in range(15):
            if not time_ok(): break
            if call_count >= MAX_CALLS: break
            
            # Finite difference derivative
            h = 1e-7 * max(1.0, abs(x))
            try:
                fxm = safe_f(x - h)
                fxp = safe_f(x + h)
            except:
                break
                
            df = (fxp - fxm) / (2 * h)
            
            if abs(df) < 1e-15:
                break
                
            x_new = x - fx / df
            
            # Check for divergence or step too large
            if abs(x_new - x) > 10 * max(1.0, abs(x)):
                break
                
            x = x_new
            try:
                fx_new = safe_f(x)
            except:
                break
                
            if abs(fx_new) < best_f_val:
                best_f_val = abs(fx_new)
                best_x = x
                
            if abs(fx_new) <= 1e-8:
                return x
                
            fx = fx_new
            
            # Convergence check
            if abs(fx_new) < 1e-12:
                return x

        # Phase 2: Bracketing Scan
        # Try scales from small to large
        scales = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
        
        for s in scales:
            if not time_ok(): break
            if call_count >= MAX_CALLS - 20: break # Leave room for Brent
            
            a = x0 - s
            b = x0 + s
            
            try:
                fa = safe_f(a)
            except:
                continue
                
            # Check bracket [a, x0]
            if np.isfinite(fa) and np.isfinite(f0) and fa * f0 < 0:
                try:
                    root = brentq(safe_f, a, x0, xtol=1e-12, maxiter=50)
                    if np.isfinite(root) and abs(safe_f(root)) <= 1e-8:
                        return root
                except:
                    pass
                    
            try:
                fb = safe_f(b)
            except:
                continue
                
            # Check bracket [x0, b]
            if np.isfinite(fb) and np.isfinite(f0) and f0 * fb < 0:
                try:
                    root = brentq(safe_f, x0, b, xtol=1e-12, maxiter=50)
                    if np.isfinite(root) and abs(safe_f(root)) <= 1e-8:
                        return root
                except:
                    pass

            # If we have fa and fb, check [a, b] as well? 
            # If fa and fb have opposite signs, there is an odd number of roots in [a, b].
            # Brent can handle this.
            if np.isfinite(fa) and np.isfinite(fb) and fa * fb < 0:
                 try:
                    root = brentq(safe_f, a, b, xtol=1e-12, maxiter=50)
                    if np.isfinite(root) and abs(safe_f(root)) <= 1e-8:
                        return root
                 except:
                    pass

        # Phase 3: If no bracket found, try a denser scan in a local area
        # This might be unnecessary if scales cover it, but let's add a small grid scan
        # if we haven't used too many calls.
        if call_count < MAX_CALLS - 50:
            h = 1e-2
            x_left = x0 - h
            x_right = x0 + h
            # We already checked x0-s and x0+s for s=1e-2 in the loop above?
            # Yes, scales included 1e-2.
            # So this phase is redundant if the loop ran.
            
        return best_x

    except Exception as e:
        # If anything goes wrong, return the best x we found
        # Note: Returning a non-root is a failure, but crashing is also a failure.
        # It's better to return a float.
        return best_x
