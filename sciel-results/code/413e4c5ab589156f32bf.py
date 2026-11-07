import numpy as np
from scipy.optimize import brentq

def find_root(f, x0):
    x = float(x0)
    
    # Phase 1: Newton's Method (Fast, but can fail)
    # Use central difference for derivative for better accuracy, but costs 2 calls.
    # Let's use forward difference (1 call) to save calls, as we have a fallback.
    for _ in range(10):
        fx = f(x)
        if not np.isfinite(fx):
            break
        if abs(fx) <= 1e-8:
            return x
        
        h = 1e-5 * max(1.0, abs(x))
        fx_plus = f(x + h)
        if not np.isfinite(fx_plus):
            break
            
        d = (fx_plus - fx) / h
        if abs(d) < 1e-15:
            break
            
        x_new = x - fx / d
        
        if not np.isfinite(x_new):
            break
            
        # Safeguard: if the new step is too large, reduce it
        # Simple safeguard: if |x_new - x| > 10 * |x|, dampen
        if abs(x_new - x) > 10.0 * max(1.0, abs(x)):
            x_new = x + 0.5 * (x_new - x)
            
        x = x_new

    # Phase 2: Robust Bracketing + Brent's Method
    # We try to find a bracket around the current x (which might be improved by Newton)
    # or around x0 if Newton moved us far away.
    # Let's try bracketing around x first. If that fails, try x0.
    
    def try_brent(center):
        h = 1e-4
        for _ in range(50):
            a = center - h
            b = center + h
            try:
                fa = f(a)
                fb = f(b)
            except:
                return None
            
            if not (np.isfinite(fa) and np.isfinite(fb)):
                h *= 2.0
                continue
                
            if fa == 0:
                return a
            if fb == 0:
                return b
                
            if fa * fb < 0:
                # Found bracket
                try:
                    root = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
                    if abs(f(root)) <= 1e-8:
                        return root
                    else:
                        return root
                except:
                    return None
            
            h *= 2.0
            if h > 1e6:
                break
        return None

    root = try_brent(x)
    if root is not None:
        return root

    root = try_brent(float(x0))
    if root is not None:
        return root

    # Phase 3: Fallback - Local Minimization of |f(x)|
    # If we couldn't find a sign change, maybe it's a double root or we missed it.
    # Do a quick local search.
    best_x = x
    best_val = abs(f(x)) if np.isfinite(f(x)) else float('inf')
    
    # Check a few points around x and x0
    candidates = [x, float(x0)]
    for c in candidates:
        for scale in [1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
            for sign in [1, -1]:
                x_try = c + sign * scale
                try:
                    fx_try = f(x_try)
                    if np.isfinite(fx_try) and abs(fx_try) < best_val:
                        best_val = abs(fx_try)
                        best_x = x_try
                except:
                    pass

    return best_x
