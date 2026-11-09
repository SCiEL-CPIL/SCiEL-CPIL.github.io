import numpy as np
from scipy.optimize import brentq

def find_root(f, x0):
    x = float(x0)
    
    # 1. Check initial point
    fx = f(x)
    if not np.isfinite(fx):
        # If f(x0) is bad, try a small perturbation? 
        # Or just return x0 as best guess?
        # The task says non-finite x is failure. 
        # If f(x0) is inf/nan, we can't evaluate derivative.
        # Let's try a few nearby points to find a finite value.
        for h in [1e-3, 1e-6, 1e-9]:
            x_try = x + h
            fx_try = f(x_try)
            if np.isfinite(fx_try):
                x = x_try
                fx = fx_try
                break
        else:
            return x # Give up, return x0

    if abs(fx) <= 1e-8:
        return x

    # 2. Attempt to find a bracket
    # Start with a small step, expand exponentially
    h = 1e-3 * max(1.0, abs(x))
    a = b = None
    fa = fb = None
    best_x = x
    best_f_abs = abs(fx)
    
    for i in range(30): # Max 30 doublings
        a = x - h
        b = x + h
        
        fa = f(a)
        if np.isfinite(fa):
            if abs(fa) < best_f_abs:
                best_f_abs = abs(fa)
                best_x = a
            if abs(fa) <= 1e-8:
                return a
        else:
            fa = None

        fb = f(b)
        if np.isfinite(fb):
            if abs(fb) < best_f_abs:
                best_f_abs = abs(fb)
                best_x = b
            if abs(fb) <= 1e-8:
                return b
        else:
            fb = None

        # Check for sign change
        if fa is not None and fb is not None and fa * fb < 0:
            try:
                root = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
                if np.isfinite(root):
                    return root
            except Exception:
                pass # Fall through to fallback

        h *= 2.0

    # 3. Fallback: Newton's method from best_x
    # This handles cases where no sign change was found (e.g., double roots)
    x = best_x
    fx = f(x)
    
    # Use Secant method which reuses function calls? 
    # We have fx. Let's get another point.
    x1 = x + 1e-6 * max(1.0, abs(x))
    fx1 = f(x1)
    
    if not np.isfinite(fx1):
        return x

    for i in range(50):
        if abs(fx) <= 1e-8:
            return x
        if abs(fx1) <= 1e-8:
            return x1
            
        d = (fx1 - fx) / (x1 - x)
        if d == 0:
            break
            
        x_new = x - fx / d
        
        # Safety: if step is too large or non-finite, break
        if not np.isfinite(x_new):
            break
            
        fx_new = f(x_new)
        if not np.isfinite(fx_new):
            break
            
        # Update points for Secant
        x, fx = x1, fx1
        x1, fx1 = x_new, fx_new
        
        if abs(fx) <= 1e-8:
            return x
        if abs(fx1) <= 1e-8:
            return x1

    return x
