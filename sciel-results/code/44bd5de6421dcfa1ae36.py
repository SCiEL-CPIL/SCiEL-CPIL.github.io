import math
from scipy.optimize import brentq

def find_root(f, x0):
    x = float(x0)
    
    # Check x0
    try:
        fx = f(x)
    except Exception:
        return x # Will fail, but we must return something
    
    if math.isfinite(fx) and abs(fx) <= 1e-8:
        return x

    if not math.isfinite(fx):
        # If x0 is bad, try a small step
        # This is a fallback, might not work
        pass

    # Bracket search
    a = None
    b = None
    fa = fx
    fb = None
    
    steps = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
    
    for step in steps:
        # Try positive
        xp = x + step
        try:
            fpx = f(xp)
        except Exception:
            fpx = float('nan')
        
        if math.isfinite(fpx):
            if abs(fpx) <= 1e-8:
                return xp
            if math.isfinite(fa) and fa * fpx < 0:
                a = x
                b = xp
                fa = fa
                fb = fpx
                break
        
        # Try negative
        xm = x - step
        try:
            fxm = f(xm)
        except Exception:
            fxm = float('nan')
            
        if math.isfinite(fxm):
            if abs(fxm) <= 1e-8:
                return xm
            if math.isfinite(fa) and fa * fxm < 0:
                a = xm
                b = x
                fa = fxm
                fb = fa
                break
            # Check if xp and xm bracket (skipping x)
            if math.isfinite(fpx) and fpx * fxm < 0:
                a = xm
                b = xp
                fa = fxm
                fb = fpx
                break
        
    if a is None:
        # Fallback: If no bracket found, try Newton? Or return x0?
        # If no bracket, we might be in a region with no roots.
        # Return x0 to fail gracefully? Or try a wider search?
        # Let's try a few more larger steps
        for step in [1000.0, 10000.0]:
             # ... same logic ...
             pass
        
        # If still no bracket, return x0 (failure)
        return x

    # Use Brent's method
    try:
        root = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
        return root
    except Exception:
        # If brentq fails, return midpoint or best guess
        return (a + b) / 2.0
