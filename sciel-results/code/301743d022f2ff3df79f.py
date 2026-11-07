import numpy as np
from scipy.optimize import brentq

def find_root(f, x0):
    x0 = float(x0)
    
    # Check initial point
    f0 = f(x0)
    if abs(f0) <= 1e-8:
        return x0
    if not np.isfinite(f0):
        return x0 # Avoid further calls if f is broken

    # Exponential expanding bracket search
    # Start with a very small step to catch high-frequency oscillations
    step = 1e-6
    max_step = 1000.0
    
    # We need to be careful with the number of calls.
    # Each step costs 2 calls (left and right).
    # We can cache f(x0) as f0.
    # Actually, we can just evaluate new points.
    
    last_a = x0
    last_fa = f0
    last_b = x0
    last_fb = f0
    
    while step < max_step:
        a = x0 - step
        b = x0 + step
        
        # Evaluate f at a and b
        try:
            fa = f(a)
            fb = f(b)
        except Exception:
            # If f fails, break and return x0 (failure)
            break
            
        if not (np.isfinite(fa) and np.isfinite(fb)):
            # If non-finite, maybe skip or break. Let's break to avoid wasting calls.
            break
            
        # Check for sign change in [a, b]
        # Note: we should check [a, x0] and [x0, b] separately to ensure we don't miss a root at x0?
        # No, x0 is already checked.
        # If fa and fb have different signs, there is at least one root in [a, b].
        if fa * fb < 0:
            # We have a bracket [a, b].
            # However, brentq requires the function to be continuous and defined on [a, b].
            # It also requires f(a) and f(b) to have different signs.
            try:
                root = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
                fr = f(root)
                if abs(fr) <= 1e-8:
                    return root
            except Exception:
                pass
        
        # If no sign change, expand the bracket
        # We can reuse the previous evaluations?
        # In the next iteration, step becomes 2*step.
        # a_new = x0 - 2*step = x0 - 2*(step_old)
        # b_new = x0 + 2*step
        # We don't have f(a_new) or f(b_new) yet.
        # So we just evaluate new points.
        
        step *= 2.0
        
    # If we didn't find a root in the expanding bracket, 
    # try a wider fixed step scan as a fallback?
    # Or just return x0.
    # Given the "oscillatory" nature, a root likely exists nearby or the function is periodic.
    # If exponential expansion up to 1000 fails, it's unlikely to be a simple oscillation with period < 1000.
    
    return x0
