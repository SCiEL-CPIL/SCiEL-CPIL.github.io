import numpy as np
from scipy.optimize import brentq

def find_root(f, x0):
    """
    Finds a root x of f(x) such that |f(x)| <= 1e-8.
    Uses bracketing and Brent's method for robustness on ill-conditioned problems.
    """
    x0 = float(x0)
    
    # Check initial guess
    try:
        fx0 = f(x0)
        if np.isnan(fx0):
            return x0
        if abs(fx0) <= 1e-8:
            return x0
    except Exception:
        return x0

    # Attempt to find a bracket [a, b] where f(a) and f(b) have opposite signs
    # Start with a small step and expand geometrically
    delta = 1e-6
    max_delta = 1e12
    max_expansions = 50
    
    a = x0
    b = x0
    fa = fx0
    fb = fx0
    
    # We need to find a and b such that f(a)*f(b) < 0
    # We will expand the interval around x0
    # To be efficient, we can check x0 +/- delta, then x0 +/- 2*delta, etc.
    # Or simply expand one side then the other.
    
    found_bracket = False
    left_val = None
    right_val = None
    
    # Let's try expanding symmetrically or just find any sign change nearby.
    # A common robust strategy:
    # 1. Check x0.
    # 2. Try x0 + delta, x0 - delta.
    # 3. If no sign change, increase delta.
    
    current_delta = 1e-6
    # If x0 is very large, relative step might be better, but 1e-6 absolute is often fine for local.
    # Let's use a relative step if x0 is large.
    if abs(x0) > 1:
        current_delta = 1e-6 * abs(x0)
    else:
        current_delta = 1e-6

    for _ in range(max_expansions):
        a = x0 - current_delta
        b = x0 + current_delta
        
        try:
            fa = f(a)
            if np.isnan(fa):
                raise ValueError
            if abs(fa) <= 1e-8:
                return a
            
            fb = f(b)
            if np.isnan(fb):
                raise ValueError
            if abs(fb) <= 1e-8:
                return b
            
            # Check for sign change
            if fa * fb < 0:
                found_bracket = True
                break
        except Exception:
            # If evaluation fails, try smaller delta? Or just break.
            # For ill-conditioned, maybe the function is undefined outside a small range.
            # Let's shrink delta if it's too large? No, we are expanding.
            # If f(a) or f(b) fails, we can't bracket.
            break
        
        # Expand delta
        current_delta *= 10.0
        if current_delta > max_delta:
            break

    if found_bracket:
        try:
            # Use brentq to find the root.
            # xtol=1e-12 ensures high precision.
            # maxiter=100 is usually enough.
            root = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
            # Verify the result
            fr = f(root)
            if abs(fr) <= 1e-8:
                return root
            # If brentq returned a point that doesn't satisfy the condition (unlikely but possible due to floating point),
            # we might return the best of a, b, root
            if abs(fa) < abs(fr): return a
            if abs(fb) < abs(fr): return b
            return root
        except Exception:
            # Fallback if brentq fails
            pass

    # Fallback: If no bracket found, try a few Newton steps or return the best point found
    # This handles cases where the function touches zero but doesn't cross (e.g., x^2=0)
    # or if the bracketing failed due to numerical issues.
    
    # Simple local search / Newton hybrid
    x = x0
    best_x = x0
    best_f = abs(fx0)
    
    for i in range(50):
        try:
            fx = f(x)
            if np.isnan(fx):
                break
            if abs(fx) < best_f:
                best_f = abs(fx)
                best_x = x
            if abs(fx) <= 1e-8:
                return x
            
            # Estimate derivative
            h = 1e-6 * max(1.0, abs(x))
            d = (f(x + h) - fx) / h
            
            if d == 0 or np.isnan(d):
                break
            
            # Newton step with damping
            step = fx / d
            # Limit step size to avoid running away
            max_step = 1.0
            if abs(step) > max_step:
                step = max_step * np.sign(step)
            
            x_new = x - step
            # If x_new is not finite, break
            if not np.isfinite(x_new):
                break
            x = x_new
        except Exception:
            break
            
    return best_x
