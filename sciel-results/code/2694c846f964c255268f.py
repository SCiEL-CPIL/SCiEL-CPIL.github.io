"""
Root finder using Adaptive Bracketing + Brent's Method.
Strategy:
1. Check if x0 is already a root.
2. Use geometric expansion (doubling step size) to find a bracket [a, b] 
   where f(a) and f(b) have opposite signs.
3. Use scipy.optimize.brentq to find the root within the bracket.
   Brent's method is robust and superlinearly convergent.
4. Include a time guard to prevent exceeding the 150s limit.
"""
# EVOLVE-BLOCK-START
import math
import time
from scipy.optimize import brentq

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using Brent's method with adaptive bracketing.
    f: callable, takes float, returns float
    x0: float starting point
    returns: float x with |f(x)| <= 1e-8
    """
    start_time = time.time()
    time_limit = 140.0  # Safety margin before 150s kill
    
    x0 = float(x0)
    
    # Check initial point
    fx0 = f(x0)
    if not math.isfinite(fx0):
        # If f(x0) is non-finite, we are in trouble, but let's try to move away
        # For now, just return x0 as a fallback, though it will likely fail
        return x0
        
    if abs(fx0) <= 1e-8:
        return x0

    # Adaptive Bracketing
    # We want to find a, b such that f(a)*f(b) < 0
    # Start with a = x0
    a = x0
    fa = fx0
    
    # Try expanding in both directions
    # Geometric expansion: step *= 2
    step = 1.0
    b = None
    fb = None
    
    # We'll alternate directions or just try positive first, then negative
    # Actually, let's just try expanding in the positive direction first, then negative
    # Or better, try both in parallel? No, sequential is fine.
    
    # Direction 1: Positive
    for i in range(50):
        if time.time() - start_time > time_limit:
            # Return best guess? Or just x0?
            # If we haven't found a bracket, we can't use brentq.
            # We might try a local minimization or just return x0.
            # For now, return x0.
            return x0
            
        b = a + step
        fb = f(b)
        if not math.isfinite(fb):
            step *= 2.0
            continue
            
        if fa * fb < 0:
            # Found bracket
            break
        
        step *= 2.0
        if step > 1e10:
            break
            
    # If not found in positive, try negative
    if b is None or not math.isfinite(fb) or fa * fb >= 0:
        step = 1.0
        for i in range(50):
            if time.time() - start_time > time_limit:
                return x0
                
            b = a - step
            fb = f(b)
            if not math.isfinite(fb):
                step *= 2.0
                continue
                
            if fa * fb < 0:
                # Found bracket
                break
                
            step *= 2.0
            if step > 1e10:
                break

    # If we still don't have a bracket, we might try a wider random search or just fail
    # But for smooth polynomials, a root is usually close if x0 is a "starting point"
    # If no bracket found, we can't use brentq.
    # Fallback: Try a simple secant method without bracket guarantee?
    # Or just return the point with smallest |f| found so far?
    
    if b is None or not math.isfinite(fb) or fa * fb >= 0:
        # No bracket found.
        # Try a few secant steps just in case?
        # Or return x0.
        # Let's try a quick secant attempt as a last resort
        x_prev = x0
        f_prev = fx0
        x_curr = x0 + 1.0
        f_curr = f(x_curr)
        
        for i in range(20):
            if time.time() - start_time > time_limit:
                break
            if abs(f_curr) <= 1e-8:
                return x_curr
            if abs(f_prev) <= 1e-8:
                return x_prev
            if f_curr == f_prev:
                break
            x_new = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
            if not math.isfinite(x_new):
                break
            x_prev, f_prev = x_curr, f_curr
            x_curr = x_new
            f_curr = f(x_curr)
            
        if abs(f_curr) <= 1e-8:
            return x_curr
        if abs(f_prev) <= 1e-8:
            return x_prev
        
        # Return the best point found
        return x_curr if abs(f_curr) < abs(f_prev) else x_prev

    # We have a bracket [a, b] with fa*fb < 0
    # Use Brent's method
    try:
        # xtol=1e-12 is tighter than 1e-8 to ensure |f(x)| <= 1e-8
        # maxiter=100 is usually plenty
        root = brentq(f, a, b, xtol=1e-12, maxiter=100)
        return root
    except Exception:
        # If brentq fails for some reason, return the midpoint or best endpoint
        return (a + b) / 2.0

# EVOLVE-BLOCK-END
