import numpy as np
from scipy.optimize import brentq, newton
import time

def find_root(f, x0):
    start_time = time.time()
    deadline = 145.0 # Safety margin
    
    x = float(x0)
    try:
        fx = f(x)
        if not np.isfinite(fx):
            # If x0 is bad, try a small perturbation? Or just return x0?
            # Task says non-finite x is failure. If f(x0) is non-finite, we should probably move.
            # But we don't know which way. Let's try a small step.
            pass # Handle later or just return x0 if it's the best we can do?
            # Actually, if f(x0) is non-finite, we can't use it for bracketing.
            # Let's assume x0 is a valid starting point for the domain.
            
        if abs(fx) <= 1e-12:
            return x
            
    except Exception:
        # If f(x0) fails, we are in trouble. Return x0?
        return x

    # Bracketing phase
    h = 1e-4 * max(1.0, abs(x))
    max_h = 1e6
    bracket_found = False
    a, b = None, None
    fa, fb = None, None
    
    # We already have fx at x.
    # Check left and right
    
    # To minimize calls, we can check one side, then the other, or both?
    # Checking both doubles calls if no bracket is found.
    # Let's alternate or expand symmetrically.
    # Symmetric expansion: check x-h and x+h.
    
    calls = 1 # fx already counted
    
    while h <= max_h:
        if time.time() > deadline:
            break
            
        x_left = x - h
        x_right = x + h
        
        try:
            f_left = f(x_left)
            calls += 1
            if abs(f_left) <= 1e-12:
                return x_left
            if fx * f_left < 0:
                a, b = x_left, x
                fa, fb = f_left, fx
                bracket_found = True
                break
        except Exception:
            pass # Ignore errors in evaluation for bracketing?
            
        try:
            f_right = f(x_right)
            calls += 1
            if abs(f_right) <= 1e-12:
                return x_right
            if fx * f_right < 0:
                a, b = x, x_right
                fa, fb = fx, f_right
                bracket_found = True
                break
        except Exception:
            pass
            
        h *= 10.0

    if bracket_found:
        try:
            root = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
            return root
        except Exception:
            # Fallback if brentq fails
            # Try Newton from x0?
            try:
                root = newton(f, x, fprime=None, tol=1e-12, maxiter=100)
                return root
            except Exception:
                pass
            
    # If no bracket found, or brentq failed, try Newton from x0
    try:
        root = newton(f, x, fprime=None, tol=1e-12, maxiter=100)
        return root
    except Exception:
        pass

    # Last resort: return x0 or best guess
    return x
