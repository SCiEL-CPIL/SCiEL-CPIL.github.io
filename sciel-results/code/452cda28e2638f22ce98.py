import numpy as np
from scipy.optimize import brentq, newton

def find_root(f, x0):
    eval_count = 0
    max_evals = 500
    
    def f_counted(x):
        nonlocal eval_count
        eval_count += 1
        if eval_count > max_evals:
            raise ValueError("Max evaluations exceeded")
        return f(x)

    x = float(x0)
    fx = f_counted(x)
    
    if abs(fx) <= 1e-8:
        return x

    # Try to find a bracket
    # Start with a small step
    h = 1e-4 * max(1.0, abs(x))
    a, b = None, None
    
    # We will try to expand outwards
    # Check immediate neighbors
    # f(x-h) and f(x+h)
    
    fa = None
    fb = None
    xa = x - h
    xb = x + h
    
    # We need to be careful with eval count.
    # Let's try a few steps of expansion.
    
    found_bracket = False
    # Try expanding h up to some limit
    for i in range(30):
        xa = x - h
        xb = x + h
        
        # Evaluate f(xa) and f(xb) if not done
        if fa is None or xa != x - h: # This logic is getting complex with state
            pass
            
    # Simpler bracketing strategy:
    # 1. Check f(x0)
    # 2. Check f(x0 - h), f(x0 + h)
    # 3. If no sign change, increase h.
    
    h = 1e-4 * max(1.0, abs(x))
    # We already have fx
    for i in range(40):
        xa = x - h
        xb = x + h
        
        # To save evals, we can check one side at a time or both.
        # Let's check both.
        try:
            fa = f_counted(xa)
            fb = f_counted(xb)
        except ValueError:
            break
        
        if fa * fx < 0:
            a, b = xa, x
            found_bracket = True
            break
        if fx * fb < 0:
            a, b = x, xb
            found_bracket = True
            break
            
        # If no sign change, expand
        h *= 2.0
        # Limit expansion to avoid overflow or huge numbers
        if h > 1e10:
            break

    if found_bracket:
        try:
            root = brentq(f_counted, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
            return root
        except Exception:
            pass # Fallback if brentq fails

    # Fallback: Newton's method
    # We have used some evals. Let's try newton.
    # newton uses finite difference if fprime is not given.
    try:
        # maxiter limits function calls roughly
        root = newton(f_counted, x, maxiter=100)
        if abs(f_counted(root)) <= 1e-8:
            return root
    except Exception:
        pass

    # Fallback: Secant method using x and x+h (small h)
    # Secant needs two points. We have x and maybe xa/xb from before?
    # Let's just pick x and x + 1e-6
    try:
        # scipy.optimize.root_scalar method='secant'
        from scipy.optimize import root_scalar
        res = root_scalar(f_counted, x0=x, x1=x + 1e-6 * max(1.0, abs(x)), method='secant', maxiter=100)
        if res.converged:
            return res.root
    except Exception:
        pass

    # If all fails, return x0 or best guess
    return x
