import math
import time
import sys
from scipy.optimize import brentq

def find_root(f, x0):
    TOL = 1e-8
    MAX_CALLS = 450
    DEADLINE = time.time() + 140
    
    calls = 0
    best_x = float(x0)
    best_abs = float('inf')
    
    def safe_f(x):
        nonlocal calls, best_x, best_abs
        if calls >= MAX_CALLS:
            return float('inf')
        if time.time() > DEADLINE:
            return float('inf')
        calls += 1
        try:
            val = f(x)
            if not math.isfinite(val):
                return float('inf')
            abs_val = abs(val)
            if abs_val < best_abs:
                best_abs = abs_val
                best_x = x
            if abs_val <= TOL:
                return val
            return val
        except Exception:
            return float('inf')

    # Initial evaluation
    fx0 = safe_f(x0)
    if abs(fx0) <= TOL:
        return x0
    
    # Bracketing search
    # Try to find a sign change
    offsets = [0.01, -0.01, 0.1, -0.1, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 5.0, -5.0, 10.0, -10.0, 20.0, -20.0, 50.0, -50.0]
    
    bracket_found = False
    a = x0
    b = None
    fa = fx0
    
    for d in offsets:
        if calls >= MAX_CALLS or time.time() > DEADLINE:
            break
        x_new = x0 + d
        f_new = safe_f(x_new)
        
        if math.isinf(f_new):
            continue
            
        if fa * f_new < 0:
            a = x0
            b = x_new
            bracket_found = True
            break
            
    if bracket_found:
        try:
            root = brentq(safe_f, a, b, xtol=1e-12, rtol=1e-12, maxiter=50)
            if math.isfinite(root):
                return root
        except Exception:
            pass
    
    # If no bracket found, try Newton's method from the best point found
    # This helps if the function is smooth and the minimum is close to a root
    x = best_x
    for _ in range(50):
        if calls >= MAX_CALLS or time.time() > DEADLINE:
            break
        fx = safe_f(x)
        if abs(fx) <= TOL:
            return x
        
        h = 1e-6 * max(1.0, abs(x))
        f_plus = safe_f(x + h)
        if math.isinf(f_plus):
            break
        d = (f_plus - fx) / h
        if d == 0 or not math.isfinite(d):
            break
        
        x_new = x - fx / d
        if not math.isfinite(x_new):
            break
            
        # Damping if step is too large
        if abs(x_new - x) > 10 * max(1.0, abs(x)):
             x_new = x + 10 * max(1.0, abs(x)) * (1 if x_new > x else -1)
             
        x = x_new
        
    return best_x
