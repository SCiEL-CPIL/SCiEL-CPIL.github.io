"""Root finding for ill-conditioned problems using SciPy's optimized solvers."""
# EVOLVE-BLOCK-START
import math
from scipy.optimize import brentq

def find_root(f, x0):
    """
    Find a root of f starting near x0.
    Returns x with |f(x)| <= 1e-8 using at most 500 evaluations.
    Uses SciPy's brentq for bracketed roots and a robust secant fallback.
    """
    x = float(x0)
    max_evals = 500
    eval_count = 0
    
    def eval_f(val):
        nonlocal eval_count
        if eval_count >= max_evals:
            raise RuntimeError("Max evaluations exceeded")
        result = f(val)
        eval_count += 1
        return result

    # Initial evaluation
    fx = eval_f(x)
    if abs(fx) <= 1e-8:
        return x

    # Strategy 1: Find a bracket
    a = x
    fa = fx
    h = 1e-4 * max(1.0, abs(x))
    
    # Try expanding to the right
    bracket_found = False
    b = None
    fb = None
    for i in range(30):
        b = x + h * (2**i)
        if abs(b) > 1e15:
            break
        fb = eval_f(b)
        if abs(fb) <= 1e-8:
            return b
        if fa * fb < 0:
            bracket_found = True
            break
        if eval_count >= max_evals:
            break
            
    if not bracket_found:
        # Try expanding to the left
        for i in range(30):
            b = x - h * (2**i)
            if abs(b) > 1e15:
                break
            fb = eval_f(b)
            if abs(fb) <= 1e-8:
                return b
            if fa * fb < 0:
                bracket_found = True
                break
            if eval_count >= max_evals:
                break

    if bracket_found:
        try:
            # Use SciPy's brentq which is highly optimized
            root = brentq(eval_f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
            return root
        except Exception:
            # Fallback if brentq fails for some reason
            pass

    # Strategy 2: No bracket found. Use Secant method.
    # We need two points. We have x (a) and the last b evaluated.
    # If we didn't evaluate any b (unlikely), use x+h.
    if 'b' not in locals() or b is None:
        b = x + h
        fb = eval_f(b)
        if abs(fb) <= 1e-8:
            return b

    x0_sec = a
    f0_sec = fa
    x1_sec = b
    f1_sec = fb
    
    for _ in range(max_evals - eval_count):
        if abs(f1_sec) <= 1e-8:
            return x1_sec
        if abs(f0_sec) <= 1e-8:
            return x0_sec
            
        if f0_sec == f1_sec:
             x1_sec += 1e-6
             f1_sec = eval_f(x1_sec)
             continue
             
        x_new = x1_sec - f1_sec * (x1_sec - x0_sec) / (f1_sec - f0_sec)
        
        # Safeguard against large steps
        step = x_new - x1_sec
        if abs(step) > 10 * max(1.0, abs(x1_sec)):
            x_new = x1_sec + 0.5 * step
            
        f_new = eval_f(x_new)
        if abs(f_new) <= 1e-8:
            return x_new
            
        x0_sec, f0_sec = x1_sec, f1_sec
        x1_sec, f1_sec = x_new, f_new
        
        if abs(x_new - x0_sec) < 1e-12 * max(1.0, abs(x0_sec), abs(x1_sec)):
            break
            
    return x1_sec if abs(f1_sec) < abs(f0_sec) else x0_sec

# EVOLVE-BLOCK-END
