"""Root finding for ill-conditioned equations: Robust Adaptive Hybrid."""
import math
import time
import numpy as np
from scipy.optimize import brentq

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin below 120s limit
    MAX_EVALS = 450   # Safety margin below 500 limit
    
    # State tracking
    eval_count = 0
    best_x = float(x0)
    best_f = None
    
    def check_budget():
        nonlocal eval_count
        if eval_count > MAX_EVALS:
            return True
        if time.time() - start_time > DEADLINE:
            return True
        return False

    def safe_eval(x):
        nonlocal eval_count, best_x, best_f
        if check_budget():
            return None
        try:
            val = f(x)
            eval_count += 1
            if val is None or not np.isfinite(val):
                return None
            val = float(val)
            if best_f is None or abs(val) < abs(best_f):
                best_f = val
                best_x = x
            return val
        except Exception:
            return None

    # Initial evaluation
    fx0 = safe_eval(best_x)
    if fx0 is None:
        return best_x
    if abs(fx0) <= 1e-8:
        return best_x

    # Strategy 1: Try Newton's method with finite differences for a few steps
    # This is fast if it works, but we guard it.
    x = best_x
    fx = fx0
    for i in range(10):
        if check_budget():
            break
        h = 1e-6 * max(1.0, abs(x))
        f_xh = safe_eval(x + h)
        if f_xh is None:
            break
        d = (f_xh - fx) / h
        if d == 0 or not np.isfinite(d):
            break
        x_new = x - fx / d
        if not np.isfinite(x_new):
            break
        
        fx_new = safe_eval(x_new)
        if fx_new is None:
            break
            
        if abs(fx_new) <= 1e-8:
            return x_new
            
        # Check for divergence or stagnation
        if abs(x_new - x) > 1e6 * max(1.0, abs(x)):
            break
            
        x = x_new
        fx = fx_new
        
        if abs(fx) <= 1e-8:
            return x

    # Strategy 2: If Newton failed or didn't converge, try to bracket a root
    # We expand from x0 to find a sign change.
    # If no sign change is found, we might try to minimize |f(x)| locally,
    # but for this task, sign change bracketing is the primary robust method.
    
    # We will try to find a bracket [a, b] such that f(a)*f(b) < 0
    # or |f(a)| or |f(b)| is small enough.
    
    # Start from current best x (which might be different from x0 if Newton moved)
    center = x
    fc = fx
    
    # If Newton moved us to a worse spot, maybe reset to x0? 
    # Actually, best_x tracks the global minimum of |f| so far.
    # Let's try expanding from the best_x found so far.
    
    a = best_x
    fa = best_f
    
    # If we haven't found a sign change, we need to expand.
    # We will try expanding in both directions.
    
    # Helper to try to find a bracket starting from a point
    def try_bracket(start_x, start_f):
        if check_budget():
            return None
        
        # If start_f is already good, return it
        if abs(start_f) <= 1e-8:
            return (start_x, start_x) # Degenerate bracket, but root is here
        
        step = 1e-3
        max_steps = 50
        
        # Try expanding to the right
        for i in range(max_steps):
            if check_budget():
                return None
            x_try = start_x + step
            f_try = safe_eval(x_try)
            if f_try is None:
                break
            if abs(f_try) <= 1e-8:
                return (x_try, x_try)
            if start_f * f_try < 0:
                return (start_x, x_try)
            # Double the step if we haven't found a sign change
            step *= 2.0
            if step > 1e10:
                break
        
        # Try expanding to the left
        step = 1e-3
        for i in range(max_steps):
            if check_budget():
                return None
            x_try = start_x - step
            f_try = safe_eval(x_try)
            if f_try is None:
                break
            if abs(f_try) <= 1e-8:
                return (x_try, x_try)
            if start_f * f_try < 0:
                return (x_try, start_x)
            step *= 2.0
            if step > 1e10:
                break
                
        return None

    bracket = try_bracket(best_x, best_f)
    
    if bracket:
        a, b = bracket
        if a == b:
            return a
        # Use Brent's method for robust convergence within the bracket
        try:
            # maxiter is limited to ensure we don't blow the eval budget
            # brentq uses roughly 10-20 evals per iteration usually, but let's be safe
            # We have ~400 evals left max.
            root = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
            return root
        except Exception:
            pass

    # If bracketing failed, return the best point found so far
    return best_x

# EVOLVE-BLOCK-END
