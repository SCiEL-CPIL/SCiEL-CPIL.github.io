import math
import time
import numpy as np
from scipy.optimize import brentq

def find_root(f, x0):
    start_time = time.time()
    MAX_WALL = 115.0
    MAX_EVALS = 500
    TARGET_TOL = 1e-8
    
    eval_count = [0]
    
    def eval_f(x):
        eval_count[0] += 1
        return f(x)

    def time_ok():
        return time.time() - start_time < MAX_WALL

    x0 = float(x0)
    fx0 = eval_f(x0)
    best_x = x0
    best_fx = fx0
    
    if abs(fx0) <= TARGET_TOL:
        return x0

    # Strategy 1: Newton's Method
    # Try a few steps of Newton. If it converges, great. If it diverges, stop.
    # We use a finite difference derivative.
    h = 1e-5 * max(1.0, abs(x0))
    x = x0
    fx = fx0
    
    for i in range(20):
        if abs(fx) <= TARGET_TOL:
            return x
        if not time_ok() or eval_count[0] >= MAX_EVALS - 10:
            break
        
        # Calculate derivative
        # Use central difference if possible, but forward is cheaper and usually fine
        # for the "try" phase.
        if eval_count[0] >= MAX_EVALS - 1:
            break
        fx_plus = eval_f(x + h)
        df = (fx_plus - fx) / h
        
        if not math.isfinite(df) or abs(df) < 1e-15:
            # Derivative is zero or problematic, try larger h or break
            h *= 10
            if eval_count[0] >= MAX_EVALS - 1:
                break
            fx_plus = eval_f(x + h)
            df = (fx_plus - fx) / h
            if not math.isfinite(df) or abs(df) < 1e-15:
                break
        
        step = fx / df
        if not math.isfinite(step):
            break
            
        # Damping to prevent wild jumps
        max_step = 10 * max(1.0, abs(x))
        if abs(step) > max_step:
            step = math.copysign(max_step, step)
            
        x_new = x - step
        if not math.isfinite(x_new):
            break
            
        if eval_count[0] >= MAX_EVALS:
            break
        fx_new = eval_f(x_new)
        
        # If we are moving away from the root, stop Newton
        if abs(fx_new) > 10 * abs(fx):
            break
            
        x, fx = x_new, fx_new
        if abs(fx) < abs(best_fx):
            best_x, best_fx = x, fx
            
        # Adaptive h for next step
        h = 1e-5 * max(1.0, abs(x))

    if abs(fx) <= TARGET_TOL:
        return x

    # Strategy 2: Find a Bracket and use Brent's Method
    # Search for a sign change
    base_h = max(1e-6, abs(x0) * 1e-6)
    bracket_found = False
    a = None
    b = None
    fa = None
    fb = None
    
    # We already have x0, fx0.
    # Check neighbors first
    for scale in [1, 10, 100, 1000, 10000, 100000, 1000000]:
        if not time_ok() or eval_count[0] >= MAX_EVALS - 20:
            break
        h = base_h * scale
        if h < 1e-15:
            continue
            
        # Check right
        x_r = x0 + h
        if x_r != x0:
            if eval_count[0] >= MAX_EVALS - 1:
                break
            fx_r = eval_f(x_r)
            if abs(fx_r) < abs(best_fx):
                best_x, best_fx = x_r, fx_r
            if abs(fx_r) <= TARGET_TOL:
                return x_r
            if fx0 * fx_r < 0:
                a, b = x0, x_r
                fa, fb = fx0, fx_r
                bracket_found = True
                break
                
        # Check left
        x_l = x0 - h
        if x_l != x0:
            if eval_count[0] >= MAX_EVALS - 1:
                break
            fx_l = eval_f(x_l)
            if abs(fx_l) < abs(best_fx):
                best_x, best_fx = x_l, fx_l
            if abs(fx_l) <= TARGET_TOL:
                return x_l
            if fx0 * fx_l < 0:
                a, b = x_l, x0
                fa, fb = fx_l, fx0
                bracket_found = True
                break

    if bracket_found:
        # Use Brent's method
        # Wrap f to count evals and respect time limit
        # Note: brentq might call f many times. We have plenty of budget if Newton didn't use it all.
        # But we must be careful.
        
        def f_brent(x):
            if not time_ok():
                raise ValueError("Time limit exceeded")
            if eval_count[0] >= MAX_EVALS:
                raise ValueError("Eval limit exceeded")
            return eval_f(x)

        try:
            root = brentq(f_brent, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
            # Verify
            if abs(f_brent(root)) <= TARGET_TOL:
                return root
        except Exception:
            pass

    # Strategy 3: If no bracket, try to find local min of |f|
    # This handles cases where f touches 0 but doesn't cross (double root)
    # We can use a simple golden section search or just check a few points
    # Given the "local" nature, the root is likely near x0 or best_x.
    
    # Try Newton from best_x with different h
    if time_ok() and eval_count[0] < MAX_EVALS - 10:
        x = best_x
        fx = best_fx
        for h_try in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
            if not time_ok() or eval_count[0] >= MAX_EVALS - 5:
                break
            h = h_try * max(1.0, abs(x))
            if eval_count[0] >= MAX_EVALS - 1:
                break
            fx_plus = eval_f(x + h)
            df = (fx_plus - fx) / h
            if not math.isfinite(df) or abs(df) < 1e-15:
                continue
            step = fx / df
            if not math.isfinite(step):
                continue
            if abs(step) > 10 * max(1.0, abs(x)):
                step = math.copysign(10 * max(1.0, abs(x)), step)
            x_new = x - step
            if not math.isfinite(x_new):
                continue
            if eval_count[0] >= MAX_EVALS:
                break
            fx_new = eval_f(x_new)
            if abs(fx_new) <= TARGET_TOL:
                return x_new
            if abs(fx_new) < abs(best_fx):
                best_x, best_fx = x_new, fx_new

    return best_x
