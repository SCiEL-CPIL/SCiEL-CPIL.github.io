"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np
from scipy.optimize import minimize_scalar

def find_root(f, x0):
    """
    Find a root x of f such that |f(x)| <= 1e-8.
    Uses global optimization (minimize_scalar) on f(x)^2 to find the root.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    max_calls = 480  # Stay well under 500
    
    call_count = 0
    best_x = float(x0)
    best_f = None
    
    # Wrapper to count calls and handle limits
    def _f(x):
        nonlocal call_count
        if call_count >= max_calls:
            raise RuntimeError("call limit")
        if time.time() > deadline:
            raise RuntimeError("time limit")
        call_count += 1
        val = f(x)
        if not math.isfinite(val):
            return float('inf')
        return val

    def _update_best(x, fx):
        nonlocal best_x, best_f
        if fx is not None and math.isfinite(fx) and (best_f is None or not math.isfinite(best_f) or abs(fx) < abs(best_f)):
            best_x = x
            best_f = fx

    # Objective function for minimization: f(x)^2
    def objective(x):
        fx = _f(x)
        if not math.isfinite(fx):
            return float('inf')
        return fx * fx

    # Initial evaluation
    try:
        best_f = _f(x0)
    except:
        best_f = float('inf')
    
    _update_best(x0, best_f)
    
    if best_f is not None and math.isfinite(best_f) and abs(best_f) <= 1e-8:
        return best_x

    # Global search strategy:
    # 1. Start with a small window around x0
    # 2. Expand the window if no root is found
    # 3. Use minimize_scalar (bounded) to find the minimum of f(x)^2
    
    # Determine initial window size
    # Polynomials usually have roots within a reasonable distance of x0 if x0 is a "starting point"
    # But we don't know the scale. Start with 1.0 and expand.
    
    current_scale = 1.0
    max_scale = 1e6
    
    while current_scale <= max_scale:
        if time.time() > deadline - 5.0 or call_count >= max_calls - 10:
            break
        
        lower = x0 - current_scale
        upper = x0 + current_scale
        
        try:
            # Use bounded minimization
            # method='bounded' is robust and uses Brent's method internally
            # options: maxiter controls iterations, but we have a call limit wrapper
            res = minimize_scalar(objective, bounds=(lower, upper), method='bounded', 
                                  options={'xatol': 1e-12, 'maxiter': 100})
            
            if res.success:
                x_candidate = res.x
                f_candidate = _f(x_candidate)
                _update_best(x_candidate, f_candidate)
                
                if math.isfinite(f_candidate) and abs(f_candidate) <= 1e-8:
                    return x_candidate
            else:
                # Even if not fully successful, check the best point found
                x_candidate = res.x
                f_candidate = _f(x_candidate)
                _update_best(x_candidate, f_candidate)
                
                if math.isfinite(f_candidate) and abs(f_candidate) <= 1e-8:
                    return x_candidate
                    
        except Exception:
            pass
        
        # Expand the search window
        current_scale *= 10.0

    # If global search didn't find a root within the expanded bounds,
    # fallback to a local Newton step from the best point found so far
    # This handles cases where the root is very close to x0 but minimize_scalar
    # might have missed it due to local minima or numerical issues, or if the
    # root is extremely close and the bounded method converged to a nearby point
    # but not precise enough.
    
    if call_count < max_calls - 5:
        x_local = best_x
        f_local = best_f
        
        if f_local is None or not math.isfinite(f_local):
            # Try a small perturbation if best_f is bad
            h = 1e-5 * max(1.0, abs(x_local))
            x_pert = x_local + h
            try:
                f_pert = _f(x_pert)
                _update_best(x_pert, f_pert)
                x_local = best_x
                f_local = best_f
            except:
                pass
        
        if f_local is not None and math.isfinite(f_local) and abs(f_local) > 1e-8:
            # Simple secant/Newton hybrid from best point
            h = 1e-5 * max(1.0, abs(x_local))
            x1 = x_local
            f1 = f_local
            x2 = x_local + h
            try:
                f2 = _f(x2)
                _update_best(x2, f2)
                
                for _ in range(20):
                    if call_count >= max_calls - 5 or time.time() > deadline - 5.0:
                        break
                    if abs(f2) <= 1e-8:
                        return x2
                    if abs(f1) <= 1e-8:
                        return x1
                    if f1 == f2:
                        break
                    x_new = x2 - f2 * (x2 - x1) / (f2 - f1)
                    if not math.isfinite(x_new):
                        break
                    # Damping
                    if abs(x_new - x2) > 10 * abs(x2) + 10:
                        x_new = x2 + 0.5 * (x2 - x1)
                    try:
                        f_new = _f(x_new)
                    except:
                        break
                    _update_best(x_new, f_new)
                    if abs(f_new) <= 1e-8:
                        return x_new
                    x1, f1 = x2, f2
                    x2, f2 = x_new, f_new
            except:
                pass

    return best_x
# EVOLVE-BLOCK-END
