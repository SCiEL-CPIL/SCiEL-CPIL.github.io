import math
import time
import numpy as np
from scipy.optimize import minimize_scalar

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

    # Objective function for minimization: f(x)^2
    # We minimize this to find where f(x) is closest to 0.
    def objective(x):
        if not time_ok():
            raise ValueError("Time limit exceeded")
        if eval_count[0] >= MAX_EVALS:
            raise ValueError("Eval limit exceeded")
        val = eval_f(x)
        # If val is nan or inf, return a large penalty
        if not math.isfinite(val):
            return 1e300
        return val * val

    # We will try a sequence of expanding intervals around x0.
    # The "bounded" method of minimize_scalar requires a finite interval [a, b].
    # For ill-conditioned problems, the root is likely close to x0, but we expand
    # to be safe.
    
    # Initial scale for the interval
    # We start with a small interval and expand if the minimum is at the boundary.
    # However, minimize_scalar(bounded) finds a local minimum within the bounds.
    # If the root is outside, it will find the boundary.
    
    # Strategy:
    # 1. Try a few different interval widths.
    # 2. For each width, run minimize_scalar.
    # 3. Track the best result.
    # 4. Stop if we find |f(x)| <= 1e-8 or if we run out of evals/time.
    
    # Scales to try for the half-width of the interval
    # Start small to capture local minima efficiently, then expand.
    # Note: If x0 is 0, we use absolute scale.
    base_scale = max(1.0, abs(x0))
    
    # We'll use a geometric progression of scales
    # 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100
    scales = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
    
    best_result_x = best_x
    best_result_fx = best_fx
    best_result_val = best_fx * best_fx

    for scale in scales:
        if not time_ok():
            break
        if eval_count[0] >= MAX_EVALS - 20: # Leave some buffer
            break
            
        h = scale * base_scale
        if h < 1e-15:
            h = 1e-15
            
        a = x0 - h
        b = x0 + h
        
        # Ensure a < b
        if a >= b:
            a = x0 - 1e-15
            b = x0 + 1e-15

        try:
            # Use bounded method. It uses Brent's method for minimization.
            # It does not require derivatives.
            # xtol and rtol control the precision of the minimizer.
            # We want high precision to satisfy |f(x)| <= 1e-8.
            # Note: minimizing f(x)^2 means the derivative is 2f(x)f'(x).
            # If f'(x) is very small (ill-conditioned), the minimum might be flat,
            # but the value f(x)^2 will be small.
            
            res = minimize_scalar(
                objective, 
                bounds=(a, b), 
                method='bounded',
                options={'xatol': 1e-10, 'maxiter': 100}
            )
            
            if res.success or hasattr(res, 'x'):
                x_candidate = res.x
                # Evaluate at the candidate
                if eval_count[0] < MAX_EVALS and time_ok():
                    fx_candidate = eval_f(x_candidate)
                    if math.isfinite(fx_candidate):
                        val_candidate = fx_candidate * fx_candidate
                        if val_candidate < best_result_val:
                            best_result_val = val_candidate
                            best_result_x = x_candidate
                            best_result_fx = fx_candidate
                        
                        if abs(fx_candidate) <= TARGET_TOL:
                            return x_candidate
                            
        except Exception:
            # If minimization fails, continue to next scale
            pass
            
    return best_result_x
