import numpy as np
from scipy.optimize import brentq
import time

def find_root(f, x0):
    """
    Finds a root of f(x) starting from x0.
    Returns x such that |f(x)| <= 1e-8.
    Uses at most 500 evaluations.
    """
    start_time = time.time()
    deadline = 115.0 # 115s margin out of 120s limit
    
    max_evals = 500
    eval_count = 0
    best_x = float(x0)
    best_fx = np.inf
    
    # Wrap f to count evaluations and handle exceptions/timeouts
    def safe_f(x):
        nonlocal eval_count, best_x, best_fx
        if eval_count >= max_evals:
            raise ValueError("Max evaluations exceeded")
        if time.time() > deadline:
            raise TimeoutError("Deadline exceeded")
        try:
            val = f(x)
            eval_count += 1
            # Handle non-finite values
            if not np.isfinite(val):
                return np.inf
            # Track best x
            if abs(val) < abs(best_fx):
                best_fx = val
                best_x = x
            return val
        except Exception:
            return np.inf

    x = float(x0)
    
    # Initial evaluation
    try:
        fx = safe_f(x)
    except (TimeoutError, ValueError):
        return best_x

    if abs(fx) <= 1e-8:
        return x

    # Strategy 1: Try to find a bracket
    h = 1e-4 * max(1.0, abs(x))
    
    a = None
    b = None
    
    # Try immediate neighbors
    x_plus = x + h
    x_minus = x - h
    
    try:
        f_plus = safe_f(x_plus)
        f_minus = safe_f(x_minus)
    except (TimeoutError, ValueError):
        return best_x

    # Check for sign change with x
    if np.isfinite(fx) and np.isfinite(f_plus) and fx * f_plus < 0:
        a, b = x, x_plus
    elif np.isfinite(fx) and np.isfinite(f_minus) and fx * f_minus < 0:
        a, b = x_minus, x
    else:
        # Try to expand the bracket
        if np.isfinite(f_plus) and np.isfinite(f_minus) and f_plus * f_minus < 0:
            a, b = x_minus, x_plus
        else:
            # Expand outwards
            for _ in range(20):
                h *= 10.0
                x_plus = x + h
                x_minus = x - h
                
                try:
                    f_plus = safe_f(x_plus)
                    f_minus = safe_f(x_minus)
                except (TimeoutError, ValueError):
                    return best_x
                
                if np.isfinite(fx) and np.isfinite(f_plus) and fx * f_plus < 0:
                    a, b = x, x_plus
                    break
                elif np.isfinite(fx) and np.isfinite(f_minus) and fx * f_minus < 0:
                    a, b = x_minus, x
                    break
                elif np.isfinite(f_plus) and np.isfinite(f_minus) and f_plus * f_minus < 0:
                    a, b = x_minus, x_plus
                    break
                else:
                    if abs(fx) <= 1e-8:
                        return x

    # If a bracket is found, use Brent's method
    if a is not None and b is not None:
        try:
            root = brentq(safe_f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
            if abs(safe_f(root)) <= 1e-8:
                return root
        except Exception:
            pass # Fall through to other methods

    # Strategy 2: Safeguarded Newton
    # We will use the best_x found so far as the starting point for Newton
    x_best = best_x
    fx_best = best_fx
    
    # If best_fx is inf, we need to re-evaluate or handle carefully
    if not np.isfinite(fx_best):
        # Try to find a finite value
        for dx in [1e-3, 1e-2, 1e-1, 1.0, 10.0]:
            try:
                x_try = x0 + dx
                f_try = safe_f(x_try)
                if np.isfinite(f_try):
                    x_best = x_try
                    fx_best = f_try
                    break
            except (TimeoutError, ValueError):
                return best_x
        if not np.isfinite(fx_best):
            return best_x

    for i in range(50):
        if time.time() > deadline - 5:
            break
            
        if abs(fx_best) <= 1e-8:
            return x_best
            
        # Finite difference derivative
        h_d = 1e-6 * max(1.0, abs(x_best))
        x_probe = x_best + h_d
        try:
            f_probe = safe_f(x_probe)
        except (TimeoutError, ValueError):
            break
        
        if not np.isfinite(f_probe):
            # Try smaller h
            h_d *= 0.1
            x_probe = x_best + h_d
            try:
                f_probe = safe_f(x_probe)
            except (TimeoutError, ValueError):
                break
            if not np.isfinite(f_probe):
                break
        
        d = (f_probe - fx_best) / h_d
        
        if abs(d) < 1e-15:
            break
            
        x_new = x_best - fx_best / d
        
        try:
            f_new = safe_f(x_new)
        except (TimeoutError, ValueError):
            break
        
        if np.isfinite(f_new) and abs(f_new) < abs(fx_best):
            x_best, fx_best = x_new, f_new
            if abs(fx_best) <= 1e-8:
                return x_best
        else:
            # Try damping
            for damp in [0.5, 0.25, 0.1]:
                x_damped = x_best - damp * fx_best / d
                try:
                    f_damped = safe_f(x_damped)
                except (TimeoutError, ValueError):
                    break
                if np.isfinite(f_damped) and abs(f_damped) < abs(fx_best):
                    x_best, fx_best = x_damped, f_damped
                    break
    
    return x_best
