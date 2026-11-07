import time
import numpy as np
from scipy.optimize import minimize_scalar

def find_root(f, x0):
    start_time = time.time()
    deadline = 100.0  # Leave margin for 120s limit
    call_count = 0
    max_calls = 450
    
    best_x = float(x0)
    best_f2 = float('inf')
    
    def obj(x):
        nonlocal call_count, best_x, best_f2
        if time.time() - start_time > deadline:
            raise TimeoutError("Time limit exceeded")
        call_count += 1
        if call_count > max_calls:
            raise TimeoutError("Call limit exceeded")
        fx = f(x)
        f2 = fx * fx
        if f2 < best_f2:
            best_f2 = f2
            best_x = x
        return f2

    # Try expanding intervals
    # Start with a small interval, expand if no root found
    widths = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 500.0]
    
    for w in widths:
        lo = x0 - w
        hi = x0 + w
        try:
            # maxiter=50 is usually enough for bounded Brent
            # xatol=1e-12 ensures high precision in x
            res = minimize_scalar(obj, bounds=(lo, hi), method='bounded', 
                                  options={'maxiter': 50, 'xatol': 1e-12, 'fatol': 1e-16})
            x_opt = res.x
            # Check the actual residual
            # Note: res.fun is f(x_opt)^2
            if np.sqrt(res.fun) <= 1e-8:
                return x_opt
        except TimeoutError:
            break
        except Exception:
            continue
            
    # If no root found with |f(x)| <= 1e-8, return the best x found
    # This might still be a failure if |f(best_x)| > 1e-8, but it's the best we have
    return best_x
