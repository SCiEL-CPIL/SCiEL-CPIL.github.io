"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np
from scipy.optimize import newton

def find_root(f, x0):
    """
    Find a root x of f such that |f(x)| <= 1e-8.
    Uses scipy.optimize.newton with finite-difference derivative for quadratic convergence.
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

    # Initial evaluation
    try:
        best_f = _f(x0)
    except:
        best_f = float('inf')
    
    _update_best(x0, best_f)
    
    if best_f is not None and math.isfinite(best_f) and abs(best_f) <= 1e-8:
        return best_x

    # Use scipy newton with finite difference derivative
    # maxiter is set high enough to be safe, but the call limit wrapper will stop it earlier if needed
    # tol=1e-12 is tighter than required 1e-8 to ensure final check passes
    try:
        root = newton(_f, x0, tol=1e-12, maxiter=100, full_output=False)
        # newton returns the root, but we need to evaluate it to be sure and update best
        if math.isfinite(root):
            f_root = _f(root)
            _update_best(root, f_root)
            if math.isfinite(f_root) and abs(f_root) <= 1e-8:
                return root
    except Exception:
        pass
    
    # If newton failed or didn't converge sufficiently, try a few secant steps as fallback
    # This is a structural fallback, not the primary strategy
    if call_count < max_calls - 10:
        x1 = best_x
        f1 = best_f
        h = 1e-5 * max(1.0, abs(x1))
        x2 = x1 + h
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
