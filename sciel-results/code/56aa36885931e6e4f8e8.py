import time
import numpy as np
from scipy.optimize import brentq

def find_root(f, x0):
    """
    Find a root of f near x0 using a multi-strategy approach:
    1. Quick Newton iteration (fast if close to root)
    2. Bracket expansion + Brent's method (robust)
    3. Fallback: return best point found
    """
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    x = float(x0)
    best_x = x
    best_fx = None
    
    def time_check():
        return time.time() - start_time > deadline
    
    # Phase 1: Try Newton's method for a few iterations
    for i in range(20):
        if time_check():
            break
        try:
            fx = f(x)
        except Exception:
            break
        if not np.isfinite(fx):
            break
        if best_fx is None or abs(fx) < abs(best_fx):
            best_fx = fx
            best_x = x
        if abs(fx) <= 1e-10:
            return x
        
        # Central difference for better derivative estimate
        h = 1e-8 * max(1.0, abs(x))
        try:
            fx_plus = f(x + h)
            fx_minus = f(x - h)
        except Exception:
            break
        if not (np.isfinite(fx_plus) and np.isfinite(fx_minus)):
            break
        d = (fx_plus - fx_minus) / (2 * h)
        if d == 0 or not np.isfinite(d):
            break
        x_new = x - fx / d
        if not np.isfinite(x_new):
            break
        # Damping if step is too large
        if abs(x_new - x) > 10 * max(1.0, abs(x)):
            x_new = x + np.sign(x_new - x) * 10 * max(1.0, abs(x))
        x = x_new
    
    # Phase 2: Try to bracket the root and use Brent's method
    # First, try to find a sign change near x0
    step = max(1e-6, abs(x0) * 1e-4)
    bracket_found = False
    a = None
    b = None
    
    # Try expanding outward from x0
    for expand in range(30):
        if time_check():
            break
        left = x0 - step * (2 ** expand)
        right = x0 + step * (2 ** expand)
        
        try:
            f_left = f(left)
            f_right = f(right)
        except Exception:
            continue
        
        if not (np.isfinite(f_left) and np.isfinite(f_right)):
            continue
        
        if f_left * f_right < 0:
            a, b = left, right
            bracket_found = True
            break
    
    if not bracket_found:
        # Try smaller steps around x0
        for expand in range(30):
            if time_check():
                break
            step_small = step * (0.5 ** expand)
            for offset in [step_small, -step_small, 2*step_small, -2*step_small]:
                x_try = x0 + offset
                try:
                    f_try = f(x_try)
                except Exception:
                    continue
                if not np.isfinite(f_try):
                    continue
                if best_fx is None or abs(f_try) < abs(best_fx):
                    best_fx = f_try
                    best_x = x_try
                if abs(f_try) <= 1e-10:
                    return x_try
                # Check sign change with current best_x
                if best_fx is not None and f_try * best_fx < 0:
                    a, b = (x_try, best_x) if x_try < best_x else (best_x, x_try)
                    bracket_found = True
                    break
            if bracket_found:
                break
    
    if bracket_found:
        try:
            if time_check():
                return best_x
            root = brentq(f, a, b, xtol=1e-12, rtol=1e-14, maxiter=100)
            if np.isfinite(root):
                return root
        except Exception:
            pass
    
    # Phase 3: Fallback - return best point found
    return best_x
