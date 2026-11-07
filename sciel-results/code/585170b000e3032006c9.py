"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize_scalar, brentq

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 for oscillatory equations.
    
    Strategy:
    1. Try a quick local Newton step to see if we're already close.
    2. Use global minimization of |f(x)|^2 with expanding bounded intervals.
    3. If a good bracket is found, polish with Brent's method.
    4. Keep a deadline guard to stay within wall-clock limits.
    """
    start_time = time.time()
    DEADLINE = 110.0  # safety margin before 120s limit
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    x = float(x0)
    best_x = x
    best_f = abs(f(x))
    
    # Quick check: if we're already close, return immediately
    if best_f <= 1e-8:
        return x
    
    # Phase 1: Try a few Newton steps locally (cheap, might work)
    for i in range(5):
        if time_left() < 5.0:
            break
        fx = f(x)
        if not np.isfinite(fx):
            break
        if abs(fx) <= 1e-8:
            return x
        h = 1e-6 * max(1.0, abs(x))
        d = (f(x + h) - fx) / h
        if not np.isfinite(d) or abs(d) < 1e-15:
            break
        new_x = x - fx / d
        if not np.isfinite(new_x):
            break
        x = new_x
        fx_new = f(x)
        if not np.isfinite(fx_new):
            x = best_x
            break
        if abs(fx_new) < best_f:
            best_f = abs(fx_new)
            best_x = x
        if abs(fx_new) <= 1e-8:
            return x
    
    if best_f <= 1e-8:
        return best_x
    
    # Phase 2: Global search via minimizing |f(x)|^2 over expanding intervals
    def objective(x):
        fx = f(x)
        if not np.isfinite(fx):
            return 1e10
        return fx * fx
    
    # Try multiple scales around x0
    scales = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4]
    
    for scale in scales:
        if time_left() < 10.0:
            break
        
        lo = x0 - scale
        hi = x0 + scale
        
        # Ensure the interval is reasonable
        if lo >= hi:
            continue
        
        try:
            result = minimize_scalar(
                objective,
                bounds=(lo, hi),
                method='bounded',
                options={'xatol': 1e-10, 'maxiter': 100}
            )
        except Exception:
            continue
        
        if result.success and np.isfinite(result.x):
            fx_at_min = f(result.x)
            if np.isfinite(fx_at_min) and abs(fx_at_min) < best_f:
                best_f = abs(fx_at_min)
                best_x = result.x
                if best_f <= 1e-8:
                    return best_x
        
        # If we found a very good point, try to bracket a root nearby
        if best_f < 1e-4:
            # Try to find a sign change around best_x
            for delta in [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]:
                if time_left() < 5.0:
                    break
                a = best_x - delta
                b = best_x + delta
                try:
                    fa = f(a)
                    fb = f(b)
                    if np.isfinite(fa) and np.isfinite(fb):
                        if fa * fb < 0:
                            # Sign change found, use Brent's method
                            root = brentq(f, a, b, xtol=1e-12, rtol=1e-14, maxiter=50)
                            if np.isfinite(root):
                                froot = f(root)
                                if np.isfinite(froot) and abs(froot) <= 1e-8:
                                    return root
                                if abs(froot) < best_f:
                                    best_f = abs(froot)
                                    best_x = root
                except Exception:
                    continue
    
    # Phase 3: If we have a decent candidate, try a final polish with tighter bounds
    if best_f < 1e-4 and time_left() > 5.0:
        for delta in [1e-3, 1e-2, 1e-1, 1.0]:
            if time_left() < 3.0:
                break
            a = best_x - delta
            b = best_x + delta
            try:
                fa = f(a)
                fb = f(b)
                if np.isfinite(fa) and np.isfinite(fb) and fa * fb < 0:
                    root = brentq(f, a, b, xtol=1e-13, rtol=1e-15, maxiter=60)
                    if np.isfinite(root):
                        froot = f(root)
                        if np.isfinite(froot) and abs(froot) <= 1e-8:
                            return root
                        if abs(froot) < best_f:
                            best_f = abs(froot)
                            best_x = root
            except Exception:
                continue
    
    return best_x
# EVOLVE-BLOCK-END
