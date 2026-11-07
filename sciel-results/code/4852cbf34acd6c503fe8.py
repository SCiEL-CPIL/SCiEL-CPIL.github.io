import math
import time
from scipy.optimize import brentq, minimize_scalar

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 105.0  # Safety margin before 150s limit
    
    x = float(x0)
    fx = f(x)
    
    # Check if x0 is already a root
    if abs(fx) <= 1e-8:
        return x

    # Strategy: Find a bracket [a, b] where f(a) and f(b) have opposite signs.
    # Then use Brent's method to converge quickly.
    
    best_x = x
    best_f = fx
    best_abs = abs(fx)
    
    a = x
    fa = fx
    
    # Try expanding in both directions
    # We will test points x0 + delta and x0 - delta
    # delta starts small and grows exponentially
    
    # Adaptive bracketing search
    # Start with small delta, expand geometrically
    delta = 1e-4
    max_delta = 1e6
    
    bracket_found = False
    lo = 0.0
    hi = 0.0
    flo = 0.0
    fhi = 0.0
    
    while delta <= max_delta:
        if time.time() > deadline:
            break
            
        # Check right side
        x_right = x + delta
        try:
            fr = f(x_right)
        except:
            fr = None
        
        if fr is not None and math.isfinite(fr):
            abs_fr = abs(fr)
            if abs_fr < best_abs:
                best_abs = abs_fr
                best_x = x_right
                best_f = fr
            if abs_fr <= 1e-8:
                return x_right
            if fa * fr < 0:
                lo, hi = x, x_right
                flo, fhi = fa, fr
                bracket_found = True
                break
        
        # Check left side
        x_left = x - delta
        try:
            fl = f(x_left)
        except:
            fl = None
            
        if fl is not None and math.isfinite(fl):
            abs_fl = abs(fl)
            if abs_fl < best_abs:
                best_abs = abs_fl
                best_x = x_left
                best_f = fl
            if abs_fl <= 1e-8:
                return x_left
            if fa * fl < 0:
                lo, hi = x_left, x
                flo, fhi = fl, fa
                bracket_found = True
                break
        
        if time.time() > deadline:
            break
        
        # Expand delta
        delta *= 2.0

    if bracket_found:
        # Use Brent's method for fast, robust convergence
        try:
            root = brentq(f, lo, hi, xtol=1e-12, rtol=1e-12, maxiter=100)
            if time.time() < deadline:
                # Verify
                if abs(f(root)) <= 1e-8:
                    return root
        except Exception:
            pass

    # If no bracket found, or brentq failed, try local minimization of |f(x)|
    # This handles cases where the root is a tangent (no sign change) or bracketing failed
    if best_abs > 1e-8:
        # Define the objective function to minimize: f(x)^2
        def obj(x):
            try:
                val = f(x)
                if not math.isfinite(val):
                    return 1e100
                return val * val
            except:
                return 1e100

        # Search in a window around best_x
        search_width = max(delta, 1.0) * 10
        
        try:
            res = minimize_scalar(obj, bounds=(best_x - search_width, best_x + search_width), method='bounded', options={'xatol': 1e-12, 'maxiter': 100})
            if res.success and res.fun <= 1e-16:
                return res.x
        except Exception:
            pass
        
        # Fallback to Newton-Raphson refinement
        for _ in range(20):
            if time.time() > deadline:
                break
            h = 1e-8 * max(1.0, abs(best_x))
            try:
                f_best = f(best_x)
                f_pert = f(best_x + h)
                if not (math.isfinite(f_best) and math.isfinite(f_pert)):
                    break
                d = (f_pert - f_best) / h
                if d == 0 or not math.isfinite(d):
                    break
                x_new = best_x - f_best / d
                if not math.isfinite(x_new):
                    break
                f_new = f(x_new)
                if math.isfinite(f_new) and abs(f_new) < best_abs:
                    best_x = x_new
                    best_f = f_new
                    best_abs = abs(f_new)
                    if best_abs <= 1e-8:
                        break
                else:
                    h *= 0.1
                    if h < 1e-15:
                        break
            except:
                break

    return best_x
# EVOLVE-BLOCK-END
