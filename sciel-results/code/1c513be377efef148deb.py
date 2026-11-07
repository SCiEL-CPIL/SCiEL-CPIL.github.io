import math
import time
import numpy as np
import scipy.optimize

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find a root of f starting near x0, returning x with |f(x)| <= 1e-8.
    Uses Halley's method for fast local convergence, falling back to Brent's method.
    """
    TOL = 1e-8
    DEADLINE = 110.0
    START_TIME = time.time()
    
    def is_finite(x):
        return math.isfinite(x)

    def check_deadline():
        return (time.time() - START_TIME) < DEADLINE

    # 1. Initial check
    try:
        fx0 = f(x0)
    except Exception:
        return x0
        
    if not is_finite(fx0):
        return x0

    if abs(fx0) <= TOL:
        return x0

    best_x = x0
    best_f_abs = abs(fx0)
    
    eval_cache = {x0: fx0}
    
    def f_cached(x):
        """Evaluate f(x) with caching."""
        if x in eval_cache:
            return eval_cache[x]
        try:
            val = f(x)
            if is_finite(val):
                eval_cache[x] = val
                return val
        except Exception:
            pass
        return float('nan')

    def try_halley(x_start, max_iter=20):
        """
        Attempt to find a root starting from x_start using Halley's method.
        Returns (success, root)
        """
        x = x_start
        fx = f_cached(x)
        
        if not is_finite(fx):
            return False, None
            
        if abs(fx) <= TOL:
            return True, x

        # Adaptive step size for finite differences
        # Scale h based on x magnitude to avoid numerical issues
        h = 1e-5 * max(1.0, abs(x))
        
        for i in range(max_iter):
            if not check_deadline():
                return False, best_x
                
            # Central differences for f' and f''
            # f(x-h), f(x+h)
            x_h = x + h
            x_mh = x - h
            
            fx_h = f_cached(x_h)
            fx_mh = f_cached(x_mh)
            
            if not (is_finite(fx_h) and is_finite(fx_mh)):
                return False, best_x
                
            # f'(x) ~ (f(x+h) - f(x-h)) / (2h)
            # f''(x) ~ (f(x+h) - 2f(x) + f(x-h)) / h^2
            fp = (fx_h - fx_mh) / (2 * h)
            fpp = (fx_h - 2 * fx + fx_mh) / (h * h)
            
            # Halley's step: x_new = x - 2*f*f' / (2*f'^2 - f*f'')
            denom = 2 * fp * fp - fx * fpp
            
            if abs(denom) < 1e-15:
                # Denominator too small, fallback to Newton or fail
                # Try Newton step
                if abs(fp) < 1e-15:
                    return False, best_x
                x_new = x - fx / fp
            else:
                x_new = x - (2 * fx * fp) / denom
                
            if not is_finite(x_new):
                return False, best_x
                
            # Safeguard: If step is too large, reduce it
            # Or check if f(x_new) is better
            fx_new = f_cached(x_new)
            
            if not is_finite(fx_new):
                return False, best_x
                
            if abs(fx_new) < abs(fx):
                x = x_new
                fx = fx_new
                if abs(fx) <= TOL:
                    return True, x
                # Adjust h for next iteration? Keep constant or shrink slightly
                # Halley's is cubic, so h can stay similar
            else:
                # If f increased, the step might have been too large or we jumped over the root
                # Try a smaller step or break
                # For now, break and let fallback handle it
                return False, best_x
                
        return False, best_x

    # 2. Primary Strategy: Halley's Method
    # We have fx0. Let's try Halley from x0.
    success, root = try_halley(x0)
    if success:
        return root

    # 3. Secondary Strategy: Bracketing + Brent
    # If Halley failed, we need to find a bracket.
    # Use a geometric expansion search.
    
    # Determine initial step size
    # If x0 is large, step should be large
    step = 1.0
    if abs(x0) > 1.0:
        step = abs(x0) * 0.1 + 1.0
    else:
        step = 1.0
        
    a = x0
    fa = fx0
    
    found_bracket = False
    # Search in both directions
    for direction in [1, -1]:
        if not check_deadline():
            break
            
        b = x0
        fb = fx0
        current_step = step
        
        for i in range(50):
            if not check_deadline():
                break
                
            b = x0 + direction * current_step
            fb = f_cached(b)
            
            if not is_finite(fb):
                break
                
            if abs(fb) < best_f_abs:
                best_f_abs = abs(fb)
                best_x = b
                
            if fa * fb < 0:
                found_bracket = True
                # Use Brent's method
                try:
                    # Ensure a < b for brentq
                    if a < b:
                        root = scipy.optimize.brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
                    else:
                        root = scipy.optimize.brentq(f, b, a, xtol=1e-12, rtol=1e-12, maxiter=100)
                    
                    if is_finite(root):
                        f_root = f_cached(root)
                        if is_finite(f_root) and abs(f_root) <= TOL:
                            return root
                        # Even if not strictly <= TOL, it's likely the best we can do
                        # But the contract says |f(x)| <= 1e-8 is required for success.
                        # Brent with xtol=1e-12 usually achieves this for smooth funcs.
                        return root
                except Exception:
                    pass
                break
                
            # Move a to b for next step in this direction
            # This creates a chain of brackets
            a = b
            fa = fb
            current_step *= 2.0
            
            if current_step > 1e12:
                break
                
    # 4. Fallback: If no bracket found, try polynomial interpolation on cached points
    # This is the logic from the original program, kept as a last resort
    if len(eval_cache) >= 3:
        xs = list(eval_cache.keys())
        ys = [eval_cache[x] for x in xs]
        
        # Try fitting a polynomial of degree min(len(xs)-1, 5)
        deg = min(len(xs) - 1, 5)
        try:
            coeffs = np.polyfit(np.array(xs), np.array(ys), deg)
            roots = np.roots(coeffs)
            
            valid_roots = []
            for r in roots:
                if abs(r.imag) < 1e-9:
                    r_real = r.real
                    if is_finite(r_real):
                        fr = f_cached(r_real)
                        if is_finite(fr) and abs(fr) <= TOL:
                            valid_roots.append(r_real)
                            
            if valid_roots:
                # Return the one closest to x0
                return min(valid_roots, key=lambda r: abs(r - x0))
        except Exception:
            pass

    # 5. Final Polish with Halley/Newton on best_x
    if is_finite(best_x) and best_f_abs < 10.0:
        success, root = try_halley(best_x, max_iter=10)
        if success:
            return root
            
    return best_x
# EVOLVE-BLOCK-END
