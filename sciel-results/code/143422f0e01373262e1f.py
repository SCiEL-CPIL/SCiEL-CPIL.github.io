"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import brentq, minimize_scalar

def find_root(f, x0):
    """
    Robust root finder using adaptive bracketing and Brent's method.
    1. Check initial point.
    2. Search for a sign-change bracket using exponential and linear expansion.
    3. Use Brent's method (superlinear convergence, global convergence) to refine the root.
    4. If no sign change is found (e.g., even multiplicity), use bounded scalar minimization of |f(x)|.
    """
    x = float(x0)
    fx = f(x)
    
    # Check if initial guess is already a root
    if abs(fx) <= 1e-8:
        return x

    # Strategy 1: Find a bracket [a, b] such that f(a)*f(b) < 0
    # We use a combination of small local steps and exponential global steps.
    # This handles both roots very close to x0 and roots far away.
    
    a = None
    b = None
    fa = None
    fb = None
    
    # Try small perturbations first (local ill-conditioning might mean root is close)
    # Then expand exponentially.
    scales = [1e-8, 1e-6, 1e-4, 1e-2, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    
    for scale in scales:
        # Check left and right neighbors
        x_left = x - scale
        x_right = x + scale
        
        # We need to be careful not to re-evaluate x if scale is 0, but scales are > 0.
        # We already have fx.
        
        # Check interval [x_left, x]
        f_left = f(x_left)
        if abs(f_left) <= 1e-8:
            return x_left
        if fx * f_left < 0:
            a, fa = x_left, f_left
            b, fb = x, fx
            break
            
        # Check interval [x, x_right]
        f_right = f(x_right)
        if abs(f_right) <= 1e-8:
            return x_right
        if fx * f_right < 0:
            a, fa = x, fx
            b, fb = x_right, f_right
            break
            
        # If we didn't find a sign change with x as an endpoint, 
        # check if the sign changed between x_left and x_right (skipping x? No, x is inside).
        # Actually, if f(x_left) and f(x_right) have opposite signs, there is a root in [x_left, x_right].
        # We might have missed it if f(x) was 0 (handled above) or if the root is not in [x_left, x] or [x, x_right] but is in [x_left, x_right] due to oscillation? 
        # No, if f(x_left) * f(x_right) < 0, there is at least one root.
        if f_left * f_right < 0:
            a, fa = x_left, f_left
            b, fb = x_right, f_right
            break

    if a is not None and b is not None:
        # Use Brent's method to find the root in [a, b]
        # brentq is robust and fast.
        try:
            root = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
            # Verify the result
            if abs(f(root)) <= 1e-8:
                return root
            # If brentq converged but precision wasn't enough (unlikely with these settings), 
            # or if it failed, we might need to refine. But usually brentq is good.
            return root
        except Exception:
            pass

    # Strategy 2: If no sign change bracket found, the root might be of even multiplicity
    # or the function is very flat. We try to minimize |f(x)| in a local region.
    # We define a bounded search interval around x0.
    # Start with a small window, expand if the minimum is at the boundary.
    
    window_size = 1e-2
    best_x = x
    best_f = abs(fx)
    
    for _ in range(5):
        a_min = x - window_size
        b_min = x + window_size
        
        # minimize_scalar with method='bounded'
        try:
            res = minimize_scalar(lambda t: abs(f(t)), bounds=(a_min, b_min), method='bounded', options={'xatol': 1e-12})
            if res.success:
                x_candidate = res.x
                f_candidate = abs(f(x_candidate))
                if f_candidate < best_f:
                    best_x = x_candidate
                    best_f = f_candidate
                    if best_f <= 1e-8:
                        return best_x
                # If the minimum is at the boundary, expand the window
                if abs(x_candidate - a_min) < 1e-12 or abs(x_candidate - b_min) < 1e-12:
                    window_size *= 10
                else:
                    # Local minimum found, check if it's good enough
                    if best_f <= 1e-8:
                        return best_x
                    # If not, try a slightly larger window just in case, or accept best
                    # For ill-conditioned problems, we might need to try harder.
                    # But if we found a local min, it's likely the root.
                    break
            else:
                break
        except Exception:
            break
    
    if best_f <= 1e-8:
        return best_x

    # Strategy 3: Last resort - high precision Newton with damping
    # This is similar to the seed but used only as a last resort.
    x_newton = best_x if best_f < abs(fx) else x
    fx_newton = f(x_newton)
    
    for i in range(50):
        if abs(fx_newton) <= 1e-8:
            return x_newton
        h = 1e-7 * max(1.0, abs(x_newton))
        fp = f(x_newton + h)
        fm = f(x_newton - h)
        d = (fp - fm) / (2 * h)
        if d == 0 or not np.isfinite(d):
            break
        step = fx_newton / d
        max_step = 10 * max(1.0, abs(x_newton))
        if abs(step) > max_step:
            step = max_step * np.sign(step)
        x_candidate = x_newton - step
        fx_candidate = f(x_candidate)
        if abs(fx_candidate) < abs(fx_newton):
            x_newton = x_candidate
            fx_newton = fx_candidate
        else:
            x_candidate = x_newton - step / 2
            fx_candidate = f(x_candidate)
            if abs(fx_candidate) < abs(fx_newton):
                x_newton = x_candidate
                fx_newton = fx_candidate
            else:
                break

    if abs(fx_newton) <= 1e-8:
        return x_newton

    return best_x

# EVOLVE-BLOCK-END
