"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np
from scipy.optimize import minimize_scalar

def _safe_f(f, x):
    """Safely evaluate f(x), returning inf on failure or non-finite result."""
    try:
        val = f(float(x))
        if not math.isfinite(val):
            return float('inf')
        return float(val)
    except Exception:
        return float('inf')

def _squared_residual(f, x):
    """Objective function for minimization: f(x)^2."""
    fx = _safe_f(f, x)
    if math.isinf(fx):
        return float('inf')
    return fx * fx

def _newton_refine(f, x, max_iter=10):
    """
    Newton's method with finite-difference derivative for final polish.
    Assumes x is already close to the root.
    """
    x = float(x)
    for _ in range(max_iter):
        fx = _safe_f(f, x)
        if not math.isfinite(fx):
            return x
        if abs(fx) <= 1e-12:
            return x
        
        # Central difference for derivative
        h = 1e-5 * max(1.0, abs(x))
        fx_plus = _safe_f(f, x + h)
        fx_minus = _safe_f(f, x - h)
        
        if not math.isfinite(fx_plus) or not math.isfinite(fx_minus):
            break
            
        # If central diff fails, try forward
        if not math.isfinite(fx_plus):
            break
            
        # Approximate derivative
        # (f(x+h) - f(x-h)) / (2h)
        if math.isfinite(fx_minus):
            d = (fx_plus - fx_minus) / (2 * h)
        else:
            d = (fx_plus - fx) / h
            
        if d == 0 or not math.isfinite(d):
            break
            
        x_new = x - fx / d
        if not math.isfinite(x_new):
            break
            
        x = x_new
        
    return x

def find_root(f, x0):
    """
    Global root finder using scipy.optimize.minimize_scalar on f(x)^2.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    x0 = float(x0)
    
    # 1. Initial evaluation at x0
    fx0 = _safe_f(f, x0)
    best_x = x0
    best_val = fx0 * fx0 if math.isfinite(fx0) else float('inf')
    
    # If x0 is already a root, return it
    if math.isfinite(best_val) and best_val <= 1e-16: # 1e-8^2
        # Try to refine just in case, but it's likely good
        x_ref = _newton_refine(f, x0, max_iter=5)
        fx_ref = _safe_f(f, x_ref)
        if math.isfinite(fx_ref) and fx_ref*fx_ref < best_val:
            return x_ref
        return x0

    # 2. Global Minimization Strategy
    # We define a search space around x0. Since we don't know the scale,
    # we try a few different window sizes.
    # The function is a smooth polynomial. The global minimum of f(x)^2 is 0 at the root.
    
    # Strategy:
    # a) Try a wide window to catch the root if it's far from x0.
    # b) Try a narrow window for local convergence if the root is close.
    
    # We'll use 'bounded' method which is robust and derivative-free.
    # However, 'bounded' requires a finite interval.
    
    # Let's try a sequence of intervals centered at x0 with increasing width.
    # If the root is within the interval, minimize_scalar should find it.
    
    # Note: minimize_scalar on a non-convex function (polynomial squared) might find local minima.
    # But for low-degree polynomials, the global min is often the only min or the dominant one.
    # Also, we can use 'nelder-mead' with a large initial simplex if we want unbounded search,
    # but 'bounded' is more predictable for call counts if we pick the right box.
    
    # Let's estimate the scale. We don't know it.
    # Let's try windows: [x0-10, x0+10], [x0-100, x0+100], [x0-1000, x0+1000]
    # And also very small: [x0-0.1, x0+0.1]
    
    # To save calls, we should pick the window that is most likely to contain the root.
    # Usually x0 is a "starting point" which implies proximity.
    # Let's start with a moderate window and expand if we don't find a zero.
    
    # We'll run minimize_scalar with a limited number of function evaluations (maxiter).
    # Each iteration of bounded method (Brent's) typically takes a few function calls.
    # maxiter=50 for bounded method is usually sufficient for high precision.
    
    # However, we must be careful with the number of calls.
    # The score penalizes mean_calls.
    # minimize_scalar will use as many calls as needed to converge to xatol/fatol.
    # We can set xatol and fatol to be relatively loose (e.g., 1e-10) to save calls,
    # then refine with Newton.
    
    windows = [1.0, 10.0, 100.0, 1000.0, 1e4, 1e5]
    
    # Also try asymmetric windows in case x0 is biased
    # But symmetric is a good start.
    
    found_root = False
    
    for w in windows:
        if time.time() > deadline:
            break
            
        a = x0 - w
        b = x0 + w
        
        # Skip if a or b are non-finite
        if not (math.isfinite(a) and math.isfinite(b)):
            continue
            
        # If the interval is too small or degenerate
        if b - a < 1e-10:
            continue
            
        try:
            # Use bounded method
            # maxiter limits the number of iterations, not directly function calls,
            # but it's a good proxy.
            # tol controls the convergence.
            result = minimize_scalar(
                _squared_residual, 
                bounds=(a, b), 
                method='bounded', 
                options={'maxiter': 30, 'xatol': 1e-8, 'fatol': 1e-16}
            )
            
            if result.success and result.fun < best_val:
                best_x = result.x
                best_val = result.fun
                if best_val <= 1e-16:
                    found_root = True
                    break
            elif result.fun < best_val:
                best_x = result.x
                best_val = result.fun
                if best_val <= 1e-16:
                    found_root = True
                    break
                    
        except Exception:
            continue

    # 3. Final Refinement
    # If we found a good candidate, use Newton's method to polish it to |f(x)| <= 1e-8.
    # Newton is very fast (quadratic convergence) and uses 2-3 calls per iteration.
    if math.isfinite(best_val) and best_val < 1e-4: # f(x) < 0.01
        x_ref = _newton_refine(f, best_x, max_iter=10)
        fx_ref = _safe_f(f, x_ref)
        if math.isfinite(fx_ref):
            val_ref = fx_ref * fx_ref
            if val_ref < best_val:
                best_x = x_ref
                best_val = val_ref
            # Check if we met the strict requirement
            if abs(fx_ref) <= 1e-8:
                return x_ref
        
        # If Newton didn't quite get there, check current best
        fx_best = _safe_f(f, best_x)
        if math.isfinite(fx_best) and abs(fx_best) <= 1e-8:
            return best_x

    # 4. Fallback: If we still don't have a root, try a direct scan?
    # Or just return the best we have.
    # If best_val is inf, we haven't found anything finite.
    # Return x0 as last resort? Or best_x?
    # best_x is initialized to x0.
    
    return best_x

# EVOLVE-BLOCK-END
