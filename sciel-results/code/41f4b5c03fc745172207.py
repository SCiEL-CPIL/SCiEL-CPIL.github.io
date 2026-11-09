import math
import time
import numpy as np
from scipy.optimize import brentq

def find_root(f, x0):
    """
    Finds a root of f starting near x0.
    Strategy: Global Expanding Bracket + Brent's Method.
    1. Try to bracket the root by expanding interval [x0-d, x0+d].
    2. If bracketed, use Brent's method for robust, fast convergence.
    3. If no bracket found, fall back to a dense local search to find a sign change,
       then use Brent's method.
    4. If still no success, return the point with minimum |f(x)| found.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    x = float(x0)
    best_x = x
    best_fx = f(x)
    best_abs_fx = abs(best_fx)
    eval_count = 1
    
    # If already good enough, return
    if best_abs_fx <= 1e-8:
        return best_x
        
    # Phase 1: Expanding Bracket
    # We try to find a, b such that f(a)*f(b) < 0.
    # Start with a small step and expand exponentially.
    # We limit the number of expansions to avoid infinite loops or huge ranges.
    
    d = 1e-3 * max(1.0, abs(x))
    a = x - d
    b = x + d
    
    fa = f(a)
    fb = f(b)
    eval_count += 2
    
    # Check for NaN/Inf and update best if necessary
    if not (math.isnan(fa) or math.isinf(fa)) and abs(fa) < best_abs_fx:
        best_x = a
        best_fx = fa
        best_abs_fx = abs(fa)
    if not (math.isnan(fb) or math.isinf(fb)) and abs(fb) < best_abs_fx:
        best_x = b
        best_fx = fb
        best_abs_fx = abs(fb)
        
    bracket_found = False
    max_expansions = 30  # Limit expansions to keep total evals low
    
    for i in range(max_expansions):
        if time.time() > deadline:
            break
            
        # Check if we have a sign change
        if not (math.isnan(fa) or math.isinf(fa) or math.isnan(fb) or math.isinf(fb)):
            if fa * fb < 0:
                bracket_found = True
                break
            # If one of them is already very close to 0, we might be done
            if abs(fa) <= 1e-8:
                return a
            if abs(fb) <= 1e-8:
                return b
        
        # Expand the bracket
        # We expand symmetrically around x, but if one side is better, we might bias?
        # For simplicity, symmetric expansion is robust for "local" roots.
        # However, if the root is far, this is efficient.
        d *= 2.0
        
        # If d becomes huge, we might hit domain limits or overflow.
        if d > 1e150:
            break
            
        a = x - d
        b = x + d
        
        # Re-evaluate
        fa = f(a)
        fb = f(b)
        eval_count += 2
        
        if not (math.isnan(fa) or math.isinf(fa)) and abs(fa) < best_abs_fx:
            best_x = a
            best_fx = fa
            best_abs_fx = abs(fa)
        if not (math.isnan(fb) or math.isinf(fb)) and abs(fb) < best_abs_fx:
            best_x = b
            best_fx = fb
            best_abs_fx = abs(fb)

    if bracket_found:
        # Use Brent's method
        try:
            # rtol is important for precision. 1e-12 is usually sufficient for double precision.
            # xtol is for the x-value change.
            root = brentq(f, a, b, maxiter=100, rtol=1e-12, xtol=1e-12)
            fr = f(root)
            # brentq counts evals internally, but we don't have a hook. 
            # However, brentq is very efficient, usually < 20 evals.
            # We assume the total evals are well within 500.
            if abs(fr) < best_abs_fx:
                return root
            else:
                # Sometimes brentq converges to a point where f is not minimal in our scan?
                # Unlikely if bracketed, but check.
                if abs(fr) <= 1e-8:
                    return root
        except Exception:
            pass

    # Phase 2: If no bracket was found, the function might not change sign in the vicinity
    # (e.g., tangent root, or we missed it). 
    # Or it's ill-conditioned such that sign flips are noisy.
    # Let's try a finer local scan to find a sign change or a very small value.
    # We scan a local region around x0.
    
    # Define a local grid around x0
    # Range: x0 +/- 1.0 (or scaled)
    scale = max(1.0, abs(x))
    local_left = x - scale
    local_right = x + scale
    
    # Use a moderate number of points, say 100, to scan.
    # This uses 100 evals. We have budget.
    num_points = 100
    if eval_count + num_points > 450:
        num_points = max(10, 450 - eval_count)
        
    if num_points > 2:
        xs = np.linspace(local_left, local_right, num_points)
        # We already have f(x0). Let's just evaluate the grid.
        # To save evals, we can skip if we already evaluated some, but simple is fine.
        prev_x = None
        prev_f = None
        for xi in xs:
            if time.time() > deadline:
                break
            fi = f(xi)
            eval_count += 1
            if math.isnan(fi) or math.isinf(fi):
                prev_x = xi
                prev_f = fi
                continue
                
            if abs(fi) < best_abs_fx:
                best_x = xi
                best_fx = fi
                best_abs_fx = abs(fi)
                
            if abs(fi) <= 1e-8:
                return xi
                
            if prev_x is not None and prev_f is not None:
                if not (math.isnan(prev_f) or math.isinf(prev_f)):
                    if prev_f * fi < 0:
                        # Found bracket in the scan
                        try:
                            root = brentq(f, prev_x, xi, maxiter=50, rtol=1e-12, xtol=1e-12)
                            fr = f(root)
                            if abs(fr) < best_abs_fx:
                                return root
                            if abs(fr) <= 1e-8:
                                return root
                        except Exception:
                            pass
            prev_x = xi
            prev_f = fi

    return best_x
