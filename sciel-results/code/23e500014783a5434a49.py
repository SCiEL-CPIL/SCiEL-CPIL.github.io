# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize_scalar

def find_root(f, x0):
    """
    Find a root of f starting near x0 by minimizing f(x)^2 using SciPy's minimize_scalar
    over a set of exponentially expanding intervals.
    """
    x0 = float(x0)
    call_count = 0
    call_limit = 480  # Stay well under 500
    
    def counted_f(x):
        nonlocal call_count
        if call_count >= call_limit:
            raise RuntimeError("Call limit exceeded")
        call_count += 1
        val = f(x)
        if not np.isfinite(val):
            return float('inf')
        return val

    # Check x0 first
    try:
        fx0 = counted_f(x0)
    except RuntimeError:
        return x0 # Should not happen, but safe
        
    if abs(fx0) <= 1e-8:
        return x0
        
    best_x = x0
    best_abs = abs(fx0)

    # Define a set of half-widths for intervals [x0 - w, x0 + w]
    # We want to cover a wide range but start local.
    # Logarithmic spacing.
    widths = []
    # Small widths to catch roots very close to x0
    for exp in range(-6, 0):
        widths.append(10.0 ** exp)
    # Medium and large widths
    for exp in range(0, 7):
        widths.append(10.0 ** exp)
        
    # Sort widths: try small ones first? Or a mix?
    # If the root is close, small intervals converge faster (fewer function evals for minimize_scalar? 
    # Actually, minimize_scalar complexity depends on interval width relative to precision.
    # Smaller intervals -> fewer iterations needed for same xatol.
    # So trying small intervals first is efficient if root is close.
    
    # However, if root is far, we waste calls on small intervals.
    # Let's interleave or just go small to large.
    # Given 480 calls, and minimize_scalar taking ~10-20 calls per successful run,
    # we can afford ~20-40 runs.
    # We have ~12 widths. This is reasonable.

    for w in widths:
        if call_count > call_limit - 20:
            break
            
        a = x0 - w
        b = x0 + w
        
        # Ensure interval is valid (a < b)
        if a >= b:
            continue
            
        try:
            # Define objective function
            def obj(x):
                val = counted_f(x)
                return val * val
            
            res = minimize_scalar(
                obj, 
                bounds=(a, b), 
                method='bounded', 
                options={'xatol': 1e-12, 'maxiter': 50}
            )
            
            x_candidate = res.x
            if np.isfinite(x_candidate):
                fx_candidate = abs(counted_f(x_candidate))
                
                if fx_candidate < best_abs:
                    best_abs = fx_candidate
                    best_x = x_candidate
                    
                    if best_abs <= 1e-8:
                        return best_x
                        
        except (RuntimeError, Exception):
            continue

    # If we haven't found a root, return the best candidate found
    return best_x
# EVOLVE-BLOCK-END
