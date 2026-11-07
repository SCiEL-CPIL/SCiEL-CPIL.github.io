"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize_scalar

def find_root(f, x0):
    """
    Find a root of f starting near x0 by minimizing f(x)^2 over dynamically
    expanded bounded intervals using SciPy's minimize_scalar.
    
    Strategy:
    1. Coarse scan around x0 to locate region of smallest |f(x)|
    2. Use that region as initial bracket for minimize_scalar
    3. Expand bracket if needed
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
    
    # Step 1: Coarse scan to find best starting region
    best_x = x0
    best_abs = abs(counted_f(x0))
    
    if best_abs <= 1e-8:
        return best_x
    
    # Scan at different radii with multiple points per radius
    scan_radii = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 500.0]
    points_per_radius = 8
    
    for r in scan_radii:
        if call_count >= call_limit - 50:
            break
        for i in range(points_per_radius):
            if call_count >= call_limit - 50:
                break
            t = i / (points_per_radius - 1) if points_per_radius > 1 else 0.5
            x_sample = x0 - r + 2 * r * t
            try:
                fx = counted_f(x_sample)
                abs_fx = abs(fx)
                if abs_fx < best_abs:
                    best_abs = abs_fx
                    best_x = x_sample
                    if best_abs <= 1e-8:
                        return best_x
            except RuntimeError:
                break
        if best_abs <= 1e-8:
            return best_x
    
    # Step 2: Use best_x as center for minimize_scalar with expanding brackets
    base_width = max(1e-4, best_abs ** 0.5 * 0.1)
    if not np.isfinite(base_width) or base_width <= 0:
        base_width = 1e-3
    
    widths = [base_width, base_width * 5, base_width * 25, base_width * 125, 
              base_width * 625, base_width * 3125, base_width * 15625]
    
    for w in widths:
        if call_count >= call_limit - 30:
            break
        a = best_x - w
        b = best_x + w
        try:
            res = minimize_scalar(
                lambda x: counted_f(x) ** 2,
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
    
    # Step 3: Broader search if still not found
    if call_count < call_limit - 20:
        for w in [10.0, 100.0, 1000.0]:
            if call_count >= call_limit - 20:
                break
            a = x0 - w
            b = x0 + w
            try:
                res = minimize_scalar(
                    lambda x: counted_f(x) ** 2,
                    bounds=(a, b),
                    method='bounded',
                    options={'xatol': 1e-10, 'maxiter': 30}
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
    
    return best_x
# EVOLVE-BLOCK-END
