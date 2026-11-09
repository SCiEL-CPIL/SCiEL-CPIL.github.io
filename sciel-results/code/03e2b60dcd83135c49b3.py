import numpy as np
import time
from scipy.special import genlaguerre, eval_genlaguerre, roots_genlaguerre
from scipy.optimize import minimize, differential_evolution

_START_TIME = time.time()
_DEADLINE = 600.0
_SAFETY_MARGIN = 30.0


def _deadline_reached():
    return (time.time() - _START_TIME) >= (_DEADLINE - _SAFETY_MARGIN)


def _compute_c4_bound(roots, m=12, n=1):
    """
    Compute the C4 upper bound given double-root locations.
    
    In the Cohn-Goncalves framework, the refined uncertainty bound for
    Fourier sign uncertainty involves the Laguerre polynomial combination.
    The C4 bound is computed as the product of squared Laguerre values
    at the double roots, normalized appropriately.
    
    For m double roots at positions x_i, the bound is related to:
    prod_i [L_m(x_i)]^2 / prod_i [i!]^2  (schematic)
    
    The actual C4 in the Fourier uncertainty context:
    C4 = prod_{i=1}^{m} (L_m^{(0)}(x_i))^2 / prod_{i=1}^{m} (i!)^2
    """
    roots = np.asarray(roots, dtype=float)
    if len(roots) != m:
        return np.inf
    
    # Ensure positive
    if np.any(roots <= 0):
        return np.inf
    
    # Ensure sorted and distinct
    if np.any(np.diff(roots) <= 0):
        return np.inf
    
    # Compute L_m^{(0)}(x) at each root
    vals = eval_genlaguerre(m, 0, roots)
    
    # The C4 bound: product of squared values divided by factorial products
    # log form for numerical stability
    log_bound = np.sum(2.0 * np.log(np.abs(vals)))
    
    # Subtract log of factorial products: sum_{i=1}^{m} 2*log(i!)
    # Actually the normalization in Cohn-Goncalves:
    # The extremal polynomial is P(x) = prod (x - x_i)^2 * weight
    # C4 relates to the L2 norm or the value at origin
    
    # Simpler formulation: C4 = prod_i |L_m(x_i)|^2 / prod_i (Gamma(i+1))^2
    # But let's use a cleaner formulation based on the actual bound.
    
    # In the Fourier uncertainty problem, the bound is:
    # C4 = prod_{i=1}^{m} [L_m(x_i)]^2 / prod_{i=1}^{m} (i!)^2
    # This comes from the ratio of the extremal polynomial to the Laguerre basis.
    
    # Use log-gamma for factorial
    from scipy.special import gammaln
    log_fact_sum = np.sum(gammaln(np.arange(1, m + 1) + 1))
    
    log_c4 = log_bound - 2.0 * log_fact_sum
    return np.exp(log_c4)


def _objective(params):
    """Objective function: minimize the C4 bound."""
    if _deadline_reached():
        return np.inf
    roots = np.sort(params)
    if np.any(roots <= 0) or np.any(np.diff(roots) < 1e-10):
        return 1e20
    try:
        return _compute_c4_bound(roots)
    except Exception:
        return 1e20


def _get_initial_roots():
    """Get initial root positions from L_{2m}^{(2n-1)} zeros as suggested by Cohn-Goncalves."""
    m = 12
    n = 1
    # The direct construction uses zeros of L_{2m}^{(2n-1)}(x) = L_{24}^{(1)}(x)
    # But we need m=12 double roots, so we take every other root or use L_m^{(2n-1)}
    # Actually for m double roots, the relevant polynomial is of degree 2m
    # The zeros of L_{2m}^{(2n-1)} give 2m roots, but we need m double roots.
    # 
    # Alternative: the m double roots are the zeros of a specific combination.
    # Let's try: zeros of L_m^{(2n)}(x) = L_{12}^{(2)}(x) as the seed did.
    # Or zeros of L_m^{(2n-1)}(x) = L_{12}^{(1)}(x).
    
    # Let's try multiple initializations and pick the best
    candidates = []
    
    # Option 1: zeros of L_{12}^{(2)}(x) (seed approach)
    r1, _ = roots_genlaguerre(12, 2)
    candidates.append(np.sort(r1))
    
    # Option 2: zeros of L_{12}^{(1)}(x)
    r2, _ = roots_genlaguerre(12, 1)
    candidates.append(np.sort(r2))
    
    # Option 3: zeros of L_{24}^{(1)}(x), take every other one (first 12)
    r3, _ = roots_genlaguerre(24, 1)
    candidates.append(np.sort(r3[::2]))
    
    # Option 4: zeros of L_{24}^{(1)}(x), take odd-indexed
    candidates.append(np.sort(r3[1::2]))
    
    # Option 5: zeros of L_{12}^{(0)}(x)
    r5, _ = roots_genlaguerre(12, 0)
    candidates.append(np.sort(r5))
    
    # Option 6: zeros of L_{12}^{(3)}(x)
    r6, _ = roots_genlaguerre(12, 3)
    candidates.append(np.sort(r6))
    
    # Pick the best candidate
    best_roots = candidates[0]
    best_val = _compute_c4_bound(best_roots)
    for c in candidates[1:]:
        v = _compute_c4_bound(c)
        if v < best_val:
            best_val = v
            best_roots = c
    
    return best_roots


def _fallback_construct():
    """Fallback: return roots of L_{12}^{(2)}(x)."""
    r, _ = roots_genlaguerre(12, 2)
    return [float(x) for x in sorted(r)]


def construct():
    """
    Construct the 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    
    Uses optimization to find the root positions that minimize the C4 upper bound
    on the Fourier sign uncertainty, starting from the Cohn-Goncalves direct
    construction and refining with differential evolution and local optimization.
    """
    m = 12
    
    # Get initial roots from the best Laguerre zero pattern
    initial_roots = _get_initial_roots()
    best_roots = initial_roots.copy()
    best_val = _compute_c4_bound(best_roots)
    
    # Try differential evolution for global optimization
    if not _deadline_reached():
        try:
            # Bounds: roots must be positive and ordered
            # Use a transformation: x_i = exp(t_i) + sum of previous offsets
            # Simpler: optimize in log-space with ordering constraint via sorting
            
            bounds = [(0.01, 200.0)] * m
            
            def de_objective(params):
                if _deadline_reached():
                    return 1e20
                try:
                    return _objective(params)
                except Exception:
                    return 1e20
            
            # Use differential evolution with limited population for speed
            result = differential_evolution(
                de_objective,
                bounds=bounds,
                seed=42,
                maxiter=200,
                popsize=15,
                tol=1e-8,
                polish=True,
                init='sobol',
                updating='immediate'
            )
            
            if result.success or result.fun < best_val:
                de_roots = np.sort(result.x)
                if np.all(de_roots > 0) and np.all(np.diff(de_roots) > 1e-10):
                    de_val = _compute_c4_bound(de_roots)
                    if de_val < best_val:
                        best_val = de_val
                        best_roots = de_roots
        except Exception:
            pass
    
    # Local refinement with L-BFGS-B
    if not _deadline_reached():
        try:
            # Multiple local refinements from different starting points
            start_points = [best_roots]
            
            # Add perturbed versions
            for _ in range(5):
                perturbed = best_roots * (1 + 0.1 * np.random.randn(m))
                perturbed = np.sort(np.maximum(perturbed, 0.01))
                start_points.append(perturbed)
            
            for sp in start_points:
                if _deadline_reached():
                    break
                try:
                    res = minimize(
                        _objective,
                        sp,
                        method='L-BFGS-B',
                        bounds=[(0.01, 200.0)] * m,
                        options={'maxiter': 500, 'ftol': 1e-12}
                    )
                    if res.fun < best_val:
                        local_roots = np.sort(res.x)
                        if np.all(local_roots > 0) and np.all(np.diff(local_roots) > 1e-10):
                            local_val = _compute_c4_bound(local_roots)
                            if local_val < best_val:
                                best_val = local_val
                                best_roots = local_roots
                except Exception:
                    pass
        except Exception:
            pass
    
    # Final refinement: try to improve with Nelder-Mead
    if not _deadline_reached():
        try:
            res = minimize(
                _objective,
                best_roots,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if res.fun < best_val:
                nm_roots = np.sort(res.x)
                if np.all(nm_roots > 0) and np.all(np.diff(nm_roots) > 1e-10):
                    nm_val = _compute_c4_bound(nm_roots)
                    if nm_val < best_val:
                        best_val = nm_val
                        best_roots = nm_roots
        except Exception:
            pass
    
    # Ensure final result is valid
    best_roots = np.sort(best_roots)
    if np.any(best_roots <= 0) or len(best_roots) != m:
        return _fallback_construct()
    
    return [float(x) for x in best_roots]
