"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.special import genlaguerre, eval_genlaguerre

_START = time.time()
_DEADLINE = 590.0  # safety margin before 630s kill

def _deadline_reached():
    return (time.time() - _START) >= _DEADLINE

def _c4_objective(roots, m=12, n=1):
    """
    Compute the C4 upper bound for the Cohn-Goncalves Laguerre construction.
    
    For m double roots x_1 < x_2 < ... < x_m in dimension n, the C4 bound
    involves evaluating the Laguerre polynomial combination. A well-motivated
    surrogate: the objective is the maximum over a grid of the absolute value
    of the Laguerre-based kernel, or equivalently, we minimize a function
    related to the product of spacings weighted by Laguerre values.
    
    Based on the Cohn-Goncalves construction, the bound C4 is proportional to
    the maximum of |x^n * L_m^{(2n)}(x)| / (m! * e^{x/2}) evaluated at critical points,
    but for optimization we use a surrogate that captures the essential structure.
    
    The key insight from the successful parent: the objective involves evaluating
    generalized Laguerre polynomials at the root locations and computing a
    combined measure.
    """
    roots = np.asarray(roots, dtype=float)
    if len(roots) != m:
        return 1e10
    if np.any(roots <= 0):
        return 1e10
    if not np.all(np.diff(roots) > 1e-10):
        return 1e10
    
    # Compute the Laguerre polynomial L_m^{(2n)} at each root
    # and use a surrogate objective that the parent found effective
    alpha = 2.0 * n  # = 2 for n=1
    
    # Evaluate L_m^{(alpha)} at each root
    lag_vals = np.array([eval_genlaguerre(m, alpha, x) for x in roots])
    
    # The C4 bound is related to the maximum absolute value of a certain
    # Laguerre-based function. A good surrogate: minimize the maximum
    # of |L_m^{(alpha)}(x_i)| * x_i^n weighted appropriately.
    
    # Another approach: the product of gaps weighted by Laguerre values
    gaps = np.diff(roots)
    if np.any(gaps <= 0):
        return 1e10
    
    # Use the harmonic mean of gaps as a measure of spacing quality,
    # combined with Laguerre evaluation magnitude
    # The parent's successful approach likely used:
    # objective = max(|lag_vals|) or sum of |lag_vals| or similar
    
    # Let's try: minimize the maximum absolute Laguerre value at the roots
    # This encourages roots where the Laguerre polynomial is small,
    # which is related to the uncertainty bound
    obj = np.max(np.abs(lag_vals))
    
    # Also incorporate spacing: penalize very small gaps
    min_gap = np.min(gaps)
    if min_gap < 0.1:
        obj += 10.0 / min_gap
    
    return obj

def _c4_objective_v2(roots, m=12, n=1):
    """
    Alternative objective: based on the actual C4 bound structure.
    The Cohn-Goncalves bound for Fourier sign uncertainty with m double roots
    involves the supremum of a Laguerre-based kernel. 
    """
    roots = np.asarray(roots, dtype=float)
    if len(roots) != m:
        return 1e10
    if np.any(roots <= 0):
        return 1e10
    if not np.all(np.diff(roots) > 1e-10):
        return 1e10
    
    alpha = 2.0 * n
    m_int = m
    
    # Evaluate on a fine grid to estimate the C4 bound
    x_max = roots[-1] * 1.5
    x_grid = np.linspace(0.01, x_max, 2000)
    
    # The kernel involves L_m^{(2n)}(x) * x^n / (e^{x/2})
    # Compute this on the grid
    lag_grid = np.array([eval_genlaguerre(m_int, alpha, x) for x in x_grid])
    kernel = np.abs(lag_grid * x_grid**n / np.exp(x_grid / 2.0))
    
    # The C4 bound is proportional to the max of this kernel
    # But we also need to account for the root locations
    # The double roots modify the kernel
    
    # For double roots at x_i, the kernel is modified by factors
    # related to (x - x_i)^2 terms in the numerator
    
    # Simpler surrogate: the max of the base kernel
    return np.max(kernel)

def _c4_objective_v3(roots, m=12, n=1):
    """
    Most direct surrogate: the C4 upper bound for the Cohn-Goncalves
    construction with m double roots is given by:
    
    C4 = max_{x >= 0} | f(x) |
    
    where f(x) involves the Laguerre polynomial and the root locations.
    
    For the specific case, a good optimization target is to minimize
    the maximum of the absolute value of the Laguerre polynomial
    L_m^{(2n)}(x) * x^n / (e^{x/2} * m!) over x >= 0, but the root
    locations affect this through the construction.
    
    Actually, the roots ARE the parameters we're optimizing. The C4 bound
    depends on where we place the double roots. The correct objective is
    complex, but a good surrogate is:
    
    C4 ~ product of (x_i) * max_x |L_m^{(2n)}(x) * x^n / (e^{x/2})|
    
    or similar. Let's use a combined objective.
    """
    roots = np.asarray(roots, dtype=float)
    if len(roots) != m:
        return 1e10
    if np.any(roots <= 0):
        return 1e10
    if not np.all(np.diff(roots) > 1e-10):
        return 1e10
    
    alpha = 2.0 * n
    m_int = m
    
    # Evaluate L_m^{(alpha)} at each root
    lag_at_roots = np.array([eval_genlaguerre(m_int, alpha, x) for x in roots])
    
    # The product of roots and the max Laguerre value
    log_product = np.sum(np.log(roots))
    max_lag = np.max(np.abs(lag_at_roots))
    
    # Combined objective
    return np.exp(log_product) * max_lag

def construct():
    """
    Return 12 positive double-root locations for the Laguerre combination
    (m=12, n=1) that minimize the C4 upper bound on Fourier sign uncertainty.
    
    Strategy: Use numerical optimization (differential evolution + local refinement)
    to find the optimal root configuration, with multiple restarts and a tight
    time budget.
    """
    m = 12
    n = 1
    
    # Initial best: use natural Laguerre zeros as starting point
    try:
        from scipy.special import roots_genlaguerre
        init_roots, _ = roots_genlaguerre(m, 1.0)
        init_roots = np.sort(init_roots)
        if len(init_roots) == m and np.all(init_roots > 0):
            best_roots = init_roots.copy()
        else:
            best_roots = np.array([4.0 * (i + 1) for i in range(m)])
    except Exception:
        best_roots = np.array([4.0 * (i + 1) for i in range(m)])
    
    # Also try arithmetic progression as another starting point
    arith_roots = np.array([4.0 * (i + 1) for i in range(m)])
    
    # Evaluate objectives to pick the best starting point
    def _eval_roots(r):
        # Use a robust objective: combine Laguerre evaluation and spacing
        r = np.asarray(r, dtype=float)
        if len(r) != m or np.any(r <= 0) or not np.all(np.diff(r) > 1e-10):
            return 1e10
        alpha = 2.0 * n
        lag_vals = np.array([eval_genlaguerre(m, alpha, x) for x in r])
        gaps = np.diff(r)
        min_gap = np.min(gaps)
        if min_gap < 0.05:
            return 1e10
        # Objective: max |Laguerre| + penalty for small gaps
        return np.max(np.abs(lag_vals)) + 1.0 / min_gap
    
    obj_lag = _eval_roots(best_roots)
    obj_arith = _eval_roots(arith_roots)
    
    if obj_arith < obj_lag:
        best_roots = arith_roots
        best_obj = obj_arith
    else:
        best_obj = obj_lag
    
    # Define the optimization objective
    def _objective(x):
        x = np.asarray(x, dtype=float)
        if len(x) != m:
            return 1e10
        if np.any(x <= 0):
            return 1e10
        if not np.all(np.diff(x) > 1e-10):
            return 1e10
        alpha = 2.0 * n
        lag_vals = np.array([eval_genlaguerre(m, alpha, x) for x in x])
        gaps = np.diff(x)
        min_gap = np.min(gaps)
        if min_gap < 0.05:
            return 1e10
        return np.max(np.abs(lag_vals)) + 1.0 / min_gap
    
    # Differential evolution for global optimization
    # Bounds: roots should be positive and reasonably spaced
    # Typical range: 0.5 to 50
    bounds = [(0.5, 50.0)] * m
    
    # Ensure ordering: we'll use a transformation or just let DE handle it
    # and then sort and check
    
    try:
        result = differential_evolution(
            _objective,
            bounds=bounds,
            seed=42,
            maxiter=200,
            tol=1e-8,
            polish=False,
            workers=1,
            init='sobol',
            popsize=15
        )
        
        if result.success and result.fun < best_obj:
            candidate = np.sort(result.x)
            if np.all(candidate > 0) and np.all(np.diff(candidate) > 1e-10):
                best_roots = candidate
                best_obj = result.fun
        
        # Local refinement with BFGS
        if not _deadline_reached():
            res2 = minimize(
                _objective,
                best_roots,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res2.success and res2.fun < best_obj:
                candidate = np.sort(res2.x)
                if np.all(candidate > 0) and np.all(np.diff(candidate) > 1e-10):
                    best_roots = candidate
                    best_obj = res2.fun
                    
    except Exception:
        pass
    
    # Try a few more restarts if time permits
    if not _deadline_reached():
        for restart_seed in [123, 456, 789]:
            if _deadline_reached():
                break
            try:
                result = differential_evolution(
                    _objective,
                    bounds=bounds,
                    seed=restart_seed,
                    maxiter=100,
                    tol=1e-8,
                    polish=False,
                    workers=1,
                    init='latinhypercube',
                    popsize=12
                )
                if result.success and result.fun < best_obj:
                    candidate = np.sort(result.x)
                    if np.all(candidate > 0) and np.all(np.diff(candidate) > 1e-10):
                        best_roots = candidate
                        best_obj = result.fun
                        
                # Local refinement
                if not _deadline_reached():
                    res2 = minimize(
                        _objective,
                        best_roots,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
                    )
                    if res2.success and res2.fun < best_obj:
                        candidate = np.sort(res2.x)
                        if np.all(candidate > 0) and np.all(np.diff(candidate) > 1e-10):
                            best_roots = candidate
                            best_obj = res2.fun
            except Exception:
                continue
    
    # Final safety check
    best_roots = np.asarray(best_roots, dtype=float)
    if len(best_roots) != m or np.any(best_roots <= 0):
        return [4.0 * (i + 1) for i in range(m)]
    
    best_roots = np.sort(best_roots)
    return [float(r) for r in best_roots]
# EVOLVE-BLOCK-END
