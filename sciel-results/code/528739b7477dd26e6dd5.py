"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import math

def _compute_c4_exact(roots):
    """Compute the C4 upper bound (1/sqrt(norm_sq)) using exact moment calculation.
    norm_sq = ||P||_2^2 with weight e^-x on [0, inf).
    P(x) = prod (x - r_i)^2.
    <x^i, x^j> = (i+j)!.
    """
    roots = np.asarray(roots, dtype=float)
    if np.any(roots <= 0) or np.any(np.diff(roots) <= 0):
        return 1e10
    
    # Compute coefficients of P(x) = prod (x - r_i)^2
    # Start with 1
    coeffs = np.array([1.0])
    for r in roots:
        # (x - r)^2 = x^2 - 2rx + r^2
        factor = np.array([1.0, -2.0*r, r*r])
        coeffs = np.convolve(coeffs, factor)
        
    # coeffs[k] is coefficient of x^k
    # norm_sq = sum_{i=0}^{24} sum_{j=0}^{24} c_i c_j (i+j)!
    n = len(coeffs)
    # Precompute factorials
    max_fact = 2 * (n - 1)
    fact = np.array([math.factorial(k) for k in range(max_fact + 1)], dtype=float)
    
    # Efficient calculation using matrix multiplication or loops
    # norm_sq = coeffs @ (fact_matrix) @ coeffs
    # fact_matrix[i, j] = fact[i+j]
    # Since n=25, this is small.
    norm_sq = 0.0
    for i in range(n):
        ci = coeffs[i]
        if ci == 0: continue
        for j in range(n):
            cj = coeffs[j]
            if cj == 0: continue
            norm_sq += ci * cj * fact[i+j]
            
    if norm_sq <= 0:
        return 1e10
        
    return 1.0 / np.sqrt(norm_sq)

def _objective(params):
    """Objective function for optimization.
    params: log-roots.
    """
    roots = np.exp(params)
    # Check ordering and positivity
    if np.any(roots <= 0) or np.any(np.diff(roots) <= 0):
        return 1e10
    if np.any(roots > 1e6) or np.any(roots < 1e-6):
        return 1e10
    return _compute_c4_exact(roots)

def construct():
    """Construct 12 positive double-root locations minimizing C4."""
    start_time = time.time()
    deadline = start_time + 600.0
    safety_margin = 30.0
    
    best_score = 1e10
    best_roots = None
    
    # Initial guess: geometric progression
    seed_roots = np.array([4.0 * (i + 1) for i in range(12)])
    seed_log = np.log(seed_roots)
    
    np.random.seed(42)
    
    # 1. Try Differential Evolution for global search
    # Bounds for log-roots: [log(0.1), log(1000)] approx [-2.3, 6.9]
    # We want them to be ordered, but DE works on unconstrained box.
    # We will penalize non-ordered roots in _objective.
    bounds = [(-2.0, 8.0)] * 12
    
    # Run DE with limited iterations to fit in time
    # DE is stochastic, so set seed for reproducibility
    de_result = differential_evolution(
        _objective,
        bounds,
        seed=42,
        maxiter=100,
        tol=1e-8,
        polish=True
    )
    
    if de_result.fun < best_score:
        best_score = de_result.fun
        best_roots = np.exp(de_result.x)
        
    # Check time
    if time.time() > deadline - safety_margin:
        if best_roots is not None and np.all(best_roots > 0) and np.all(np.diff(best_roots) > 0):
            return list(best_roots)
        return list(seed_roots)
        
    # 2. Refine with L-BFGS-B from DE result and other starts
    starts = [de_result.x.copy()]
    # Add some random starts
    for _ in range(20):
        starts.append(np.random.uniform(-2.0, 8.0, size=12))
    # Add seed
    starts.append(seed_log.copy())
    
    for x0 in starts:
        if time.time() > deadline - safety_margin:
            break
        try:
            res = minimize(
                _objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_roots = np.exp(res.x)
        except Exception:
            continue
            
    if best_roots is not None and np.all(best_roots > 0) and np.all(np.diff(best_roots) > 0):
        return list(best_roots)
    else:
        return list(seed_roots)
# EVOLVE-BLOCK-END
