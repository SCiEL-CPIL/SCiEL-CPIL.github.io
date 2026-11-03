import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 11

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = np.sqrt(3) / 4.0

# Precompute indices for all triples
_IDX_TRIPLES = None
def _get_indices():
    global _IDX_TRIPLES
    if _IDX_TRIPLES is None:
        n = N
        _IDX_TRIPLES = np.array([(i, j, k) for i in range(n) for j in range(i+1, n) for k in range(j+1, n)])
    return _IDX_TRIPLES

def _min_triangle_area_vectorized(points):
    """Compute the smallest area among all triangles formed by triples of points, vectorized."""
    idx = _get_indices()
    pi = points[idx[:, 0]]
    pj = points[idx[:, 1]]
    pk = points[idx[:, 2]]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _bary_to_cart(bary):
    """Convert barycentric coords (u, v) to cartesian."""
    x = bary[:, 0] * V1[0] + bary[:, 1] * V2[0]
    y = bary[:, 0] * V1[1] + bary[:, 1] * V2[1]
    return np.column_stack([x, y])

def _cart_to_bary(points):
    """Convert cartesian points to barycentric (u, v)."""
    u = points[:, 0] - 0.5 * points[:, 1]
    v = 2.0 * points[:, 1] / np.sqrt(3)
    return np.column_stack([u, v])

def _clamp_bary(bary):
    """Clamp barycentric coordinates to the valid region u>=0, v>=0, u+v<=1."""
    bary = np.clip(bary, 0.0, None)
    s = bary[:, 0] + bary[:, 1]
    mask = s > 1.0
    if np.any(mask):
        bary[mask] = bary[mask] / s[mask, None]
    return bary

def _objective_bary(x):
    """Objective in barycentric coordinates: negative min triangle area."""
    bary = x.reshape(N, 2)
    bary = _clamp_bary(bary)
    points = _bary_to_cart(bary)
    return -_min_triangle_area_vectorized(points)

def _softmin_objective(x, beta=100.0):
    """Smooth surrogate using soft-min for the min triangle area."""
    bary = x.reshape(N, 2)
    bary = _clamp_bary(bary)
    points = _bary_to_cart(bary)
    idx = _get_indices()
    pi = points[idx[:, 0]]
    pj = points[idx[:, 1]]
    pk = points[idx[:, 2]]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    max_a = np.max(areas)
    log_sum = np.log(np.sum(np.exp(-beta * (areas - max_a))))
    soft_min = max_a - log_sum / beta
    return -soft_min

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Bounds for DE: each of the 22 variables is in [0, 1]
    bounds = [(0, 1)] * (2 * N)
    
    # 1. Global Search using Differential Evolution
    # DE is robust for non-smooth, non-convex problems.
    # maxiter=200, popsize=15 (default) -> ~3000 evals.
    # This should be fast enough.
    try:
        de_result = differential_evolution(
            _objective_bary, 
            bounds, 
            maxiter=300, 
            popsize=15, 
            tol=1e-8,
            seed=42,
            init='latinhypercube'
        )
        best_bary = _clamp_bary(de_result.x.reshape(N, 2))
    except Exception:
        # Fallback if DE fails
        best_bary = np.random.uniform(0, 1, size=(N, 2))
        best_bary = _clamp_bary(best_bary)
    
    # 2. Local Refinement
    # Use Nelder-Mead on the exact objective to polish the DE result.
    # Nelder-Mead handles non-differentiability well.
    try:
        res_nm = minimize(
            _objective_bary, 
            best_bary.flatten(), 
            method='Nelder-Mead',
            options={'maxiter': 500, 'fatol': 1e-12, 'xatol': 1e-12, 'adaptive': True}
        )
        if res_nm.fun < _objective_bary(best_bary.flatten()):
            best_bary = _clamp_bary(res_nm.x.reshape(N, 2))
    except Exception:
        pass

    # 3. Further Refinement with L-BFGS-B on Soft-Min
    # Use a high beta to approximate the min closely while maintaining differentiability.
    try:
        res_lbfgs = minimize(
            lambda x: _softmin_objective(x, beta=200.0), 
            best_bary.flatten(), 
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-8}
        )
        # Check if this improved the EXACT objective
        if -res_lbfgs.fun > -_objective_bary(best_bary.flatten()):
             # The soft-min value is an approximation, so we must verify against exact
             candidate_bary = _clamp_bary(res_lbfgs.x.reshape(N, 2))
             if -_objective_bary(candidate_bary.flatten()) > -_objective_bary(best_bary.flatten()):
                 best_bary = candidate_bary
    except Exception:
        pass

    # 4. Final Time-Boxed Local Search
    # If we have time, do a few more Nelder-Mead runs from slightly perturbed best_bary
    while time.time() < deadline - 5.0:
        perturbed = best_bary + np.random.normal(0, 0.01, size=(N, 2))
        perturbed = _clamp_bary(perturbed)
        try:
            res = minimize(
                _objective_bary, 
                perturbed.flatten(), 
                method='Nelder-Mead',
                options={'maxiter': 200, 'fatol': 1e-10, 'xatol': 1e-10}
            )
            if res.fun < _objective_bary(best_bary.flatten()):
                best_bary = _clamp_bary(res.x.reshape(N, 2))
        except Exception:
            break

    return _bary_to_cart(best_bary)
