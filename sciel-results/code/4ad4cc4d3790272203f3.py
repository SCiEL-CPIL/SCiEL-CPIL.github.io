"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution

def _bary_to_cart(bary):
    """Convert barycentric coordinates (n,3) to Cartesian (n,2).
    Triangle vertices: A=(0,0), B=(1,0), C=(0.5, sqrt(3)/2)
    bary[i] = (lambda_A, lambda_B, lambda_C), sum to 1, all >= 0
    """
    # Normalize to ensure sum=1
    bary = np.clip(bary, 0, None)
    sums = bary.sum(axis=1, keepdims=True)
    sums = np.where(sums < 1e-12, 1.0, sums)
    bary = bary / sums
    # p = lambda_A * A + lambda_B * B + lambda_C * C
    # A=(0,0), B=(1,0), C=(0.5, sqrt(3)/2)
    x = bary[:, 1] + 0.5 * bary[:, 2]
    y = (3 ** 0.5) / 2 * bary[:, 2]
    return np.column_stack([x, y])

def _min_triangle_area_vec(points):
    """Vectorized computation of minimum triangle area among all C(n,3) triples.
    points: (n, 2) array
    Returns: float, the minimum area
    """
    n = len(points)
    if n < 3:
        return 0.0
    # Compute all pairwise cross products for area
    # For each triple (i,j,k): area = 0.5 * |cross(p_j - p_i, p_k - p_i)|
    # Use broadcasting: 
    # i: (C(n,3), 1), j: (C(n,3), 1), k: (C(n,3), 1)
    # But more efficient: compute all (i,j) pairs first, then for each k
    
    # Generate all combinations
    idx = np.array(list(_combinations_range(n, 3)))
    if len(idx) == 0:
        return 0.0
    i, j, k = idx[:, 0], idx[:, 1], idx[:, 2]
    pi = points[i]
    pj = points[j]
    pk = points[k]
    # cross product: (pj - pi) x (pk - pi)
    d1 = pj - pi  # (m, 2)
    d2 = pk - pi  # (m, 2)
    cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    areas = 0.5 * np.abs(cross)
    return float(np.min(areas))

def _combinations_range(n, r):
    """Generate combinations as list of tuples."""
    import itertools
    return itertools.combinations(range(n), r)

def _objective(flat_bary):
    """Objective function for differential evolution.
    flat_bary: (3*n,) array of barycentric coordinates
    Returns: negative of min triangle area (to minimize)
    """
    bary = flat_bary.reshape(N, 3)
    points = _bary_to_cart(bary)
    min_area = _min_triangle_area_vec(points)
    return -min_area

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 10s safety margin before 140s kill
    
    # Bounds: each barycentric coordinate in [0, 1]
    bounds = [(0.0, 1.0)] * (3 * N)
    
    # Use differential evolution for global search
    best_result = None
    best_score = -np.inf
    
    try:
        de_result = differential_evolution(
            _objective,
            bounds=bounds,
            seed=42,
            maxiter=200,
            popsize=15,
            tol=1e-6,
            polish=True,
            init='latinhypercube',
            callback=None
        )
        elapsed = time.time() - start_time
        if elapsed < deadline:
            best_flat = de_result.x
            best_bary = best_flat.reshape(N, 3)
            best_points = _bary_to_cart(best_bary)
            best_score = -de_result.fun
            best_result = best_points
    except Exception as e:
        pass
    
    elapsed = time.time() - start_time
    
    # If we still have time, try a few more restarts with different seeds
    if best_result is None or elapsed < deadline - 5.0:
        import random
        rng = random.Random(123)
        for attempt in range(5):
            if time.time() > deadline - 5.0:
                break
            # Random initialization
            bary_init = np.random.rand(N, 3)
            flat_init = bary_init.flatten()
            
            try:
                from scipy.optimize import minimize
                res = minimize(
                    _objective,
                    flat_init,
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                if -res.fun > best_score:
                    best_score = -res.fun
                    best_result = _bary_to_cart(res.x.reshape(N, 3))
            except Exception:
                pass
    
    if best_result is None:
        # Fallback: simple random points
        rng = np.random.RandomState(7)
        bary = rng.rand(N, 3)
        bary = bary / bary.sum(axis=1, keepdims=True)
        best_result = _bary_to_cart(bary)
    
    return best_result
# EVOLVE-BLOCK-END
