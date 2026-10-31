import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

def _get_triangle_areas(points):
    """
    Vectorized calculation of areas for all triples.
    points: (N, 2) array
    returns: array of shape (num_triangles,)
    """
    n = len(points)
    indices = list(combinations(range(n), 3))
    if not indices:
        return np.array([])
    
    # Extract coordinates for all triangles
    # This is memory efficient enough for N=13
    i_idx = np.array([p[0] for p in indices])
    j_idx = np.array([p[1] for p in indices])
    k_idx = np.array([p[2] for p in indices])
    
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    
    # Area = 0.5 * |(xj-xi)*(yk-yi) - (xk-xi)*(yj-yi)|
    # Vectorized cross product in 2D
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return areas

def _hull_area(points):
    """Calculate convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Calculate the raw score: min_triangle_area / hull_area."""
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    areas = _get_triangle_areas(points)
    if len(areas) == 0:
        return 0.0
    min_area = np.min(areas)
    return min_area / hull_area

def _softmin_score(points, alpha):
    """
    Smooth approximation of the score to allow gradient-free optimization 
    to converge better.
    """
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    
    areas = _get_triangle_areas(points)
    if len(areas) == 0:
        return 0.0
    
    # Softmin: -1/alpha * log(sum(exp(-alpha * areas)))
    # To avoid overflow/underflow, subtract max (which is -min)
    # Actually, for min, we use: -1/alpha * log(sum(exp(-alpha * areas)))
    # This is stable if we handle the exponent carefully.
    # Since areas are positive, -alpha * areas is negative.
    # max_exponent = max(-alpha * areas) = -alpha * min(areas)
    # This is close to 0 if min(areas) is small.
    # Let's just use the formula. If alpha is large, exp(-alpha * max_area) will be 0.
    # exp(-alpha * min_area) will be the dominant term.
    
    # To be safe against underflow for large alpha and small areas:
    # log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
    x = -alpha * areas
    x_max = np.max(x)
    softmin_val = -(x_max + np.log(np.sum(np.exp(x - x_max)))) / alpha
    
    return softmin_val / hull_area

def _neg_softmin_score(params, alpha):
    """Objective function for minimization."""
    points = params.reshape(N, 2)
    return -_softmin_score(points, alpha)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_score = 0.0
    best_points = None
    
    # Generate initial configurations
    inits = []
    
    # 1. Regular polygon
    angles = 2 * np.pi * np.arange(N) / N
    x0 = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(x0)
    
    # 2. Random points
    rng = np.random.RandomState(42)
    for _ in range(20):
        pts = rng.uniform(-1, 1, (N, 2))
        inits.append(pts)
        
    # 3. Points on circle with random radii
    for _ in range(10):
        angles = 2 * np.pi * np.arange(N) / N + rng.uniform(-0.1, 0.1, N)
        radii = 1.0 + rng.uniform(-0.2, 0.2, N)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        inits.append(pts)

    # 4. "Grid" like structures
    # Try to spread points more evenly
    for seed in range(5):
        rng_seed = np.random.RandomState(seed + 100)
        pts = rng_seed.uniform(-1, 1, (N, 2))
        inits.append(pts)

    # Optimization loop
    # We use Powell's method which is robust for small dimensions.
    # We will run a short optimization for each init, then refine the best ones.
    
    candidates = []
    
    for i, init_pts in enumerate(inits):
        if time.time() > deadline - 10:
            break
            
        # Phase 1: Coarse optimization with smaller alpha (smoother)
        alpha = 50
        try:
            res = minimize(
                _neg_softmin_score,
                init_pts.flatten(),
                args=(alpha,),
                method='Powell',
                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6, 'disp': False}
            )
            cand_pts = res.x.reshape(N, 2)
            score = _score(cand_pts)
            if score > 0:
                candidates.append((score, cand_pts))
        except Exception:
            continue

    # Sort candidates by score
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Phase 2: Refine the top candidates with higher alpha (sharper minimum)
    top_k = min(5, len(candidates))
    for i in range(top_k):
        if time.time() > deadline - 5:
            break
        score, pts = candidates[i]
        
        # Refine with higher alpha
        alpha = 200
        try:
            res = minimize(
                _neg_softmin_score,
                pts.flatten(),
                args=(alpha,),
                method='Powell',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8, 'disp': False}
            )
            cand_pts = res.x.reshape(N, 2)
            score = _score(cand_pts)
            if score > best_score:
                best_score = score
                best_points = cand_pts
        except Exception:
            continue

    # If we didn't find anything good, try a few random Nelder-Mead runs
    if best_score < 0.01:
        for _ in range(10):
            if time.time() > deadline - 2:
                break
            rng = np.random.RandomState(np.random.randint(10000))
            init_pts = rng.uniform(-1, 1, (N, 2))
            try:
                res = minimize(
                    lambda x: -_score(x.reshape(N, 2)),
                    init_pts.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6}
                )
                cand_pts = res.x.reshape(N, 2)
                score = _score(cand_pts)
                if score > best_score:
                    best_score = score
                    best_points = cand_pts
            except Exception:
                continue

    if best_points is None:
        # Fallback to regular polygon
        angles = 2 * np.pi * np.arange(N) / N
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_points
