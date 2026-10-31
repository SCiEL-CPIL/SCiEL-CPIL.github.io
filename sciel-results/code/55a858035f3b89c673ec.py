import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
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

def _neg_score(params):
    """Objective function for minimization (maximizing score)."""
    points = params.reshape(N, 2)
    s = _score(points)
    return -s

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_score = 0.0
    best_points = None
    
    # 1. Try Differential Evolution for global search
    # Bounds: [-1, 1] for x and y
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    try:
        # Use a smaller population and fewer iterations to fit in time budget
        # popsize=10, maxiter=20 is a quick probe
        de_result = differential_evolution(
            _neg_score, 
            bounds, 
            seed=42, 
            maxiter=30, 
            popsize=10,
            tol=1e-4,
            polish=False
        )
        cand_pts = de_result.x.reshape(N, 2)
        score = _score(cand_pts)
        if score > best_score:
            best_score = score
            best_points = cand_pts
    except Exception:
        pass

    # 2. Local Refinement with Nelder-Mead on the best candidate
    if best_points is not None:
        try:
            res = minimize(
                _neg_score,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            cand_pts = res.x.reshape(N, 2)
            score = _score(cand_pts)
            if score > best_score:
                best_score = score
                best_points = cand_pts
        except Exception:
            pass

    # 3. If DE failed or score is low, try multiple random restarts with Nelder-Mead
    if best_score < 0.05:
        rng = np.random.RandomState(42)
        for i in range(20):
            if time.time() > deadline - 5:
                break
            
            # Random init
            init_pts = rng.uniform(-1, 1, (N, 2))
            
            try:
                res = minimize(
                    _neg_score,
                    init_pts.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
                )
                cand_pts = res.x.reshape(N, 2)
                score = _score(cand_pts)
                if score > best_score:
                    best_score = score
                    best_points = cand_pts
            except Exception:
                continue

    # 4. Fallback to regular polygon if everything failed
    if best_points is None:
        angles = 2 * np.pi * np.arange(N) / N
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_points
