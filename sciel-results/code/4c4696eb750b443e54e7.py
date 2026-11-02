"""Heilbronn convex regions: maximize min triangle area / convex hull area for 13 points."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import itertools

def _min_triangle_area(pts):
    """Compute minimum triangle area among all triples, vectorized."""
    n = len(pts)
    if n < 3:
        return 0.0
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            v1 = pts[j] - pts[i]
            v2 = pts[j+1:] - pts[i]
            cross = np.abs(v1[0] * v2[:, 1] - v1[1] * v2[:, 0])
            areas = 0.5 * cross
            if len(areas) > 0:
                local_min = np.min(areas)
                if local_min < min_area:
                    min_area = local_min
    return min_area

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """Score = min triangle area / convex hull area."""
    n = len(pts)
    if n < 3:
        return 0.0
    hull_a = _hull_area(pts)
    if hull_a <= 1e-12:
        return 0.0
    return _min_triangle_area(pts) / hull_a

def _neg_score(params):
    pts = params.reshape(-1, 2)
    return -_score(pts)

def _regular_polygon_init(n, center_offset=0.0, jitter=0.0, seed=0):
    rng = np.random.RandomState(seed)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = np.ones(n)
    if jitter > 0:
        r = 1.0 + jitter * rng.randn(n)
    pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
    pts += center_offset
    return pts

def _polygon_with_center_init(n, seed=0):
    """Regular (n-1)-gon with one point at center, then jitter."""
    rng = np.random.RandomState(seed)
    m = n - 1
    angles = np.linspace(0, 2 * np.pi, m, endpoint=False)
    outer = np.column_stack([np.cos(angles), np.sin(angles)])
    center = np.array([[0.0, 0.0]])
    pts = np.vstack([outer, center])
    # Small jitter
    pts += 0.01 * rng.randn(n, 2)
    return pts

def _grid_search_init(n, seed=0):
    """Try various structured configurations."""
    rng = np.random.RandomState(seed)
    configs = []
    
    # Regular n-gon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    configs.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # (n-1)-gon + center
    m = n - 1
    angles_m = np.linspace(0, 2 * np.pi, m, endpoint=False)
    outer = np.column_stack([np.cos(angles_m), np.sin(angles_m)])
    configs.append(np.vstack([outer, [[0.0, 0.0]]]))
    
    # Two concentric rings
    n1 = n // 2
    n2 = n - n1
    ang1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
    ang2 = np.linspace(0, 2 * np.pi, n2, endpoint=False) + np.pi / n2
    ring1 = np.column_stack([0.6 * np.cos(ang1), 0.6 * np.sin(ang1)])
    ring2 = np.column_stack([1.0 * np.cos(ang2), 1.0 * np.sin(ang2)])
    configs.append(np.vstack([ring1, ring2]))
    
    # Jittered versions
    for cfg in list(configs):
        for jit in [0.01, 0.03, 0.05]:
            configs.append(cfg + jit * rng.randn(n, 2))
    
    return configs

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    best_score = 0.0
    best_pts = np.zeros((N, 2))
    rng = np.random.RandomState(42)
    
    # Generate diverse initializations
    inits = []
    
    # Regular polygon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    inits.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # Polygon with center
    inits.append(_polygon_with_center_init(N, seed=0))
    
    # Grid search configs
    inits.extend(_grid_search_init(N, seed=1))
    
    # Random points
    for _ in range(10):
        inits.append(rng.rand(N, 2))
    
    # Jittered regular polygon
    for jitter in [0.01, 0.02, 0.05, 0.1]:
        r = 1.0 + jitter * rng.randn(N)
        inits.append(np.column_stack([r * np.cos(angles), r * np.sin(angles)]))
    
    # Sort inits by initial score (best first) to prioritize good starts
    scored_inits = []
    for init_pts in inits:
        s = _score(init_pts)
        scored_inits.append((s, init_pts))
    scored_inits.sort(key=lambda x: -x[0])
    
    for init_score, init_pts in scored_inits:
        if time.time() > deadline - 8.0:
            break
        
        if init_score > best_score:
            best_score = init_score
            best_pts = init_pts.copy()
        
        init_params = init_pts.flatten()
        
        # Try Nelder-Mead
        try:
            res = minimize(
                _neg_score,
                init_params,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            candidate_pts = res.x.reshape(-1, 2)
            candidate_score = _score(candidate_pts)
            if candidate_score > best_score:
                best_score = candidate_score
                best_pts = candidate_pts.copy()
        except Exception:
            pass
        
        # Try Powell if time permits
        if time.time() > deadline - 15.0:
            continue
        
        try:
            res_powell = minimize(
                _neg_score,
                init_params,
                method='Powell',
                options={'maxiter': 800, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            candidate_pts = res_powell.x.reshape(-1, 2)
            candidate_score = _score(candidate_pts)
            if candidate_score > best_score:
                best_score = candidate_score
                best_pts = candidate_pts.copy()
        except Exception:
            pass
    
    # Additional refinement: try L-BFGS-B on the best solution
    if time.time() < deadline - 10.0:
        try:
            res_lbfgs = minimize(
                _neg_score,
                best_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-8}
            )
            candidate_pts = res_lbfgs.x.reshape(-1, 2)
            candidate_score = _score(candidate_pts)
            if candidate_score > best_score:
                best_score = candidate_score
                best_pts = candidate_pts.copy()
        except Exception:
            pass
    
    # Final polish with Nelder-Mead on best
    if time.time() < deadline - 5.0:
        try:
            res_final = minimize(
                _neg_score,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            candidate_pts = res_final.x.reshape(-1, 2)
            candidate_score = _score(candidate_pts)
            if candidate_score > best_score:
                best_score = candidate_score
                best_pts = candidate_pts.copy()
        except Exception:
            pass
    
    return best_pts
# EVOLVE-BLOCK-END
