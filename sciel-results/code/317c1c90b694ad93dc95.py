"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])

# Precompute indices for all triples to speed up area calculation
_IDX_I = None
_IDX_J = None
_IDX_K = None

def _get_indices():
    global _IDX_I, _IDX_J, _IDX_K
    if _IDX_I is None:
        i, j, k = np.indices((N, N, N))
        mask = (i < j) & (j < k)
        _IDX_I = i[mask]
        _IDX_J = j[mask]
        _IDX_K = k[mask]
    return _IDX_I, _IDX_J, _IDX_K

def _min_triangle_area_vectorized(points):
    """Compute the smallest area among all triangles formed by triples of points, vectorized."""
    i_idx, j_idx, k_idx = _get_indices()
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _param_to_points(params):
    """
    Map parameters in [0, 1]^{2N} to points inside the triangle.
    Uses barycentric coordinates.
    """
    s = params[0::2]
    t = params[1::2]
    s = np.clip(s, 0.0, 1.0)
    t = np.clip(t, 0.0, 1.0)
    lam1 = s * (1.0 - t)
    lam2 = s * t
    x = lam1 * V1[0] + lam2 * V2[0]
    y = lam1 * V1[1] + lam2 * V2[1]
    return np.column_stack([x, y])

def _points_to_param(points):
    """
    Map points in the triangle to parameters (s, t).
    Inverse of _param_to_points.
    """
    x = points[:, 0]
    y = points[:, 1]
    lam2 = 2.0 * y / np.sqrt(3)
    lam1 = x - 0.5 * lam2
    s = lam1 + lam2
    t = np.zeros_like(s)
    mask = s > 1e-12
    t[mask] = lam2[mask] / s[mask]
    s = np.clip(s, 0.0, 1.0)
    t = np.clip(t, 0.0, 1.0)
    params = np.zeros(2 * N)
    params[0::2] = s
    params[1::2] = t
    return params

def _objective(params):
    """Objective: negative min triangle area."""
    points = _param_to_points(params)
    return -_min_triangle_area_vectorized(points)

def _is_inside_triangle(point):
    """Check if a point is inside the triangle using barycentric coordinates."""
    x, y = point
    # Barycentric coordinates
    lam2 = 2.0 * y / np.sqrt(3)
    lam1 = x - 0.5 * lam2
    lam0 = 1.0 - lam1 - lam2
    return (lam0 >= -1e-10) and (lam1 >= -1e-10) and (lam2 >= -1e-10)

def _clip_point_to_triangle(point):
    """Clip a point to be inside the triangle."""
    x, y = point
    # If y is too large or negative, clip
    if y < 0:
        y = 0
    if y > np.sqrt(3) / 2:
        y = np.sqrt(3) / 2
    # For a given y, x must be between the left and right edges
    # Left edge: from (0,0) to (0.5, sqrt(3)/2), so x_left = y / sqrt(3)
    # Right edge: from (1,0) to (0.5, sqrt(3)/2), so x_right = 1 - y / sqrt(3)
    x_left = y / np.sqrt(3)
    x_right = 1.0 - y / np.sqrt(3)
    if x < x_left:
        x = x_left
    if x > x_right:
        x = x_right
    return np.array([x, y])

def _generate_fps_initialization(n_points, seed=42):
    """
    Generate an initial configuration using Greedy Farthest Point Sampling.
    This creates a well-spread set of points.
    """
    rng = np.random.RandomState(seed)
    points = []
    
    # Start with a random point near the center
    # Center of triangle is (0.5, sqrt(3)/6)
    center = np.array([0.5, np.sqrt(3) / 6])
    points.append(center.copy())
    
    for _ in range(n_points - 1):
        # Generate many random candidates
        n_candidates = 200
        candidates = []
        for _ in range(n_candidates):
            # Generate random point in bounding box, then clip
            x = rng.uniform(0, 1)
            y = rng.uniform(0, np.sqrt(3) / 2)
            p = _clip_point_to_triangle(np.array([x, y]))
            candidates.append(p)
        
        candidates = np.array(candidates)
        
        # Find the candidate that maximizes the minimum distance to existing points
        best_candidate = None
        best_min_dist = -1
        for c in candidates:
            dists = np.linalg.norm(points - c, axis=1)
            min_dist = np.min(dists)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_candidate = c
        
        points.append(best_candidate)
    
    return np.array(points)

def _generate_structured_initialization(n_points):
    """
    Generate a structured initial configuration based on a grid or known good pattern.
    For 11 points, we can use a pattern inspired by optimal packings.
    """
    points = []
    
    # A simple structured pattern: 3 rows
    # Row 0 (bottom): 4 points
    # Row 1 (middle): 3 points
    # Row 2 (top): 2 points
    # Plus 2 more points distributed
    
    # Actually, let's use a more systematic approach.
    # Divide the triangle into smaller triangles and place points at centroids.
    
    # For now, let's use a combination of vertices, edge midpoints, and centroid
    # V0, V1, V2
    points.append(V0.copy() * 0.1)  # Slightly inside from V0
    points.append(V1.copy() * 0.9)  # Slightly inside from V1
    points.append(V2.copy() * 0.9)  # Slightly inside from V2
    
    # Edge midpoints
    mid01 = (V0 + V1) / 2
    mid12 = (V1 + V2) / 2
    mid02 = (V0 + V2) / 2
    points.append(mid01)
    points.append(mid12)
    points.append(mid02)
    
    # Centroid
    centroid = (V0 + V1 + V2) / 3
    points.append(centroid)
    
    # Add 4 more points in a ring around the centroid
    for i in range(4):
        angle = 2 * np.pi * i / 4 + np.pi / 4
        r = 0.15
        p = centroid + r * np.array([np.cos(angle), np.sin(angle)])
        p = _clip_point_to_triangle(p)
        points.append(p)
    
    return np.array(points[:n_points])

def _nelder_mead_optimize(params, time_budget):
    """Run Nelder-Mead optimization with time budget."""
    start_time = time.time()
    
    def time_left():
        return time_budget - (time.time() - start_time)
    
    try:
        res = minimize(
            _objective, params, method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
        )
        return res.x
    except Exception:
        return params

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    def time_left():
        return deadline - time.time()
    
    best_params = None
    best_score = -np.inf
    
    # Generate multiple diverse initializations
    initializations = []
    
    # 1. FPS initialization with different seeds
    for seed in [42, 123, 456, 789]:
        if time_left() < 5.0:
            break
        pts = _generate_fps_initialization(N, seed)
        params = _points_to_param(pts)
        initializations.append(params)
    
    # 2. Structured initialization
    if time_left() > 5.0:
        pts = _generate_structured_initialization(N)
        params = _points_to_param(pts)
        initializations.append(params)
    
    # 3. Random initializations
    rng = np.random.RandomState(999)
    for _ in range(3):
        if time_left() < 5.0:
            break
        params = rng.uniform(0.1, 0.9, size=2 * N)
        initializations.append(params)
    
    # Evaluate all initializations and keep the best few
    scored_inits = []
    for init in initializations:
        if time_left() < 5.0:
            break
        score = -_objective(init)
        scored_inits.append((score, init))
        if score > best_score:
            best_score = score
            best_params = init.copy()
    
    if best_params is None:
        # Absolute fallback
        best_params = np.zeros(2 * N)
        best_params[0::2] = 0.5
        best_params[1::2] = 0.5
        best_score = -_objective(best_params)
    
    # Sort by score and take top candidates for local optimization
    scored_inits.sort(key=lambda x: -x[0])
    top_candidates = [init for _, init in scored_inits[:5]]
    
    # Ensure best_params is in the list
    if best_params not in top_candidates:
        top_candidates.append(best_params)
    
    # Local optimization with Nelder-Mead for each candidate
    for start_p in top_candidates:
        if time_left() < 15.0:
            break
        
        time_budget = min(time_left() - 5.0, 20.0)
        if time_budget < 5.0:
            break
        
        try:
            res = minimize(
                _objective, start_p, method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            score_res = -res.fun
            if score_res > best_score:
                best_score = score_res
                best_params = res.x
        except Exception:
            pass
    
    points = _param_to_points(best_params)
    return np.array(points)
# EVOLVE-BLOCK-END
