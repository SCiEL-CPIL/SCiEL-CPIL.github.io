"""Heilbronn triangles: place 11 points in a unit-area equilateral triangle to maximize the smallest triangle area."""
import numpy as np
from scipy.optimize import minimize
import time

N = 11

# Precompute all C(11,3) = 165 triangle indices
def _precompute_indices():
    indices = []
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                indices.append((i, j, k))
    return np.array(indices, dtype=np.int32)

TRI_INDICES = _precompute_indices()

def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 of the N points.
    points: shape (N, 2)
    Returns: float, the minimum area (not doubled)."""
    p_i = points[TRI_INDICES[:, 0]]
    p_j = points[TRI_INDICES[:, 1]]
    p_k = points[TRI_INDICES[:, 2]]
    # Area = 0.5 * |cross(p_j - p_i, p_k - p_i)|
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _objective(params):
    """Objective to minimize: -log-sum-exp approximation of the min area.
    Uses soft-min with escalating beta for smoother landscape."""
    points = params.reshape(N, 2)
    p_i = points[TRI_INDICES[:, 0]]
    p_j = points[TRI_INDICES[:, 1]]
    p_k = points[TRI_INDICES[:, 2]]
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Soft min: -beta * log(sum(exp(-areas / beta)))
    # As beta -> 0, this approaches -min(areas)
    beta = 0.05
    scaled = -areas / beta
    scaled -= np.max(scaled)  # for numerical stability
    soft_min = -beta * np.log(np.sum(np.exp(scaled)))
    return soft_min

def _random_init_bary(seed):
    """Generate random points in barycentric coordinates, convert to Cartesian."""
    rng = np.random.RandomState(seed)
    # Random points in triangle using barycentric coords
    r1 = rng.random(N)
    r2 = rng.random(N)
    # Ensure sum < 1
    r1 = r1 / (r1 + r2 + 1.0)
    r2 = r2 / (r1 + r2 + 1.0) * (1.0 - r1)
    r3 = 1.0 - r1 - r2
    # Convert to Cartesian: P = r1*A + r2*B + r3*C where A=(0,0), B=(1,0), C=(0.5, sqrt(3)/2)
    x = r2 + 0.5 * r3
    y = (np.sqrt(3) / 2) * r3
    return np.column_stack([x, y])

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s kill
    
    best_score = 0.0
    best_points = None
    
    # Try multiple restarts with different seeds
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        seed = 42 + attempt * 7
        points = _random_init_bary(seed)
        params = points.flatten()
        
        # Two-stage optimization: first with larger beta (smoother), then refine
        # Stage 1: coarse optimization
        res = minimize(
            _objective,
            params,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if time.time() > deadline:
            break
        
        # Stage 2: refine with the result
        res2 = minimize(
            _objective,
            res.x,
            method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        
        final_points = res2.x.reshape(N, 2)
        score = _min_triangle_area(final_points)
        
        if score > best_score:
            best_score = score
            best_points = final_points.copy()
    
    if best_points is None:
        # Fallback: random points
        points = _random_init_bary(0)
        best_points = points
    
    return best_points
