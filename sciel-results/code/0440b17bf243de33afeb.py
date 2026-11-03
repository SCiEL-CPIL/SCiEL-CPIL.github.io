"""Heilbronn triangles: place 11 points in an equilateral triangle to maximize the smallest triangle area."""
import time
import numpy as np
from scipy.optimize import minimize
import itertools

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
    """Convert barycentric coords (u, v) with u>=0, v>=0, u+v<=1 to cartesian."""
    x = bary[:, 0] * V1[0] + bary[:, 1] * V2[0]
    y = bary[:, 0] * V1[1] + bary[:, 1] * V2[1]
    return np.column_stack([x, y])

def _cart_to_bary(points):
    """Convert cartesian points to barycentric (u, v) where point = u*V1 + v*V2."""
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

def _softmin_objective(x, beta=80.0):
    """Smooth surrogate for the min triangle area using soft-min."""
    bary = x.reshape(-1, 2)
    bary = _clamp_bary(bary)
    points = _bary_to_cart(bary)
    idx = _get_indices()
    pi = points[idx[:, 0]]
    pj = points[idx[:, 1]]
    pk = points[idx[:, 2]]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    shifted = areas - min_area
    log_sum = np.log(np.sum(np.exp(-beta * shifted))) / beta
    soft_min = min_area - log_sum
    
    return -soft_min

def _objective_bary(x):
    """Objective: negative min triangle area, with points in barycentric coords."""
    bary = x.reshape(-1, 2)
    bary = _clamp_bary(bary)
    points = _bary_to_cart(bary)
    return -_min_triangle_area_vectorized(points)

def _make_initial_bary_grid(seed):
    """Generate a structured grid-based initial barycentric configuration."""
    rng = np.random.RandomState(seed)
    
    # Key structural points: vertices, edge midpoints, centroid
    bary_list = [
        [0.0, 0.0],       # V0
        [1.0, 0.0],       # V1
        [0.0, 1.0],       # V2
        [0.5, 0.0],       # midpoint V0-V1
        [0.5, 0.5],       # midpoint V1-V2
        [0.0, 0.5],       # midpoint V2-V0
        [1.0/3, 1.0/3],   # centroid
    ]
    
    # Add interior points in a regular pattern
    additional = [
        [0.25, 0.0],
        [0.75, 0.0],
        [0.25, 0.25],
        [0.75, 0.25],
        [0.5, 0.75],
        [0.0, 0.25],
        [0.0, 0.75],
    ]
    
    for pt in additional:
        if len(bary_list) < N:
            if pt[0] >= 0 and pt[1] >= 0 and pt[0] + pt[1] <= 1:
                bary_list.append(pt)
    
    while len(bary_list) < N:
        u = rng.uniform(0.1, 0.8)
        v = rng.uniform(0.1, 0.8)
        if u + v < 0.85:
            bary_list.append([u, v])
    
    return np.array(bary_list[:N])

def _make_initial_bary_random(seed):
    """Generate random initial barycentric coordinates."""
    rng = np.random.RandomState(seed)
    bary_list = []
    for _ in range(500):
        u = rng.uniform(0.02, 0.98)
        v = rng.uniform(0.02, 0.98)
        if u + v < 0.96:
            bary_list.append([u, v])
            if len(bary_list) == N:
                break
    while len(bary_list) < N:
        u = rng.uniform(0.05, 0.8)
        v = rng.uniform(0.05, 0.8)
        if u + v < 0.9:
            bary_list.append([u, v])
    return np.array(bary_list[:N])

def _make_initial_bary_farthest(seed):
    """Generate initial points using greedy farthest point sampling in barycentric space."""
    rng = np.random.RandomState(seed)
    
    # Start with vertices
    bary_list = [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ]
    
    for _ in range(N - 3):
        # Try many random candidates and pick the one that maximizes min distance to existing points
        best_candidate = None
        best_min_dist = -1.0
        
        for _ in range(200):
            u = rng.uniform(0.0, 1.0)
            v = rng.uniform(0.0, 1.0)
            if u + v > 1.0:
                continue
            candidate = np.array([u, v])
            
            min_dist = float('inf')
            for existing in bary_list:
                dist = np.linalg.norm(candidate - np.array(existing))
                min_dist = min(min_dist, dist)
            
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_candidate = candidate
        
        if best_candidate is not None:
            bary_list.append(best_candidate)
    
    return np.array(bary_list[:N])

def _nelder_mead_optimize(x0, maxiter=300):
    """
    Custom Nelder-Mead that works on flat arrays and projects to bounds.
    """
    x = x0.copy()
    n = len(x)
    
    # Initialize simplex
    simplex = np.zeros((n + 1, n))
    simplex[0] = x
    for i in range(n):
        dx = x.copy()
        dx[i] += 0.01
        simplex[i + 1] = _clamp_bary(dx.reshape(-1, 2)).flatten()
        
    # Evaluate initial simplex
    fvals = np.array([_objective_bary(s) for s in simplex])
    
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    for _ in range(maxiter):
        # Sort simplex
        order = np.argsort(fvals)
        simplex = simplex[order]
        fvals = fvals[order]
        
        # Check convergence
        if np.max(fvals) - np.min(fvals) < 1e-10:
            break
            
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        centroid = _clamp_bary(centroid.reshape(-1, 2)).flatten()
        
        # Reflection
        xr = centroid + alpha * (centroid - simplex[-1])
        xr = _clamp_bary(xr.reshape(-1, 2)).flatten()
        fr = _objective_bary(xr)
        
        if fr < fvals[0]:
            # Expansion
            xe = centroid + gamma * (xr - centroid)
            xe = _clamp_bary(xe.reshape(-1, 2)).flatten()
            fe = _objective_bary(xe)
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
        elif fr < fvals[-2]:
            # Accept reflection
            simplex[-1] = xr
            fvals[-1] = fr
        else:
            # Contraction
            xc = centroid + rho * (simplex[-1] - centroid)
            xc = _clamp_bary(xc.reshape(-1, 2)).flatten()
            fc = _objective_bary(xc)
            if fc < fvals[-1]:
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    simplex[i] = _clamp_bary(simplex[i].reshape(-1, 2)).flatten()
                    fvals[i] = _objective_bary(simplex[i])
                    
    return simplex[0]

def _local_search(points, max_iter=50, step_size=0.05):
    """Simple local search: try small perturbations and accept if improvement."""
    best_points = points.copy()
    best_score = _min_triangle_area_vectorized(points)
    rng = np.random.RandomState(42)
    
    for _ in range(max_iter):
        # Randomly select a point to perturb
        idx = rng.randint(N)
        delta = rng.normal(0, step_size, size=2)
        
        new_points = best_points.copy()
        new_points[idx] = new_points[idx] + delta
        
        # Clamp to triangle
        bary = _cart_to_bary(new_points)
        bary = _clamp_bary(bary)
        new_points = _bary_to_cart(bary)
        
        new_score = _min_triangle_area_vectorized(new_points)
        if new_score > best_score:
            best_points = new_points
            best_score = new_score
            step_size *= 0.9
        else:
            step_size *= 0.95
    
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_points = None
    best_score = -float('inf')
    best_bary = None
    
    def time_left():
        return deadline - time.time()
    
    # Try multiple initialization strategies
    init_strategies = [
        ("grid", _make_initial_bary_grid(0)),
        ("grid", _make_initial_bary_grid(1)),
        ("grid", _make_initial_bary_grid(2)),
        ("farthest", _make_initial_bary_farthest(0)),
        ("farthest", _make_initial_bary_farthest(1)),
        ("random", _make_initial_bary_random(0)),
        ("random", _make_initial_bary_random(1)),
    ]
    
    for name, bary_init in init_strategies:
        if time_left() < 20.0:
            break
        
        points_init = _bary_to_cart(bary_init)
        score_init = _min_triangle_area_vectorized(points_init)
        
        if score_init > best_score:
            best_score = score_init
            best_points = points_init.copy()
            best_bary = bary_init.copy()
        
        # Try Nelder-Mead optimization
        x0 = bary_init.flatten()
        result = _nelder_mead_optimize(x0, maxiter=200)
        
        if time_left() < 15.0:
            break
        
        optimized_bary = _clamp_bary(result.reshape(-1, 2))
        optimized_points = _bary_to_cart(optimized_bary)
        optimized_score = _min_triangle_area_vectorized(optimized_points)
        
        if optimized_score > best_score:
            best_score = optimized_score
            best_points = optimized_points.copy()
            best_bary = optimized_bary.copy()
        
        # Try local search refinement
        if time_left() > 30.0:
            refined_points, refined_score = _local_search(optimized_points, max_iter=30, step_size=0.03)
            if refined_score > best_score:
                best_score = refined_score
                best_points = refined_points.copy()
                best_bary = _cart_to_bary(refined_points)
    
    # Final refinement on best solution
    if time_left() > 10.0 and best_points is not None:
        final_points, final_score = _local_search(best_points, max_iter=40, step_size=0.02)
        if final_score > best_score:
            best_points = final_points
            best_score = final_score
    
    if best_points is None:
        # Fallback
        bary_init = _make_initial_bary_grid(0)
        best_points = _bary_to_cart(bary_init)
    
    return best_points
