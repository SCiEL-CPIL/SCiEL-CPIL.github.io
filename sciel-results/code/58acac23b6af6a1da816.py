"""Heilbronn triangles: place 11 points in an equilateral triangle to maximize the smallest triangle area."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 11

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = np.sqrt(3) / 4.0  # area of the unit-base equilateral triangle

# Precompute indices for all triples to speed up area calculation
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
    # Area = 0.5 * |cross(pj-pi, pk-pi)|
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

def _objective_bary(x):
    """Objective: negative min triangle area, with points in barycentric coords."""
    bary = x.reshape(N, 2)
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

def _l_bfgs_optimize(bary0, maxiter=500):
    """Run L-BFGS-B optimization in barycentric coordinates."""
    try:
        x0 = bary0.flatten()
        bounds = [(0.0, 1.0)] * (2 * N)
        
        res = minimize(
            _objective_bary, x0, method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-8}
        )
        bary_opt = _clamp_bary(res.x.reshape(N, 2))
        return bary_opt
    except Exception:
        return bary0.copy()

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_points = None
    best_score = -np.inf
    best_bary = None
    
    def time_left():
        return deadline - time.time()
    
    def update_best(bary):
        nonlocal best_points, best_score, best_bary
        points = _bary_to_cart(bary)
        score = _min_triangle_area_vectorized(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()
            best_bary = bary.copy()
            return True
        return False

    # Phase 0: Initialize with a good structured start
    bary_init = _make_initial_bary_grid(0)
    update_best(bary_init)
    
    # Phase 1: Differential Evolution for global search
    # DE is slow but good for finding diverse basins. Run with limited iterations.
    if time_left() > 10.0:
        try:
            bounds = [(0.0, 1.0)] * (2 * N)
            # Use a smaller population and fewer iterations to fit in budget
            de_result = differential_evolution(
                _objective_bary,
                bounds,
                seed=42,
                maxiter=20,
                popsize=10,
                tol=1e-10,
                mutation=(0.5, 1.0),
                recombination=0.7,
                polish=False, # We will polish manually to control time
                updating='deferred'
            )
            if de_result.x is not None:
                bary_de = _clamp_bary(de_result.x.reshape(N, 2))
                update_best(bary_de)
        except Exception:
            pass

    # Phase 2: Multi-restart local optimization with L-BFGS-B
    # L-BFGS-B is much faster than NM, allowing many more restarts.
    rng = np.random.RandomState(123)
    
    max_restarts = 200 # High number, but L-BFGS is fast
    restart = 0
    
    while restart < max_restarts and time_left() > 5.0:
        restart += 1
        
        # Diversify starting points
        if restart % 5 == 0:
            # Random restart
            bary_start = _make_initial_bary_random(restart * 1000)
        elif restart % 2 == 0:
            # Perturb best solution (local search)
            scale = 0.05
            bary_start = best_bary + rng.normal(0, scale, size=best_bary.shape)
            bary_start = _clamp_bary(bary_start)
        else:
            # Structured restart with jitter
            bary_start = _make_initial_bary_grid(restart)
            bary_start = bary_start + rng.normal(0, 0.05, size=bary_start.shape)
            bary_start = _clamp_bary(bary_start)
        
        # Local optimization
        bary_opt = _l_bfgs_optimize(bary_start, maxiter=200)
        update_best(bary_opt)
        
        # If we found a significantly better solution, we might want to refine it immediately
        # But usually, the loop handles this.

    # Phase 3: Fine-tuning from the best solution found
    if time_left() > 3.0 and best_bary is not None:
        # Run several high-precision refinements
        for ft in range(5):
            if time_left() < 2.0:
                break
            
            # Small perturbations to escape potential numerical plateaus
            scale = 0.01 * (ft + 1)
            bary_ft = best_bary + rng.normal(0, scale, size=best_bary.shape)
            bary_ft = _clamp_bary(bary_ft)
            
            # High precision polish
            bary_opt = _l_bfgs_optimize(bary_ft, maxiter=1000)
            update_best(bary_opt)
            
            # Also try polishing the current best directly with tighter tol
            if time_left() > 2.0:
                 bary_opt_direct = _l_bfgs_optimize(best_bary, maxiter=1000)
                 update_best(bary_opt_direct)

    # Final fallback
    if best_points is None:
        bary0 = _make_initial_bary_grid(0)
        best_points = _bary_to_cart(bary0)
    
    return np.array(best_points)
