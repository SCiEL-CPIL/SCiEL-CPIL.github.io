"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

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
    # Point = u*V1 + v*V2 (since V0 is origin)
    x = bary[:, 0] * V1[0] + bary[:, 1] * V2[0]
    y = bary[:, 0] * V1[1] + bary[:, 1] * V2[1]
    return np.column_stack([x, y])

def _cart_to_bary(points):
    """Convert cartesian points to barycentric (u, v) where point = u*V1 + v*V2."""
    # Solve: x = u*1 + v*0.5, y = u*0 + v*sqrt(3)/2
    # v = 2*y/sqrt(3), u = x - 0.5*v
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
    bary = x.reshape(-1, 2)
    bary = _clamp_bary(bary)
    points = _bary_to_cart(bary)
    return -_min_triangle_area_vectorized(points)

def _make_initial_bary(seed, strategy='random'):
    """Generate initial barycentric coordinates."""
    rng = np.random.RandomState(seed)
    if strategy == 'grid':
        # Use vertices, edge midpoints, and interior points
        bary_list = [
            [0, 0],      # V0
            [1, 0],      # V1
            [0, 1],      # V2
            [0.5, 0],    # midpoint V0-V1
            [0.5, 0.5],  # midpoint V1-V2
            [0, 0.5],    # midpoint V2-V0
        ]
        # Add interior points
        while len(bary_list) < N:
            u = rng.uniform(0.1, 0.9)
            v = rng.uniform(0.1, 0.9)
            if u + v < 0.8:
                bary_list.append([u, v])
        return np.array(bary_list[:N])
    else:
        # Random spread
        bary_list = []
        for _ in range(200):
            u = rng.uniform(0.05, 0.95)
            v = rng.uniform(0.05, 0.95)
            if u + v < 0.9:
                bary_list.append([u, v])
                if len(bary_list) == N:
                    break
        while len(bary_list) < N:
            u = rng.uniform(0.1, 0.8)
            v = rng.uniform(0.1, 0.8)
            if u + v < 0.85:
                bary_list.append([u, v])
        return np.array(bary_list[:N])

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    best_points = None
    best_score = -np.inf

    def time_left():
        return deadline - time.time()

    bounds = [(0.0, 1.0)] * (2 * N)
    
    # Initialize with a grid-based heuristic to have a decent starting point
    bary_best = _make_initial_bary(0, 'grid')
    points_best = _bary_to_cart(bary_best)
    best_score = _min_triangle_area_vectorized(points_best)
    best_points = points_best.copy()

    rng = np.random.RandomState(42)
    
    # Multi-restart local optimization
    # We use L-BFGS-B which is fast. To handle non-smoothness, we rely on many restarts.
    # We can also try Nelder-Mead if L-BFGS fails, but L-BFGS is usually faster per iteration.
    
    max_restarts = 200 # Upper bound, but time-limited
    
    for i in range(max_restarts):
        if time_left() < 1.0:
            break
        
        # Generate initial guess
        if i == 0:
            x0 = bary_best.flatten()
        else:
            # Perturb the best solution or use random
            if i % 5 == 0:
                # Random restart
                bary_pert = _make_initial_bary(i, 'random')
            else:
                # Perturb best
                scale = 0.1 * (1.0 - i / max_restarts) + 0.01
                bary_pert = bary_best + rng.normal(0, scale, size=bary_best.shape)
                bary_pert = _clamp_bary(bary_pert)
            
            x0 = bary_pert.flatten()
        
        try:
            # Use L-BFGS-B
            res = minimize(
                _objective_bary, x0, method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'gtol': 1e-8}
            )
            
            bary_opt = res.x.reshape(-1, 2)
            bary_opt = _clamp_bary(bary_opt)
            points = _bary_to_cart(bary_opt)
            score = _min_triangle_area_vectorized(points)
            
            if score > best_score:
                best_score = score
                best_points = points.copy()
                bary_best = bary_opt.copy()
                
        except Exception:
            pass

    # Final fallback
    if best_points is None:
        bary0 = _make_initial_bary(0, 'grid')
        best_points = _bary_to_cart(bary0)

    return np.array(best_points)

# EVOLVE-BLOCK-START
# (The construct function above is the main entry point)
# EVOLVE-BLOCK-END
