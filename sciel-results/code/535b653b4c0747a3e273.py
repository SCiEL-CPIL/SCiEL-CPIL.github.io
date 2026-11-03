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
    u = points[:, 0] - 0.5 * points[:, 1]
    v = 2.0 * points[:, 1] / np.sqrt(3)
    return np.column_stack([u, v])

def _clamp_bary(bary):
    """Clamp barycentric coordinates to the valid region."""
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

def _make_hex_initial():
    """Create a hexagonal packing initial guess.
    
    For 11 points: 3 corners + 3 edge midpoints + 1 center + 4 inner ring
    """
    # Corners
    pts = [
        [0, 0],       # V0
        [1, 0],       # V1
        [0, 1],       # V2
    ]
    # Edge midpoints
    pts.extend([
        [0.5, 0],     # midpoint V0-V1
        [0.5, 0.5],   # midpoint V1-V2
        [0, 0.5],     # midpoint V2-V0
    ])
    # Center (centroid)
    pts.append([1/3, 1/3])
    
    # 4 inner ring points at radius ~0.15 from center, at 45, 135, 225, 315 degrees
    # In barycentric coords, we need to convert from cartesian offsets
    cx, cy = 1/3, 1/3
    r = 0.15
    for angle_deg in [45, 135, 225, 315]:
        angle = np.radians(angle_deg)
        # Offset in cartesian space
        dx = r * np.cos(angle)
        dy = r * np.sin(angle)
        # Convert offset to barycentric: dx = du + 0.5*dv, dy = sqrt(3)/2*dv
        # dv = 2*dy/sqrt(3), du = dx - 0.5*dv
        dv = 2.0 * dy / np.sqrt(3)
        du = dx - 0.5 * dv
        pts.append([cx + du, cy + dv])
    
    return np.array(pts[:N])

def _make_grid_initial(seed):
    """Generate initial barycentric coordinates using a grid-like structure."""
    rng = np.random.RandomState(seed)
    bary_list = [
        [0, 0],      # V0
        [1, 0],      # V1
        [0, 1],      # V2
        [0.5, 0],    # midpoint V0-V1
        [0.5, 0.5],  # midpoint V1-V2
        [0, 0.5],    # midpoint V2-V0
        [1/3, 1/3],  # Centroid
    ]
    # Add remaining points randomly but biased towards interior
    while len(bary_list) < N:
        u = rng.uniform(0.1, 0.9)
        v = rng.uniform(0.1, 0.9)
        if u + v < 0.8:
            bary_list.append([u, v])
        else:
            continue
    return np.array(bary_list[:N])

def _make_random_initial(seed):
    """Generate random initial barycentric coordinates."""
    rng = np.random.RandomState(seed)
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
    deadline = start_time + 145.0  # safety margin before 150s kill

    best_points = None
    best_score = -np.inf
    best_bary = None

    def time_left():
        return deadline - time.time()

    # Start with hexagonal packing as initial guess
    bary_init = _make_hex_initial()
    points_init = _bary_to_cart(bary_init)
    best_score = _min_triangle_area_vectorized(points_init)
    best_points = points_init.copy()
    best_bary = bary_init.copy()

    rng = np.random.RandomState(42)
    bounds = [(0.0, 1.0)] * (2 * N)
    
    # Multi-restart optimization with both L-BFGS-B and Nelder-Mead
    n_restarts = 300  # Upper bound, time-limited
    
    for i in range(n_restarts):
        if time_left() < 1.0:
            break
        
        # Choose optimization method and starting point
        if i == 0:
            x0 = best_bary.flatten()
            method = 'Nelder-Mead'
        else:
            # Alternate between different strategies
            if i % 10 == 0:
                # Hexagonal restart
                bary_pert = _make_hex_initial()
                method = 'Nelder-Mead'
            elif i % 10 == 1:
                # Grid restart
                bary_pert = _make_grid_initial(i)
                method = 'L-BFGS-B'
            elif i % 10 == 2:
                # Random restart
                bary_pert = _make_random_initial(i)
                method = 'L-BFGS-B'
            else:
                # Perturb the best solution
                scale = 0.05 * (1.0 - i / n_restarts) + 0.01
                bary_pert = best_bary + rng.normal(0, scale, size=best_bary.shape)
                bary_pert = _clamp_bary(bary_pert)
                # Alternate methods
                method = 'Nelder-Mead' if i % 2 == 0 else 'L-BFGS-B'
            
            x0 = bary_pert.flatten()
        
        try:
            if method == 'Nelder-Mead':
                res = minimize(
                    _objective_bary, x0, method='Nelder-Mead',
                    options={'maxiter': 500, 'fatol': 1e-10, 'xatol': 1e-10}
                )
            else:
                res = minimize(
                    _objective_bary, x0, method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'gtol': 1e-8}
                )
            
            bary_opt = res.x.reshape(-1, 2)
            bary_opt = _clamp_bary(bary_opt)
            points = _bary_to_cart(bary_opt)
            score = _min_triangle_area_vectorized(points)
            
            if score > best_score:
                best_score = score
                best_points = points.copy()
                best_bary = bary_opt.copy()
                
        except Exception:
            pass

    # If we have time left, do a final polish with Nelder-Mead and tighter tolerance
    if time_left() > 5.0 and best_bary is not None:
        try:
            x0 = best_bary.flatten()
            res = minimize(
                _objective_bary, x0, method='Nelder-Mead',
                options={'maxiter': 5000, 'fatol': 1e-12, 'xatol': 1e-12}
            )
            bary_opt = res.x.reshape(-1, 2)
            bary_opt = _clamp_bary(bary_opt)
            points = _bary_to_cart(bary_opt)
            score = _min_triangle_area_vectorized(points)
            
            if score > best_score:
                best_score = score
                best_points = points.copy()
                best_bary = bary_opt.copy()
        except Exception:
            pass

    if best_points is None:
        bary0 = _make_hex_initial()
        best_points = _bary_to_cart(bary0)

    return np.array(best_points)
# EVOLVE-BLOCK-END
