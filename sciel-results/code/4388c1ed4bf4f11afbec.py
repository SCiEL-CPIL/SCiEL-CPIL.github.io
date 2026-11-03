import numpy as np
import time
from scipy.optimize import minimize

N = 11

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = np.sqrt(3) / 4.0

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
    bary = x.reshape(-1, 2)
    bary = _clamp_bary(bary)
    points = _bary_to_cart(bary)
    return -_min_triangle_area_vectorized(points)

def _make_structured_starts():
    """Generate diverse structured starting points in barycentric coordinates."""
    starts = []
    
    # Start 1: Hexagonal lattice pattern
    bary1 = np.array([
        [0.0, 0.0],    # V0
        [1.0, 0.0],    # V1
        [0.0, 1.0],    # V2
        [0.5, 0.0],    # mid V0-V1
        [0.5, 0.5],    # mid V1-V2
        [0.0, 0.5],    # mid V2-V0
        [1/3, 1/3],    # centroid
        [0.25, 0.0],   # 1/4 V0-V1
        [0.75, 0.0],   # 3/4 V0-V1
        [0.25, 0.25],  # interior
        [0.5, 0.75],   # interior
    ])
    starts.append(_clamp_bary(bary1.copy()))
    
    # Start 2: Concentric hexagons
    bary2 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.333, 0.0],
        [0.667, 0.0],
        [0.0, 0.333],
        [0.0, 0.667],
        [0.333, 0.333],
        [0.667, 0.333],
        [0.333, 0.667],
        [0.5, 0.5],
    ])
    starts.append(_clamp_bary(bary2.copy()))
    
    # Start 3: Symmetric grid with jitter
    rng = np.random.RandomState(123)
    bary3 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.25, 0.0],
        [0.75, 0.0],
        [0.0, 0.25],
        [0.0, 0.75],
        [0.25, 0.25],
        [0.5, 0.25],
        [0.25, 0.5],
        [0.5, 0.5],
    ])
    bary3 += rng.normal(0, 0.02, size=bary3.shape)
    starts.append(_clamp_bary(bary3.copy()))
    
    # Start 4: Random with structured bias
    rng2 = np.random.RandomState(456)
    bary4 = []
    bary4.append([0.0, 0.0])
    bary4.append([1.0, 0.0])
    bary4.append([0.0, 1.0])
    bary4.append([1/3, 1/3])
    for _ in range(7):
        u = rng2.uniform(0.1, 0.9)
        v = rng2.uniform(0.1, 0.9)
        if u + v < 0.85:
            bary4.append([u, v])
        else:
            bary4.append([u * 0.8, v * 0.8])
    starts.append(_clamp_bary(np.array(bary4[:N])))
    
    # Start 5: Optimized hexagonal packing approximation
    bary5 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.3, 0.0],
        [0.7, 0.0],
        [0.0, 0.3],
        [0.0, 0.7],
        [0.3, 0.3],
        [0.7, 0.3],
        [0.3, 0.7],
        [0.5, 0.5],
    ])
    starts.append(_clamp_bary(bary5.copy()))
    
    return starts

def _local_optimize_nm(x0, maxiter=300, xatol=1e-8, fatol=1e-8):
    """Run Nelder-Mead local optimization from x0."""
    try:
        res = minimize(
            _objective_bary, x0, method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': xatol, 'fatol': fatol, 'adaptive': True}
        )
        return res.x
    except Exception:
        return x0

def _local_optimize_lbfgs(x0, maxiter=200, gtol=1e-9):
    """Run L-BFGS-B local optimization from x0."""
    try:
        res = minimize(
            _objective_bary, x0, method='L-BFGS-B',
            bounds=[(0.0, 1.0)] * (2 * N),
            options={'maxiter': maxiter, 'gtol': gtol}
        )
        return res.x
    except Exception:
        return x0

def _refine_with_multiple_methods(bary_start, time_left):
    """Refine a starting point using multiple optimization methods."""
    best_bary = bary_start.copy()
    best_score = _min_triangle_area_vectorized(_bary_to_cart(_clamp_bary(bary_start)))
    
    # Try Nelder-Mead
    if time_left() > 5:
        x_nm = _local_optimize_nm(bary_start.flatten(), maxiter=200)
        bary_nm = _clamp_bary(x_nm.reshape(-1, 2))
        score_nm = _min_triangle_area_vectorized(_bary_to_cart(bary_nm))
        if score_nm > best_score:
            best_score = score_nm
            best_bary = bary_nm.copy()
    
    # Try L-BFGS-B from the best so far
    if time_left() > 5:
        x_lbfgs = _local_optimize_lbfgs(best_bary.flatten(), maxiter=200)
        bary_lbfgs = _clamp_bary(x_lbfgs.reshape(-1, 2))
        score_lbfgs = _min_triangle_area_vectorized(_bary_to_cart(bary_lbfgs))
        if score_lbfgs > best_score:
            best_score = score_lbfgs
            best_bary = bary_lbfgs.copy()
    
    # Fine-tune with Nelder-Mead from the best
    if time_left() > 3:
        x_ft = _local_optimize_nm(best_bary.flatten(), maxiter=300, xatol=1e-10, fatol=1e-10)
        bary_ft = _clamp_bary(x_ft.reshape(-1, 2))
        score_ft = _min_triangle_area_vectorized(_bary_to_cart(bary_ft))
        if score_ft > best_score:
            best_score = score_ft
            best_bary = bary_ft.copy()
    
    return best_bary, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    def time_left():
        return deadline - time.time()
    
    best_points = None
    best_score = -np.inf
    best_bary = None
    
    # Generate structured starting points
    starts = _make_structured_starts()
    
    # Evaluate each starting point
    for bary_start in starts:
        if time_left() < 5:
            break
        bary_clamped = _clamp_bary(bary_start)
        points = _bary_to_cart(bary_clamped)
        score = _min_triangle_area_vectorized(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()
            best_bary = bary_clamped.copy()
    
    # Multi-start optimization with structured starts
    rng = np.random.RandomState(42)
    max_restarts = 200
    restart = 0
    
    while restart < max_restarts and time_left() > 10.0:
        restart += 1
        
        # Choose starting point
        if restart % 5 == 0:
            # Random restart
            bary_start = []
            for _ in range(N):
                u = rng.uniform(0.05, 0.95)
                v = rng.uniform(0.05, 0.95)
                if u + v > 0.95:
                    scale = 0.95 / (u + v)
                    u *= scale
                    v *= scale
                bary_start.append([u, v])
            bary_start = np.array(bary_start)
        elif restart % 3 == 0:
            # Perturb best found so far
            bary_start = best_bary + rng.normal(0, 0.05, size=best_bary.shape)
        else:
            # Use one of the structured starts with slight perturbation
            base = starts[restart % len(starts)]
            bary_start = base + rng.normal(0, 0.03, size=base.shape)
        
        bary_start = _clamp_bary(bary_start)
        
        # Refine
        bary_opt, score_opt = _refine_with_multiple_methods(bary_start, time_left)
        
        if score_opt > best_score:
            best_score = score_opt
            best_points = _bary_to_cart(bary_opt).copy()
            best_bary = bary_opt.copy()
    
    # If no valid point was found, return a fallback
    if best_points is None:
        rng = np.random.RandomState(7)
        points = []
        while len(points) < N:
            u, v = rng.random(), rng.random()
            if u + v < 0.98 and u > 0.01 and v > 0.01:
                points.append([u + 0.5 * v, (3 ** 0.5) / 2 * v])
        best_points = np.array(points)
    
    return best_points
