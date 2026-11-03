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

def _objective_cart(x):
    """Objective in Cartesian coordinates: negative min triangle area."""
    points = x.reshape(N, 2)
    # Check if all points are inside the triangle using barycentric coords
    bary = _cart_to_bary(points)
    # If any point is outside, add a large penalty
    if np.any(bary[:, 0] < -1e-6) or np.any(bary[:, 1] < -1e-6) or np.any(bary[:, 0] + bary[:, 1] > 1.0 + 1e-6):
        # Count violations
        viol = np.sum((bary[:, 0] < -1e-6).astype(float) + (bary[:, 1] < -1e-6).astype(float) + (bary[:, 0] + bary[:, 1] > 1.0 + 1e-6).astype(float))
        return 100.0 + viol  # large positive penalty
    return -_min_triangle_area_vectorized(points)

def _clamp_cart(points):
    """Clamp Cartesian points to be inside the triangle."""
    bary = _cart_to_bary(points)
    bary = _clamp_bary(bary)
    return _bary_to_cart(bary)

def _make_initial_cart_grid(seed):
    """Generate a structured grid-based initial Cartesian configuration."""
    rng = np.random.RandomState(seed)
    
    # Key structural points in Cartesian
    cart_list = [
        [0.0, 0.0],           # V0
        [1.0, 0.0],           # V1
        [0.5, np.sqrt(3)/2],  # V2
        [0.5, 0.0],           # midpoint V0-V1
        [0.75, np.sqrt(3)/4], # midpoint V1-V2
        [0.25, np.sqrt(3)/4], # midpoint V2-V0
        [0.5, np.sqrt(3)/6],  # centroid
    ]
    
    # Add more interior points
    additional = [
        [0.25, 0.0],
        [0.75, 0.0],
        [0.375, np.sqrt(3)/8],
        [0.625, np.sqrt(3)/8],
        [0.5, np.sqrt(3)/3],
        [0.125, np.sqrt(3)/8],
        [0.375, np.sqrt(3)/4],
    ]
    
    for pt in additional:
        if len(cart_list) < N:
            # Check if inside triangle
            b = _cart_to_bary(np.array([pt]))
            if b[0, 0] >= -1e-9 and b[0, 1] >= -1e-9 and b[0, 0] + b[0, 1] <= 1.0 + 1e-9:
                cart_list.append(pt)
    
    while len(cart_list) < N:
        u = rng.uniform(0.05, 0.95)
        v = rng.uniform(0.05, 0.95)
        if u + v < 0.9:
            bary_pt = np.array([[u, v]])
            cart_pt = _bary_to_cart(bary_pt)[0]
            cart_list.append(cart_pt.tolist())
    
    return np.array(cart_list[:N])

def _make_initial_cart_random(seed):
    """Generate random initial Cartesian coordinates inside the triangle."""
    rng = np.random.RandomState(seed)
    cart_list = []
    for _ in range(1000):
        u = rng.uniform(0.01, 0.99)
        v = rng.uniform(0.01, 0.99)
        if u + v < 0.98:
            bary_pt = np.array([[u, v]])
            cart_pt = _bary_to_cart(bary_pt)[0]
            cart_list.append(cart_pt.tolist())
            if len(cart_list) == N:
                break
    while len(cart_list) < N:
        u = rng.uniform(0.05, 0.8)
        v = rng.uniform(0.05, 0.8)
        if u + v < 0.9:
            bary_pt = np.array([[u, v]])
            cart_pt = _bary_to_cart(bary_pt)[0]
            cart_list.append(cart_pt.tolist())
    return np.array(cart_list[:N])

def _nelder_mead_optimize(points0, maxiter=500):
    """Run Nelder-Mead optimization in Cartesian coordinates."""
    try:
        x0 = points0.flatten()
        res = minimize(
            _objective_cart, x0, method='Nelder-Mead',
            options={'maxiter': maxiter, 'fatol': 1e-10, 'xatol': 1e-10}
        )
        points_opt = _clamp_cart(res.x.reshape(N, 2))
        return points_opt
    except Exception:
        return points0.copy()

def _l_bfgs_optimize(points0, maxiter=300):
    """Run L-BFGS-B optimization in barycentric coordinates."""
    try:
        bary = _cart_to_bary(points0)
        x0 = bary.flatten()
        bounds = [(0.0, 1.0)] * (2 * N)
        res = minimize(
            _objective_bary, x0, method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter, 'gtol': 1e-10}
        )
        bary_opt = _clamp_bary(res.x.reshape(N, 2))
        return _bary_to_cart(bary_opt)
    except Exception:
        return points0.copy()

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    best_points = None
    best_score = -np.inf
    
    def time_left():
        return deadline - time.time()
    
    # Phase 1: Start with structured grid initialization
    cart_init = _make_initial_cart_grid(0)
    score_init = _min_triangle_area_vectorized(cart_init)
    best_score = score_init
    best_points = cart_init.copy()
    
    # Phase 2: Nelder-Mead optimization from grid start
    if time_left() > 5.0:
        points_nm = _nelder_mead_optimize(cart_init, maxiter=500)
        score_nm = _min_triangle_area_vectorized(points_nm)
        if score_nm > best_score:
            best_score = score_nm
            best_points = points_nm.copy()
    
    # Phase 3: L-BFGS-B from best
    if time_left() > 5.0:
        points_lb = _l_bfgs_optimize(best_points, maxiter=300)
        score_lb = _min_triangle_area_vectorized(points_lb)
        if score_lb > best_score:
            best_score = score_lb
            best_points = points_lb.copy()
    
    # Phase 4: Multi-restart with diverse initializations
    rng = np.random.RandomState(42)
    restart = 0
    max_restarts = 200
    
    while restart < max_restarts and time_left() > 3.0:
        restart += 1
        
        if restart % 5 == 0:
            # Random restart
            cart_start = _make_initial_cart_random(restart * 7919)
        elif restart % 3 == 0:
            # Perturb best with moderate scale
            pert = rng.normal(0, 0.08, size=(N, 2))
            cart_start = _clamp_cart(best_points + pert)
        else:
            # Perturb best with smaller scale
            pert = rng.normal(0, 0.03, size=(N, 2))
            cart_start = _clamp_cart(best_points + pert)
        
        # Alternate between Nelder-Mead and L-BFGS-B
        if restart % 2 == 0:
            points_opt = _nelder_mead_optimize(cart_start, maxiter=300)
        else:
            points_opt = _l_bfgs_optimize(cart_start, maxiter=200)
        
        score_opt = _min_triangle_area_vectorized(points_opt)
        
        if score_opt > best_score:
            best_score = score_opt
            best_points = points_opt.copy()
    
    # Phase 5: Fine-tuning
    if time_left() > 3.0:
        for ft in range(10):
            if time_left() < 2.0:
                break
            pert = rng.normal(0, 0.01, size=(N, 2))
            cart_ft = _clamp_cart(best_points + pert)
            points_opt = _nelder_mead_optimize(cart_ft, maxiter=500)
            score_opt = _min_triangle_area_vectorized(points_opt)
            if score_opt > best_score:
                best_score = score_opt
                best_points = points_opt.copy()
    
    # Final fallback
    if best_points is None:
        best_points = _make_initial_cart_grid(0)
    
    return np.array(best_points)

# EVOLVE-BLOCK-START
# (The construct function above is the main entry point)
# EVOLVE-BLOCK-END
