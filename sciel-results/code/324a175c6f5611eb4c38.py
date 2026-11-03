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

def _objective_cart(x):
    """Objective in Cartesian coordinates: negative min triangle area with penalty for outside points."""
    points = x.reshape(N, 2)
    bary = _cart_to_bary(points)
    # Penalty for points outside the triangle
    outside = (bary[:, 0] < 0) | (bary[:, 1] < 0) | (bary[:, 0] + bary[:, 1] > 1.0)
    if np.any(outside):
        penalty = 10.0 * np.sum(np.maximum(-bary[:, 0], 0) + np.maximum(-bary[:, 1], 0) + np.maximum(bary[:, 0] + bary[:, 1] - 1.0, 0))
        return 100.0 + penalty
    return -_min_triangle_area_vectorized(points)

def _clamp_cart(points):
    """Clamp Cartesian points to be inside the triangle."""
    bary = _cart_to_bary(points)
    bary = _clamp_bary(bary)
    return _bary_to_cart(bary)

def _structured_grid_init(jitter_scale=0.0):
    """
    Generate a structured grid-based initial configuration with optional jitter.
    Uses a hexagonal/triangular lattice pattern for 11 points in the equilateral triangle.
    
    The base pattern places points at:
    - 3 vertices
    - 3 edge midpoints  
    - 1 centroid
    - 4 additional interior points in a symmetric pattern
    
    jitter_scale: if > 0, adds random jitter with this standard deviation
    """
    # Base structured positions in barycentric coordinates
    # Using a systematic pattern that covers the triangle well
    base_bary = np.array([
        [0.0, 0.0],       # V0
        [1.0, 0.0],       # V1
        [0.0, 1.0],       # V2
        [0.5, 0.0],       # midpoint V0-V1
        [0.5, 0.5],       # midpoint V1-V2
        [0.0, 0.5],       # midpoint V2-V0
        [1/3, 1/3],       # centroid
        [0.25, 0.0],      # quarter V0-V1
        [0.75, 0.0],      # three-quarter V0-V1
        [0.25, 0.25],     # interior point 1
        [0.5, 0.25],      # interior point 2
    ])
    
    bary = base_bary.copy()
    
    if jitter_scale > 0:
        rng = np.random.RandomState(42)
        jitter = rng.normal(0, jitter_scale, size=bary.shape)
        bary = bary + jitter
    
    # Clamp to valid region
    bary = _clamp_bary(bary)
    
    return _bary_to_cart(bary)

def _nelder_mead_optimize(points0, maxiter=200, initial_simplex_size=0.1):
    """
    Run Nelder-Mead optimization in Cartesian coordinates.
    Uses initial_simplex_size to control the starting simplex scale.
    """
    try:
        x0 = points0.flatten()
        
        # Create initial simplex with controlled size
        x0_list = [x0.copy()]
        for i in range(2 * N):
            xi = x0.copy()
            if xi[i] != 0:
                xi[i] *= (1 + initial_simplex_size)
            else:
                xi[i] += initial_simplex_size
            x0_list.append(xi)
        
        # Use scipy's Nelder-Mead with the constructed simplex
        res = minimize(
            _objective_cart, x0, method='Nelder-Mead',
            options={
                'maxiter': maxiter, 
                'fatol': 1e-12, 
                'xatol': 1e-12,
                'adaptive': True
            }
        )
        points_opt = _clamp_cart(res.x.reshape(N, 2))
        return points_opt
    except Exception:
        return points0.copy()

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    best_points = None
    best_score = -np.inf
    
    def time_left():
        return deadline - time.time()
    
    # Phase 1: Structured grid initialization with no jitter
    cart_init = _structured_grid_init(jitter_scale=0.0)
    score_init = _min_triangle_area_vectorized(cart_init)
    best_score = score_init
    best_points = cart_init.copy()
    
    # Phase 2: Nelder-Mead from structured init
    if time_left() > 5.0:
        points_nm = _nelder_mead_optimize(cart_init, maxiter=300, initial_simplex_size=0.05)
        score_nm = _min_triangle_area_vectorized(points_nm)
        if score_nm > best_score:
            best_score = score_nm
            best_points = points_nm.copy()
    
    # Phase 3: Multi-start with diverse structured initializations
    # Use different jitter patterns to explore the space
    rng = np.random.RandomState(123)
    
    # Define a set of structured starting points with varying jitter
    jitter_configs = [
        (0.02, "small uniform"),
        (0.05, "medium uniform"),
        (0.10, "large uniform"),
        (0.03, "small asymmetric"),
        (0.07, "medium asymmetric"),
    ]
    
    restart_count = 0
    max_restarts = 150
    
    while restart_count < max_restarts and time_left() > 4.0:
        restart_count += 1
        
        # Choose initialization strategy based on restart count
        if restart_count <= len(jitter_configs):
            # Use structured grid with increasing jitter
            jitter_scale = jitter_configs[restart_count - 1][0]
            cart_start = _structured_grid_init(jitter_scale=jitter_scale)
        elif restart_count <= 2 * len(jitter_configs):
            # Perturb the best solution with small noise
            pert = rng.normal(0, 0.02, size=(N, 2))
            cart_start = _clamp_cart(best_points + pert)
        else:
            # Mix: 50% perturb best, 50% structured with random jitter
            if rng.random() < 0.5:
                pert = rng.normal(0, 0.03, size=(N, 2))
                cart_start = _clamp_cart(best_points + pert)
            else:
                jitter_scale = rng.uniform(0.02, 0.15)
                cart_start = _structured_grid_init(jitter_scale=jitter_scale)
        
        # Nelder-Mead optimization
        points_opt = _nelder_mead_optimize(cart_start, maxiter=200, initial_simplex_size=0.03)
        score_opt = _min_triangle_area_vectorized(points_opt)
        
        if score_opt > best_score:
            best_score = score_opt
            best_points = points_opt.copy()
    
    # Phase 4: Fine-tuning with very small perturbations
    if time_left() > 3.0:
        for ft in range(20):
            if time_left() < 2.0:
                break
            pert = rng.normal(0, 0.005, size=(N, 2))
            cart_ft = _clamp_cart(best_points + pert)
            points_opt = _nelder_mead_optimize(cart_ft, maxiter=300, initial_simplex_size=0.01)
            score_opt = _min_triangle_area_vectorized(points_opt)
            if score_opt > best_score:
                best_score = score_opt
                best_points = points_opt.copy()
    
    # Final fallback
    if best_points is None:
        best_points = _structured_grid_init(jitter_scale=0.0)
    
    return np.array(best_points)

# EVOLVE-BLOCK-START
# (The construct function above is the main entry point)
# EVOLVE-BLOCK-END
