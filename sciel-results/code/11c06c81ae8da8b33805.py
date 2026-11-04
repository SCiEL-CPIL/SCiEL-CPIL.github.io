# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import itertools

N = 11

# Vertices of the equilateral triangle in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = 0.5 * np.sqrt(3) / 2

# Precompute indices for all C(11,3) = 165 triples
_IDX_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])

def _bary_to_cart(u, v):
    """
    Convert barycentric coordinates (u, v) to Cartesian.
    w = 1 - u - v
    Point = u*V0 + v*V1 + w*V2
    """
    w = 1.0 - u - v
    x = v * 1.0 + w * 0.5
    y = w * np.sqrt(3) / 2
    return np.column_stack((x, y))

def _all_dets_from_bary(bary):
    """
    Compute all 165 determinants (twice the area in barycentric coords).
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    P = np.column_stack((u, v, w))
    
    i_idx = _IDX_TRIPLES[:, 0]
    j_idx = _IDX_TRIPLES[:, 1]
    k_idx = _IDX_TRIPLES[:, 2]
    
    Pi = P[i_idx]
    Pj = P[j_idx]
    Pk = P[k_idx]
    
    det = (Pi[:, 0] * (Pj[:, 1] * Pk[:, 2] - Pj[:, 2] * Pk[:, 1])
           - Pi[:, 1] * (Pj[:, 0] * Pk[:, 2] - Pj[:, 2] * Pk[:, 0])
           + Pi[:, 2] * (Pj[:, 0] * Pk[:, 1] - Pj[:, 1] * Pk[:, 0]))
           
    return det

def _min_area_ratio_from_bary(bary):
    """
    bary: (N, 2) array of (u, v). w is derived.
    Returns the minimum area ratio of any triangle formed by 3 points.
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    # Check validity
    if np.any(u < -1e-9) or np.any(v < -1e-9) or np.any(w < -1e-9):
        return -1.0
    
    dets = _all_dets_from_bary(bary)
    return np.min(np.abs(dets))

def _project_to_bary_simplex(bary):
    """
    Project points to the valid barycentric simplex.
    """
    u = np.maximum(0, bary[:, 0])
    v = np.maximum(0, bary[:, 1])
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        u[mask] /= s[mask]
        v[mask] /= s[mask]
    return np.column_stack((u, v))

def _objective(x):
    """
    Objective for minimization: -min_area.
    x: (2N,) array.
    """
    bary = x.reshape(N, 2)
    bary = _project_to_bary_simplex(bary)
    return -_min_area_ratio_from_bary(bary)

def _structured_init_fps():
    """
    Farthest Point Sampling on a grid of barycentric coordinates.
    """
    n_grid = 25
    candidates = []
    for i in range(n_grid + 1):
        for j in range(n_grid + 1 - i):
            u = i / n_grid
            v = j / n_grid
            candidates.append([u, v])
    candidates = np.array(candidates)
    
    # Start with center
    selected = [np.array([1/3, 1/3])]
    
    for _ in range(N - 1):
        dists = np.array([np.min(np.linalg.norm(c - selected, axis=1)) for c in candidates])
        max_idx = np.argmax(dists)
        selected.append(candidates[max_idx])
        
    return np.array(selected)

def _random_init(rng):
    """
    Generate random points in the triangle.
    """
    pts = np.zeros((N, 2))
    for i in range(N):
        u = rng.random()
        v = rng.random()
        while u + v > 1.0:
            u = rng.random()
            v = rng.random()
        pts[i, 0] = u
        pts[i, 1] = v
    return pts

def _local_optimize_nm(x_init, maxiter=1000):
    """
    Optimize using Nelder-Mead.
    """
    try:
        res = minimize(
            _objective,
            x_init,
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
        )
        return res.x
    except Exception:
        return x_init

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_x = None
    best_score = -float('inf')
    
    rng = np.random.RandomState(42)
    
    # 1. Generate diverse initial points
    inits = []
    
    # FPS Init
    fps_x = _structured_init_fps().flatten()
    inits.append(fps_x)
    
    # Random Inits
    for _ in range(20):
        pts = _random_init(rng)
        inits.append(pts.flatten())
        
    # Grid-like Inits (spread out)
    # Try placing points near vertices and center
    for _ in range(5):
        pts = np.zeros((N, 2))
        # Corners
        pts[0] = [0.1, 0.1]
        pts[1] = [0.8, 0.1]
        pts[2] = [0.5, 0.7]
        # Center
        pts[3] = [1/3, 1/3]
        # Others random
        for i in range(4, N):
            u = rng.random()
            v = rng.random()
            while u + v > 1.0:
                u = rng.random()
                v = rng.random()
            pts[i] = [u, v]
        inits.append(pts.flatten())

    # 2. Local Optimization from each start
    for init in inits:
        if time.time() > deadline - 20:
            break
            
        # Quick polish with Nelder-Mead
        x_opt = _local_optimize_nm(init, maxiter=500)
        x_opt = _project_to_bary_simplex(x_opt.reshape(N, 2)).flatten()
        
        score = _min_area_ratio_from_bary(x_opt.reshape(N, 2))
        
        if score > best_score:
            best_score = score
            best_x = x_opt.copy()

    if best_x is None:
        best_x = fps_x
        best_score = _min_area_ratio_from_bary(best_x.reshape(N, 2))

    # 3. Iterative Improvement (Hill Climbing with NM)
    # Use the best found so far to generate perturbations
    for _ in range(10):
        if time.time() > deadline - 10:
            break
            
        # Perturb best_x
        noise = rng.normal(0, 0.05, best_x.shape)
        perturbed = _project_to_bary_simplex((best_x + noise).reshape(N, 2)).flatten()
        
        x_opt = _local_optimize_nm(perturbed, maxiter=500)
        x_opt = _project_to_bary_simplex(x_opt.reshape(N, 2)).flatten()
        
        score = _min_area_ratio_from_bary(x_opt.reshape(N, 2))
        
        if score > best_score:
            best_score = score
            best_x = x_opt.copy()

    # 4. Final High-Precision Polish
    if time.time() < deadline - 5:
        # Run a very tight Nelder-Mead on the best solution
        final_x = _local_optimize_nm(best_x, maxiter=2000)
        final_x = _project_to_bary_simplex(final_x.reshape(N, 2)).flatten()
        final_score = _min_area_ratio_from_bary(final_x.reshape(N, 2))
        
        if final_score > best_score:
            best_score = final_score
            best_x = final_x

    cart_points = _bary_to_cart(best_x.reshape(N, 2)[:, 0], best_x.reshape(N, 2)[:, 1])
    
    return cart_points
# EVOLVE-BLOCK-END
