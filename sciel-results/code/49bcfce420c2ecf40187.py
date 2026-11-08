"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14

# EVOLVE-BLOCK-START

def _calc_areas(points):
    """
    Vectorized calculation of all triangle areas and convex hull area.
    Returns:
        min_tri_area: float
        hull_area: float
    """
    n = len(points)
    # Precompute indices for triangles
    # i < j < k
    idx_i, idx_j, idx_k = np.indices((n, n, n))
    # Filter i < j < k
    mask = (idx_i < idx_j) & (idx_j < idx_k)
    i_idx = idx_i[mask]
    j_idx = idx_j[mask]
    k_idx = idx_k[mask]
    
    # Get points
    p_i = points[i_idx]
    p_j = points[j_idx]
    p_k = points[k_idx]
    
    # Vectors
    v1 = p_j - p_i
    v2 = p_k - p_i
    
    # Cross product magnitude (z-component)
    # area = 0.5 * |v1_x * v2_y - v1_y * v2_x|
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_tri_area = np.min(areas)
    
    # Convex Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        # If hull fails (e.g. collinear), return 0 to avoid div by 0
        # But this should be a very bad configuration
        hull_area = 1e-10 
        min_tri_area = 0.0
        
    return min_tri_area, hull_area

def _score(points):
    """Calculate the normalized minimum triangle area."""
    min_tri_area, hull_area = _calc_areas(points)
    if hull_area < 1e-10:
        return 0.0
    return min_tri_area / hull_area

def _neg_score_vec(x):
    """Objective for minimization: negative score."""
    pts = x.reshape(N, 2)
    return -_score(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s limit
    
    # Initial guess: Regular N-gon
    angles = 2 * np.pi * np.arange(N) / N
    points_init = np.column_stack([np.cos(angles), np.sin(angles)])
    
    best_points = points_init.copy()
    best_score = _score(best_points)
    
    # Strategy:
    # 1. Local refinement from regular polygon.
    # 2. If time remains, try a few perturbed restarts.
    # 3. Use Nelder-Mead as it handles non-smooth functions reasonably well.
    
    # Stage 1: Refine the regular polygon
    nm_options = {
        'maxiter': 5000,
        'xatol': 1e-8,
        'fatol': 1e-8,
        'adaptive': True
    }
    
    try:
        nm_result = minimize(
            _neg_score_vec,
            best_points.flatten(),
            method='Nelder-Mead',
            options=nm_options
        )
        candidate = nm_result.x.reshape(N, 2)
        cs = _score(candidate)
        if cs > best_score:
            best_score = cs
            best_points = candidate.copy()
    except Exception:
        pass
    
    # Stage 2: Random restarts with small perturbations
    # The landscape might have local minima near the regular polygon.
    # We try to escape by adding noise and re-optimizing.
    
    rng = np.random.default_rng(42)
    num_restarts = 10
    noise_scale = 0.1
    
    for i in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Generate a perturbed start point
        # Mix of small and medium noise
        scale = noise_scale * (0.5 ** (i // 3)) # Decay noise scale
        noise = rng.normal(0, scale, (N, 2))
        start_pts = best_points + noise
        
        # Optimize
        nm_options_restart = {
            'maxiter': 2000,
            'xatol': 1e-8,
            'fatol': 1e-8,
            'adaptive': True
        }
        
        try:
            nm_result = minimize(
                _neg_score_vec,
                start_pts.flatten(),
                method='Nelder-Mead',
                options=nm_options_restart
            )
            candidate = nm_result.x.reshape(N, 2)
            cs = _score(candidate)
            if cs > best_score:
                best_score = cs
                best_points = candidate.copy()
        except Exception:
            pass
            
    return best_points

# EVOLVE-BLOCK-END
