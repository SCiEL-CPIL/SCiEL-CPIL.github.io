"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from scipy.special import logsumexp
import itertools

def _triangle_areas(pts):
    """Compute all triangle areas for N points using vectorized cross products."""
    n = len(pts)
    # Compute all pairwise differences
    # For each triple (i, j, k), area = 0.5 * |cross(pts[j]-pts[i], pts[k]-pts[i])|
    # Vectorized approach:
    # Create arrays of all pairs
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    # Filter to i < j < k
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_flat = i_idx[mask]
    j_flat = j_idx[mask]
    k_flat = k_idx[mask]
    
    pi = pts[i_flat]
    pj = pts[j_flat]
    pk = pts[k_flat]
    
    # Cross product in 2D: (pj - pi) x (pk - pi) = (pj_x - pi_x)*(pk_y - pi_y) - (pj_y - pi_y)*(pk_x - pi_x)
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return areas

def _score_points(pts):
    """Compute min triangle area / convex hull area."""
    n = len(pts)
    if n < 3:
        return 0.0
    
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume  # In 2D, volume is area
        if hull_area <= 0:
            return 0.0
    except Exception:
        return 0.0
    
    areas = _triangle_areas(pts)
    if len(areas) == 0:
        return 0.0
    
    min_area = np.min(areas)
    return min_area / hull_area

def _smooth_min_triangle(pts, temperature=0.01):
    """
    Compute a smooth surrogate for the minimum triangle area using log-sum-exp.
    Returns -softmin(areas) so that minimizing this maximizes the softmin.
    """
    areas = _triangle_areas(pts)
    if len(areas) == 0:
        return 0.0
    
    # Softmin: -log(sum(exp(-areas/temperature))) * temperature
    # This is a smooth approximation of min(areas)
    neg_areas = -areas / temperature
    # Use logsumexp for numerical stability
    log_sum = logsumexp(neg_areas)
    softmin = -temperature * log_sum
    
    return softmin

def _smooth_score(pts, temperature=0.01):
    """
    Compute smooth surrogate for min triangle area / hull area.
    We want to maximize this, so we return negative for minimization.
    """
    n = len(pts)
    if n < 3:
        return 0.0
    
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area <= 0:
            return 0.0
    except Exception:
        return 0.0
    
    softmin = _smooth_min_triangle(pts, temperature)
    return -softmin / hull_area

def _neg_smooth_score(pts_flat, temperature=0.01):
    """Objective function for optimizer: negative smooth score."""
    pts = pts_flat.reshape(-1, 2)
    return _smooth_score(pts, temperature)

def _neg_exact_score(pts_flat):
    """Objective function for exact minimization."""
    pts = pts_flat.reshape(-1, 2)
    return -_score_points(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    best_pts = None
    best_score = -np.inf
    
    def update_best(pts):
        nonlocal best_pts, best_score
        try:
            score = _score_points(pts)
            if score > best_score:
                best_score = score
                best_pts = pts.copy()
        except Exception:
            pass
    
    # Initial random points
    np.random.seed(42)
    pts = np.random.rand(N, 2)
    update_best(pts)
    
    # Multi-start optimization with smooth surrogate
    n_restarts = 30
    for restart in range(n_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Initialize with some structure
        if restart == 0:
            # Start with regular polygon-like distribution
            angles = np.linspace(0, 2*np.pi, N, endpoint=False)
            radii = np.ones(N)
            pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        elif restart == 1:
            # Two concentric rings
            outer_angles = np.linspace(0, 2*np.pi, 7, endpoint=False)
            outer_pts = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
            inner_angles = np.linspace(0, 2*np.pi, 6, endpoint=False) + np.pi/6
            inner_pts = 0.5 * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
            pts = np.vstack([outer_pts, inner_pts])
        else:
            # Random initialization
            pts = np.random.rand(N, 2)
        
        update_best(pts)
        
        # Optimize with smooth surrogate
        x0 = pts.flatten()
        
        # Use Powell method for derivative-free optimization
        try:
            result = minimize(
                _neg_smooth_score,
                x0,
                args=(0.05,),
                method='Powell',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            if result.fun < 0:
                optimized_pts = result.x.reshape(-1, 2)
                update_best(optimized_pts)
                
                # Refine with smaller temperature
                if time.time() < deadline - 5.0:
                    result2 = minimize(
                        _neg_smooth_score,
                        optimized_pts.flatten(),
                        args=(0.01,),
                        method='Powell',
                        options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12}
                    )
                    if result2.fun < 0:
                        refined_pts = result2.x.reshape(-1, 2)
                        update_best(refined_pts)
                        
                        # Final polish with exact objective
                        if time.time() < deadline - 2.0:
                            result3 = minimize(
                                _neg_exact_score,
                                refined_pts.flatten(),
                                method='Powell',
                                options={'maxiter': 200, 'xatol': 1e-12, 'fatol': 1e-14}
                            )
                            if result3.fun < 0:
                                final_pts = result3.x.reshape(-1, 2)
                                update_best(final_pts)
        except Exception:
            pass
    
    # Return best result found
    if best_pts is not None:
        return best_pts
    else:
        # Fallback: return random points
        return np.random.rand(N, 2)

# EVOLVE-BLOCK-END
