import numpy as np
import time
import itertools
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 16
D = 2

def _pairwise_dists_sq(pts):
    """Compute squared pairwise distances, returning (N, N) matrix."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    return np.sum(diff**2, axis=2)

def _score(pts):
    """Calculate (max_dist / min_dist)^2. Returns inf if invalid."""
    d_sq = _pairwise_dists_sq(pts)
    # Exclude diagonal
    np.fill_diagonal(d_sq, np.inf)
    min_d_sq = np.min(d_sq)
    max_d_sq = np.max(d_sq)
    
    if min_d_sq < 1e-12:
        return 1e10
    if max_d_sq < 1e-12:
        return 1e10
        
    return max_d_sq / min_d_sq

def _normalize(pts):
    """Center points at origin and scale so max pairwise distance is 1."""
    pts = pts - np.mean(pts, axis=0)
    d_sq = _pairwise_dists_sq(pts)
    np.fill_diagonal(d_sq, 0)
    d_max = np.sqrt(np.max(d_sq))
    if d_max < 1e-10:
        return pts
    return pts / d_max

def _get_initial_candidates():
    """Generate diverse initial configurations."""
    inits = []
    
    # 1. Hexagonal lattice subset (19 points, remove 3 outer)
    # Triangular lattice basis
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3)/2])
    
    hex_points = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            p = i * v1 + j * v2
            if np.sum(p**2) <= 4.0 + 1e-9:
                hex_points.append(p)
    
    hex_points = np.array(hex_points)
    if len(hex_points) == 19:
        # Identify outer ring (dist^2 == 4)
        d_sqs = np.sum(hex_points**2, axis=1)
        outer_indices = np.where(np.isclose(d_sqs, 4.0))[0]
        
        # Try removing 3 points from outer ring to get 16
        best_combo_score = 1e10
        best_combo_pts = None
        for remove_idx in itertools.combinations(outer_indices, 3):
            keep_idx = [i for i in range(19) if i not in remove_idx]
            pts = hex_points[keep_idx]
            s = _score(pts)
            if s < best_combo_score:
                best_combo_score = s
                best_combo_pts = pts.copy()
        
        if best_combo_pts is not None:
            inits.append(_normalize(best_combo_pts))
            
    # 2. Square grid 4x4
    grid_pts = np.array(list(itertools.product(range(4), repeat=2)), dtype=float)
    inits.append(_normalize(grid_pts))
    
    # 3. Circle
    theta = np.linspace(0, 2*np.pi, N, endpoint=False)
    circ_pts = np.column_stack([np.cos(theta), np.sin(theta)])
    inits.append(_normalize(circ_pts))
    
    # 4. Random
    np.random.seed(42)
    rand_pts = np.random.randn(N, D)
    inits.append(_normalize(rand_pts))
    
    # 5. Two clusters (8 points each)
    cluster1 = np.random.randn(8, D) * 0.5
    cluster2 = np.random.randn(8, D) * 0.5 + np.array([3.0, 0.0])
    two_cluster = np.vstack([cluster1, cluster2])
    inits.append(_normalize(two_cluster))
    
    # 6. Rectangular grid 2x8
    rect_pts = np.array(list(itertools.product(range(2), range(8))), dtype=float)
    inits.append(_normalize(rect_pts))
    
    return inits

def _optimize_from_start(start_pts, deadline):
    """
    Optimize a starting configuration using Powell's method.
    We fix p0=(0,0) and p1=(1,0) to remove translation/rotation/scale DOF.
    """
    if time.time() > deadline:
        return start_pts, _score(start_pts)
        
    # Normalize and align
    pts = start_pts.copy()
    pts = _normalize(pts)
    
    # Find min distance pair to fix
    d_sq = _pairwise_dists_sq(pts)
    np.fill_diagonal(d_sq, np.inf)
    min_idx = np.argmin(d_sq)
    p, q = np.unravel_index(min_idx, (N, N))
    
    if p > q:
        p, q = q, p
        
    p0 = pts[p]
    p1 = pts[q]
    
    v = p1 - p0
    norm_v = np.linalg.norm(v)
    
    if norm_v < 1e-8:
        return pts, _score(pts)
        
    # Rotation matrix to align v with x-axis
    cos_t = v[0] / norm_v
    sin_t = v[1] / norm_v
    R_mat = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
    
    # Transform all points
    transformed = (pts - p0) @ R_mat.T / norm_v
    
    # Reorder points so that p is at index 0 and q is at index 1
    perm = list(range(N))
    perm[0] = p
    perm[1] = q
    remaining = [idx for idx in range(N) if idx != p and idx != q]
    perm[2:] = remaining
    
    transformed = transformed[perm]
    
    # init_params are the coordinates of points 2..15
    init_params = transformed[2:].flatten()
    
    # Define objective function
    def objective(params):
        if time.time() > deadline:
            return 1e10
        pts_opt = np.zeros((N, 2))
        pts_opt[0] = [0.0, 0.0]
        pts_opt[1] = [1.0, 0.0]
        pts_opt[2:] = params.reshape(-1, 2)
        
        # Check for coincident points
        d_sq_opt = _pairwise_dists_sq(pts_opt)
        np.fill_diagonal(d_sq_opt, np.inf)
        if np.min(d_sq_opt) < 1e-10:
            return 1e10
            
        return _score(pts_opt)
    
    best_score = _score(transformed)
    best_params = init_params.copy()
    
    # Use Powell's method for robustness against non-smoothness of min/max
    try:
        res = minimize(objective, init_params, method='Powell', 
                       options={'maxiter': 100, 'maxfev': 1000, 'disp': False})
        
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x
    except Exception:
        pass
        
    # Reconstruct points
    final_pts = np.zeros((N, 2))
    final_pts[0] = [0.0, 0.0]
    final_pts[1] = [1.0, 0.0]
    final_pts[2:] = best_params.reshape(-1, 2)
    
    # Final check and normalization for output
    final_score = _score(final_pts)
    if final_score > best_score * 1.01 or final_score > 1e5:
        # Fallback to original transformed points if optimization failed badly
        final_pts = transformed
        final_score = _score(final_pts)
        
    return final_pts, final_score

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    best_points = None
    best_score = 1e10
    
    inits = _get_initial_candidates()
    
    # Optimize each initial candidate
    for init in inits:
        if time.time() > deadline - 5:
            break
            
        current_points, current_score = _optimize_from_start(init, deadline)
        
        if current_score < best_score:
            best_score = current_score
            best_points = current_points.copy()
            
    # If optimization didn't yield good results, fall back to best raw init
    if best_points is None or best_score > 100:
        for init in inits:
            s = _score(init)
            if s < best_score:
                best_score = s
                best_points = init.copy()
                
    if best_points is None:
        # Absolute fallback
        grid_pts = np.array(list(itertools.product(range(4), repeat=2)), dtype=float)
        best_points = _normalize(grid_pts)
        
    return best_points
