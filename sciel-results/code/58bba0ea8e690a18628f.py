import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Returns 16 points in 2D that maximize the min/max distance ratio.
    Uses smooth soft-min/soft-max objective with L-BFGS-B, multiple initializations,
    and a final Nelder-Mead polish pass.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    n = 16
    m = n * (n - 1) // 2  # 120 pairwise distances
    
    # Precompute pairwise index pairs for vectorized distance computation
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    def compute_dists(pts):
        """Compute all pairwise distances from points array."""
        pts = pts.reshape((n, 2))
        diff = pts[i_idx] - pts[j_idx]
        return np.sqrt(np.sum(diff ** 2, axis=1))
    
    def soft_min_max(pts, beta=100.0):
        """
        Compute soft min and soft max of pairwise distances.
        Uses log-sum-exp style formulation.
        """
        dists = compute_dists(pts)
        if np.any(dists < 0):
            dists = np.abs(dists)
        
        d_max = np.max(dists)
        if d_max < 1e-15:
            return 0.0, d_max
        
        # Soft min: -logsumexp(-beta * d) / beta
        # Soft max: logsumexp(beta * d) / beta
        scaled_min = -beta * dists
        max_scaled_min = np.max(scaled_min)
        soft_min = -(max_scaled_min + np.log(np.sum(np.exp(scaled_min - max_scaled_min)))) / beta
        
        scaled_max = beta * dists
        max_scaled_max = np.max(scaled_max)
        soft_max = (max_scaled_max + np.log(np.sum(np.exp(scaled_max - max_scaled_max)))) / beta
        
        if soft_max < 1e-15:
            return 0.0, soft_max
        
        return soft_min / soft_max, soft_max
    
    def objective(params):
        """
        Smooth objective: minimize -(soft_min / soft_max)
        """
        ratio, _ = soft_min_max(params)
        return -ratio
    
    def get_score(pts):
        dists = compute_dists(pts)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2
    
    def get_ratio(pts):
        dists = compute_dists(pts)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return d_min / d_max
    
    best_points = None
    best_score = -np.inf
    
    # Define initial configurations
    
    # 1. 4x4 Grid
    x = np.arange(4)
    y = np.arange(4)
    xx, yy = np.meshgrid(x, y)
    grid_pts = np.column_stack((xx.ravel(), yy.ravel()))
    
    # 2. Hexagonal Lattice Subset
    u = np.array([1.0, 0.0])
    v = np.array([0.5, np.sqrt(3)/2])
    points_list = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            p = i * u + j * v
            dist_sq = np.dot(p, p)
            points_list.append((dist_sq, p))
    points_list.sort(key=lambda x: x[0])
    hex_pts = np.array([p[1] for p in points_list[:16]])
    
    # 3. Regular 16-gon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    radius = 1.0
    circle_pts = np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))
    
    # 4. Random points (seeded for reproducibility)
    np.random.seed(42)
    rand_pts = np.random.randn(n, 2) * 2
    
    # 5. Two concentric squares (interlocking)
    sq1 = 1.0
    sq2 = 0.5
    pts_sq = []
    for a in [0, 45, 90, 135]:
        rad = np.radians(a)
        pts_sq.append([sq1 * np.cos(rad), sq1 * np.sin(rad)])
    for a in [22.5, 67.5, 112.5, 157.5]:
        rad = np.radians(a)
        pts_sq.append([sq2 * np.cos(rad), sq2 * np.sin(rad)])
    # Add 8 more points on a larger circle
    for a in np.linspace(0, 360, 8, endpoint=False):
        rad = np.radians(a + 22.5)
        pts_sq.append([1.5 * np.cos(rad), 1.5 * np.sin(rad)])
    interlocking_pts = np.array(pts_sq[:16])
    
    initial_guesses = [grid_pts, hex_pts, circle_pts, rand_pts, interlocking_pts]
    
    # Optimization Loop with L-BFGS-B on smooth objective
    for i, init_pts in enumerate(initial_guesses):
        if time.time() > deadline:
            break
            
        # Normalize initial points to roughly unit scale to help optimizer
        scale = np.max(np.linalg.norm(init_pts, axis=1))
        if scale > 0:
            init_pts_scaled = init_pts / scale
        else:
            init_pts_scaled = init_pts
            
        try:
            # Use L-BFGS-B with smooth soft-min/soft-max objective
            res = minimize(
                objective, 
                init_pts_scaled.flatten(), 
                method='L-BFGS-B', 
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12, 'maxfun': 5000}
            )
            
            if time.time() > deadline:
                if best_points is not None:
                    return best_points
                return res.x.reshape((n, 2))
                
            candidate_points = res.x.reshape((n, 2))
            
            # Calculate score for this candidate
            candidate_score = get_score(candidate_points)
            
            if candidate_score > best_score:
                best_score = candidate_score
                best_points = candidate_points.copy()
                
        except Exception:
            continue
    
    # Final polish with Nelder-Mead on the non-smooth objective for precision
    if best_points is not None and time.time() < deadline - 60:
        try:
            def objective_nm(params):
                pts = params.reshape((n, 2))
                dists = compute_dists(pts)
                d_min = np.min(dists)
                d_max = np.max(dists)
                if d_max < 1e-10:
                    return 0.0
                return -(d_min / d_max)
            
            res_nm = minimize(
                objective_nm,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'fatol': 1e-14, 'xatol': 1e-14, 'adaptive': True}
            )
            
            nm_points = res_nm.x.reshape((n, 2))
            nm_score = get_score(nm_points)
            
            if nm_score > best_score:
                best_score = nm_score
                best_points = nm_points.copy()
                
        except Exception:
            pass
    
    # If no valid points were found (should not happen), return grid
    if best_points is None:
        return grid_pts

    return best_points
