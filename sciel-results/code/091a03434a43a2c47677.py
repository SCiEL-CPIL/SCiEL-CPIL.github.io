# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    n = 14
    d = 3
    start_time = time.time()
    time_limit = 600  # seconds

    def get_score(pts):
        dists = pdist(pts, metric='euclidean')
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    # Initial configuration: 12 vertices of a regular icosahedron + 2 extra points
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    
    # Normalize icosahedron to unit sphere
    norms_ico = np.linalg.norm(ico_verts, axis=1)
    ico_verts_norm = ico_verts / norms_ico[:, np.newaxis]
    
    # Initial 14 points
    points = np.vstack([ico_verts_norm, np.array([[0, 0, 1.0], [0, 0, -1.0]])])
    
    def soft_min_max_objective(x, s):
        """Smooth surrogate for maximizing (d_min/d_max)^2 using log-sum-exp."""
        if time.time() - start_time > time_limit - 10:
            return 1e10
        
        p = x.reshape((n, d))
        dists = pdist(p, metric='euclidean')
        
        # Soft min: -log(sum(exp(-d_i/s))/n) approximates min(d_i)
        # Soft max: log(sum(exp(d_i/s))/n) approximates max(d_i)
        # We want to maximize (soft_min / soft_max)^2, so minimize (soft_max / soft_min)^2
        
        # For numerical stability:
        neg_d_over_s = -dists / s
        max_neg = np.max(neg_d_over_s)
        soft_min = -s * (max_neg + np.log(np.sum(np.exp(neg_d_over_s - max_neg)) / len(dists)))
        
        d_over_s = dists / s
        max_pos = np.max(d_over_s)
        soft_max = s * (max_pos + np.log(np.sum(np.exp(d_over_s - max_pos)) / len(dists)))
        
        if soft_min <= 0 or soft_max <= 0:
            return 1e10
        
        ratio_sq = (soft_max / soft_min) ** 2
        return ratio_sq

    # Optimize
    best_points = points.copy()
    best_score = get_score(points)
    
    # Try multiple initial conditions with different temperature schedules
    initializations = [
        points,
        np.vstack([ico_verts_norm, np.array([[0, 0, 0.5], [0, 0, -0.5]])]),
        np.vstack([ico_verts_norm, np.array([[0, 0, 2.0], [0, 0, -2.0]])]),
    ]
    
    # Add random initializations
    np.random.seed(42)
    for _ in range(5):
        rand_points = np.random.randn(n, d)
        rand_points = rand_points / np.linalg.norm(rand_points, axis=1)[:, np.newaxis]
        initializations.append(rand_points)
    
    # Temperature schedules: (start_s, end_s, n_steps)
    temp_schedules = [
        (0.05, 0.001, 5),  # Start wide, end narrow
        (0.02, 0.005, 3),  # Moderate
        (0.1, 0.01, 7),    # Very wide start
    ]
    
    for init_pts in initializations:
        if time.time() - start_time > time_limit - 30:
            break
        
        for (s_start, s_end, n_steps) in temp_schedules:
            if time.time() - start_time > time_limit - 30:
                break
            
            # Create temperature schedule
            temps = np.geomspace(s_start, s_end, n_steps)
            current_x = init_pts.ravel().copy()
            
            for s in temps:
                if time.time() - start_time > time_limit - 10:
                    break
                
                try:
                    # Optimize with current temperature
                    res = minimize(
                        lambda x: soft_min_max_objective(x, s),
                        current_x,
                        method='L-BFGS-B',
                        options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                    )
                    
                    if res.fun < 1e10:
                        current_x = res.x
                        opt_points = current_x.reshape((n, d))
                        opt_score = get_score(opt_points)
                        
                        if opt_score > best_score:
                            best_points = opt_points.copy()
                            best_score = opt_score
                except Exception:
                    pass
            
            # Final polish with very small temperature
            if time.time() - start_time < time_limit - 30:
                try:
                    res_polish = minimize(
                        lambda x: soft_min_max_objective(x, 0.001),
                        current_x,
                        method='L-BFGS-B',
                        options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
                    )
                    
                    if res_polish.fun < 1e10:
                        polish_points = res_polish.x.reshape((n, d))
                        polish_score = get_score(polish_points)
                        if polish_score > best_score:
                            best_points = polish_points.copy()
                            best_score = polish_score
                except Exception:
                    pass

    # Final check for validity
    if not np.all(np.isfinite(best_points)):
        best_points = points.copy()
    
    return best_points

# EVOLVE-BLOCK-END
