# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
from itertools import combinations


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """

    n = 16
    d = 2
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s kill

    def get_score(pts):
        """Compute the actual score R = (d_min/d_max)^2"""
        dists = pdist(pts)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def soft_min(dists, t=1.0):
        """Differentiable approximation of the minimum distance using log-sum-exp."""
        a = -dists / t
        a_max = np.max(a)
        if not np.isfinite(a_max):
            return 0.0
        s = np.sum(np.exp(a - a_max))
        if s == 0:
            return 0.0
        return -t * (a_max + np.log(s / len(dists)))

    def soft_max(dists, t=1.0):
        """Differentiable approximation of the maximum distance using log-sum-exp."""
        a = dists / t
        a_max = np.max(a)
        if not np.isfinite(a_max):
            return 0.0
        s = np.sum(np.exp(a - a_max))
        if s == 0:
            return 0.0
        return t * (a_max + np.log(s / len(dists)))

    def objective(x, t_min=1.0, t_max=1.0):
        """
        Objective function to minimize: -log(soft_min/soft_max)
        which is equivalent to minimizing log(soft_max/soft_min)
        """
        pts = x.reshape(n, d)
        dists = pdist(pts)
        
        sm = soft_min(dists, t_min)
        sx = soft_max(dists, t_max)
        
        if sm <= 0 or sx <= 0 or not np.isfinite(sm) or not np.isfinite(sx):
            return 1e10
        
        return np.log(sx / sm)

    # Generate the 19-point hexagonal lattice
    # Center: (0,0)
    # Ring 1: 6 points at distance 1, angles 0, 60, ...
    # Ring 2: 12 points at distance 2, angles 30, 90, ... (offset by 30 deg)
    # Actually, standard hex lattice:
    # (0,0)
    # (1,0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1,0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
    # (2,0), (1.5, sqrt(3)/2), (0.5, sqrt(3)), (-0.5, sqrt(3)), (-1.5, sqrt(3)/2), (-2,0), (-1.5, -sqrt(3)/2), (-0.5, -sqrt(3)), (0.5, -sqrt(3)), (1.5, -sqrt(3)/2)
    
    pts_19 = []
    pts_19.append([0.0, 0.0])
    
    # Ring 1
    for i in range(6):
        angle = i * np.pi / 3
        pts_19.append([np.cos(angle), np.sin(angle)])
        
    # Ring 2
    for i in range(12):
        angle = i * np.pi / 6 + np.pi / 6
        pts_19.append([2.0 * np.cos(angle), 2.0 * np.sin(angle)])
        
    pts_19 = np.array(pts_19)
    
    # Find the best subset of 16 points (remove 3) from the 19-point lattice
    best_subset_indices = None
    best_rigid_score = -1.0
    
    # There are C(19,3) = 969 combinations. This is fast.
    for remove_indices in combinations(range(19), 3):
        keep_indices = [i for i in range(19) if i not in remove_indices]
        subset = pts_19[keep_indices]
        
        # Normalize to unit max distance for comparison consistency, though ratio is scale invariant
        # But we just need the ratio
        score = get_score(subset)
        if score > best_rigid_score:
            best_rigid_score = score
            best_subset_indices = keep_indices
            
    # Use the best rigid subset as the initial guess
    initial_points = pts_19[best_subset_indices].copy()
    
    # Center and scale the initial points
    initial_points = initial_points - np.mean(initial_points, axis=0)
    # Scale to have max distance 1? Not strictly necessary for ratio, but helps numerical stability
    # The optimizer should handle scale, but fixing scale might help. 
    # However, R is scale invariant. Let's just center.
    
    best_points = initial_points.copy()
    best_score = get_score(best_points)
    
    # Optimize from this strong starting point
    # We will do a few restarts from this best rigid configuration, with slight perturbations
    np.random.seed(42)
    
    # Strategy 1: Direct L-BFGS-B from the rigid best subset
    if time.time() < deadline - 100:
        try:
            x0 = best_points.flatten()
            # Use a moderate temperature for the soft min/max to guide the gradient
            result = minimize(
                objective,
                x0,
                args=(0.1, 0.1),
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            pts_opt = result.x.reshape(n, d)
            score_opt = get_score(pts_opt)
            if score_opt > best_score:
                best_score = score_opt
                best_points = pts_opt.copy()
        except Exception:
            pass

    # Strategy 2: Nelder-Mead polish (robust to non-smoothness of true min/max)
    if time.time() < deadline - 50:
        try:
            result_nm = minimize(
                lambda x: -get_score(x.reshape(n, d)),
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            pts_nm = result_nm.x.reshape(n, d)
            score_nm = get_score(pts_nm)
            if score_nm > best_score:
                best_score = score_nm
                best_points = pts_nm.copy()
        except Exception:
            pass

    # Strategy 3: Another L-BFGS-B pass with tighter temperature
    if time.time() < deadline - 20:
        try:
            x0 = best_points.flatten()
            result_final = minimize(
                objective,
                x0,
                args=(0.01, 0.01),
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-14}
            )
            pts_final = result_final.x.reshape(n, d)
            score_final = get_score(pts_final)
            if score_final > best_score:
                best_score = score_final
                best_points = pts_final.copy()
        except Exception:
            pass

    return best_points


# EVOLVE-BLOCK-END
