# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import time
from scipy.optimize import minimize

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) of the 16 points.
    """
    start_time = time.time()
    deadline = start_time + 620.0

    def compute_score(points):
        if points.shape != (16, 2):
            return -np.inf
        d = pdist(points)
        if d.size == 0:
            return 0.0
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max == 0:
            return 0.0
        ratio = d_min / d_max
        return ratio * ratio

    def objective_smooth(points_flat, beta=100.0):
        points = points_flat.reshape(16, 2)
        centroid = np.mean(points, axis=0)
        points_centered = points - centroid
        norm = np.linalg.norm(points_centered, axis=1)
        sum_sq = np.sum(norm**2)
        if sum_sq < 1e-12:
            return 1e10
        points_normalized = points_centered / np.sqrt(sum_sq)
        
        d = pdist(points_normalized)
        if d.size == 0 or np.any(d < 1e-12):
            return 1e10
        
        d_max_val = np.max(d)
        log_sum_exp_max = d_max_val + np.log(np.sum(np.exp(beta * (d - d_max_val))))
        d_max_soft = log_sum_exp_max / beta
        
        d_min_val = np.min(d)
        log_sum_exp_min = -beta * d_min_val + np.log(np.sum(np.exp(-beta * (d - d_min_val))))
        d_min_soft = -log_sum_exp_min / beta
        
        if d_min_soft <= 1e-12:
            return 1e10
            
        return np.log(d_max_soft) - np.log(d_min_soft)

    def objective_exact(points_flat):
        points = points_flat.reshape(16, 2)
        score = compute_score(points)
        if not np.isfinite(score):
            return 1e10
        return -score

    def apply_radial_warp(points, a):
        """
        Apply radial warp: r' = r * (1 + a * r^2)
        This bulges the center if a > 0, compressing the periphery relative to the center.
        """
        r = np.linalg.norm(points, axis=1)
        # Avoid division by zero if any point is at origin
        mask = r > 1e-12
        r_new = np.zeros_like(r)
        r_new[mask] = r[mask] * (1.0 + a * r[mask]**2)
        
        # Reconstruct points
        # If r was 0, point stays at 0
        points_new = np.zeros_like(points)
        # For non-zero r, scale the direction
        inv_r = np.zeros_like(r)
        inv_r[mask] = 1.0 / r[mask]
        points_new[mask] = points[mask] * (r_new[mask] / r[mask])[:, np.newaxis]
        return points_new

    # Phase 0: Deterministic Radial Warp Construction
    # Start with a 4x4 grid centered at origin
    x = np.array([-1.5, -0.5, 0.5, 1.5])
    y = np.array([-1.5, -0.5, 0.5, 1.5])
    X, Y = np.meshgrid(x, y)
    grid_points = np.column_stack([X.ravel(), Y.ravel()])
    
    # Optimize the warp parameter 'a'
    # We want to find 'a' that maximizes the score of the warped grid
    def warp_score(a_val):
        warped = apply_radial_warp(grid_points, a_val)
        return compute_score(warped)

    # Scan a range of 'a' values to find the best deterministic warp
    a_vals = np.linspace(-0.5, 1.5, 200)
    best_a = 0
    best_warp_score = -np.inf
    for a in a_vals:
        s = warp_score(a)
        if s > best_warp_score:
            best_warp_score = s
            best_a = a

    # Refine 'a' with a local optimizer
    def neg_warp_score(a_flat):
        a_val = a_flat[0]
        return -warp_score(a_val)

    result_a = minimize(neg_warp_score, [best_a], method='Nelder-Mead', options={'maxiter': 100, 'xatol': 1e-6})
    best_a = result_a.x[0]
    
    initial_points = apply_radial_warp(grid_points, best_a)
    initial_score = compute_score(initial_points)

    # Phase 1: Polish the warped lattice with BFGS (Smooth Objective)
    best_points = initial_points.copy()
    best_score = initial_score

    # Add a few other structured seeds just in case, but prioritize the warp
    # 1. Two concentric circles
    inner_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    outer_angles = inner_angles + np.pi / 8
    inner_radius = 0.6
    outer_radius = 1.2
    inner_points = np.column_stack([inner_radius * np.cos(inner_angles), inner_radius * np.sin(inner_angles)])
    outer_points = np.column_stack([outer_radius * np.cos(outer_angles), outer_radius * np.sin(outer_angles)])
    two_circle = np.vstack([inner_points, outer_points])
    
    # 2. Regular 16-gon
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    circle_points = np.column_stack([np.cos(angles), np.sin(angles)])

    candidates = [initial_points, two_circle, circle_points]
    
    # Evaluate candidates
    for cand in candidates:
        s = compute_score(cand)
        if s > best_score:
            best_score = s
            best_points = cand.copy()

    # Optimize the best candidates
    # Sort candidates by score
    scored_candidates = [(compute_score(c), c) for c in candidates]
    scored_candidates.sort(key=lambda x: x[0], reverse=True)

    for score, cand in scored_candidates:
        if time.time() > deadline - 100:
            break
        try:
            result = minimize(objective_smooth, cand.flatten(), method='BFGS',
                             options={'maxiter': 200, 'gtol': 1e-6, 'disp': False})
            opt_points = result.x.reshape(16, 2)
            opt_score = compute_score(opt_points)
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_points.copy()
        except Exception:
            pass

    # Phase 2: Nelder-Mead Polish on Exact Objective
    if time.time() < deadline - 60:
        try:
            result = minimize(objective_exact, best_points.flatten(), method='Nelder-Mead',
                             options={'maxiter': 500, 'fatol': 1e-12, 'xatol': 1e-12, 'disp': False})
            opt_points = result.x.reshape(16, 2)
            opt_score = compute_score(opt_points)
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_points.copy()
        except Exception:
            pass

    # Phase 3: Small random perturbations of the best point to escape local minima
    if time.time() < deadline - 30:
        for attempt in range(5):
            if time.time() > deadline - 30:
                break
            np.random.seed(attempt * 100 + 77)
            noise = np.random.normal(0, 0.02, (16, 2))
            perturbed = best_points + noise
            try:
                result = minimize(objective_smooth, perturbed.flatten(), method='BFGS',
                                 options={'maxiter': 100, 'gtol': 1e-5, 'disp': False})
                opt_points = result.x.reshape(16, 2)
                opt_score = compute_score(opt_points)
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = opt_points.copy()
                    
                    # Quick NM polish
                    if time.time() < deadline - 10:
                        result2 = minimize(objective_exact, opt_points.flatten(), method='Nelder-Mead',
                                     options={'maxiter': 200, 'fatol': 1e-10, 'xatol': 1e-10, 'disp': False})
                        opt_points2 = result2.x.reshape(16, 2)
                        opt_score2 = compute_score(opt_points2)
                        if opt_score2 > best_score:
                            best_score = opt_score2
                            best_points = opt_points2.copy()
            except Exception:
                pass

    return best_points

# EVOLVE-BLOCK-END
