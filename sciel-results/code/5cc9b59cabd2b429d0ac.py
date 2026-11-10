# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time
import itertools

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the points.

    """
    n_points = 16
    start_time = time.time()
    deadline = start_time + 620.0  # Safety margin before 630s kill

    def compute_score(points):
        if points.shape != (n_points, 2):
            return -np.inf
        if not np.all(np.isfinite(points)):
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

    def normalize_points(points):
        """Scale points such that the maximum pairwise distance is 1."""
        d = pdist(points)
        d_max = np.max(d)
        if d_max > 1e-12:
            return points / d_max
        return points

    def soft_min_objective(x, beta):
        """
        Objective function to minimize: -soft_min(distances).
        The points are normalized to unit diameter internally.
        """
        points = x.reshape(n_points, 2)
        points = normalize_points(points)
        d = pdist(points)
        if np.max(d) < 1e-12:
            return 0.0
        # Use log-sum-exp trick for stability if needed, but beta is moderate
        vals = np.exp(-beta * d)
        sum_vals = np.sum(vals)
        if sum_vals == 0:
            return 0.0
        soft_min_val = -1.0 / beta * np.log(sum_vals)
        return -soft_min_val

    def optimize_candidate(points, time_budget=5.0, betas=None):
        """
        Optimizes a candidate using L-BFGS-B with increasing beta (temperature).
        """
        if betas is None:
            betas = [5, 10, 20, 50, 100, 200, 500, 1000]
            
        points = points.copy()
        points = normalize_points(points)
        
        best_points = points.copy()
        best_score = compute_score(best_points)
        
        start_opt = time.time()
        
        for beta in betas:
            if time.time() - start_opt > time_budget:
                break
            
            def obj(x):
                return soft_min_objective(x, beta)
            
            try:
                res = minimize(
                    obj, 
                    points.flatten(), 
                    method='L-BFGS-B', 
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 2000}
                )
                
                opt_points = res.x.reshape(n_points, 2)
                opt_points = normalize_points(opt_points)
                opt_score = compute_score(opt_points)
                
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = opt_points.copy()
                    
                # Update current points for next beta stage
                points = best_points.copy()
                
            except Exception:
                pass
                
            if time.time() - start_opt > time_budget:
                break
                
        return best_points, best_score

    best_points = None
    best_score = -np.inf
    all_candidates = []

    # 1. Generate Diverse Initial Candidates
    
    # 1a. Triangular Lattice (Cluster)
    s = 1.0
    h = s * np.sqrt(3) / 2
    lattice_points = []
    for i in range(-5, 6):
        for j in range(-5, 6):
            x = i * s + (j % 2) * s / 2
            y = j * h
            lattice_points.append([x, y])
    lattice_points = np.array(lattice_points)
    centroid = np.mean(lattice_points, axis=0)
    dists_to_centroid = np.sum((lattice_points - centroid) ** 2, axis=1)
    idx = np.argsort(dists_to_centroid)[:n_points]
    all_candidates.append(lattice_points[idx])

    # 1b. Square Grid
    grid_coords = []
    for i in range(4):
        for j in range(4):
            grid_coords.append([i - 1.5, j - 1.5])
    all_candidates.append(np.array(grid_coords))

    # 1c. Regular 16-gon
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    poly_points = np.column_stack([np.cos(angles), np.sin(angles)])
    all_candidates.append(poly_points)

    # 1d. Concentric Circles (4 inner, 12 outer)
    inner_angles = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    outer_angles = np.linspace(0, 2 * np.pi, 12, endpoint=False) + np.pi/12
    inner_radius = 0.4
    outer_radius = 1.0
    inner_points = np.column_stack([inner_radius * np.cos(inner_angles), inner_radius * np.sin(inner_angles)])
    outer_points = np.column_stack([outer_radius * np.cos(outer_angles), outer_radius * np.sin(outer_angles)])
    all_candidates.append(np.vstack([inner_points, outer_points]))

    # 1e. Concentric Circles (6 inner, 10 outer)
    inner_angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    outer_angles = np.linspace(0, 2 * np.pi, 10, endpoint=False) + np.pi/10
    inner_radius = 0.45
    outer_radius = 1.0
    inner_points = np.column_stack([inner_radius * np.cos(inner_angles), inner_radius * np.sin(inner_angles)])
    outer_points = np.column_stack([outer_radius * np.cos(outer_angles), outer_radius * np.sin(outer_angles)])
    all_candidates.append(np.vstack([inner_points, outer_points]))

    # 1f. Concentric Circles (1 center, 5 inner, 10 outer) - From Current Program
    center = np.array([0.0, 0.0])
    inner_r = 0.6
    outer_r = 1.0
    inner_angles = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    outer_angles = np.linspace(0, 2 * np.pi, 10, endpoint=False) + np.pi / 10
    inner_pts = np.column_stack([inner_r * np.cos(inner_angles), inner_r * np.sin(inner_angles)])
    outer_pts = np.column_stack([outer_r * np.cos(outer_angles), outer_r * np.sin(outer_angles)])
    ring_config = np.vstack([center.reshape(1, -1), inner_pts, outer_pts])
    all_candidates.append(ring_config)

    # 1g. Two Concentric Squares - From Current Program
    sq1 = np.array(list(itertools.product([-0.5, 0.5], repeat=2)), dtype=float)
    sq2 = np.array(list(itertools.product([-1.0, 1.0], repeat=2)), dtype=float)
    theta = np.pi / 4
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    sq2_rot = sq2 @ rot.T
    two_sq_config = np.vstack([sq1, sq2_rot])
    all_candidates.append(two_sq_config)

    # 1h. Random Points (Multiple seeds)
    np.random.seed(42)
    for seed in range(50):
        random_points = np.random.randn(n_points, 2)
        # Normalize roughly to unit diameter to start in a good range
        d_max_init = np.max(pdist(random_points))
        if d_max_init > 1e-9:
            random_points = random_points / d_max_init
        all_candidates.append(random_points)

    # Optimize each candidate
    # We have ~60 candidates. 620s / 60 ~ 10s per candidate.
    # L-BFGS-B with 300 iterations is fast.
    time_per_candidate = 600.0 / len(all_candidates)
    
    for i, init_pts in enumerate(all_candidates):
        if time.time() > deadline - 10:
            break
        
        remaining_time = deadline - time.time() - 5
        if remaining_time < 1.0:
            break
            
        # Allocate time based on remaining time and candidates left
        # Simple strategy: fixed budget per candidate, adjusted if running out of time
        budget = min(time_per_candidate, remaining_time / (len(all_candidates) - i))
        if budget < 0.5:
            budget = 0.5
            
        try:
            opt_pts, opt_score = optimize_candidate(init_pts, time_budget=budget)
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_pts.copy()
        except Exception:
            pass

    # If we still have time, do a few more random restarts with higher precision
    if best_points is not None and time.time() < deadline - 30:
        # Use the best found point as a base for further refinement with very high beta
        # or just run a few more random seeds with larger budgets
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            np.random.seed(int(time.time() * 1000) % 100000)
            random_points = np.random.randn(n_points, 2)
            d_max_init = np.max(pdist(random_points))
            if d_max_init > 1e-9:
                random_points = random_points / d_max_init
            
            try:
                # Give these more time as they are the last chance for improvement
                opt_pts, opt_score = optimize_candidate(random_points, time_budget=10.0)
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = opt_pts.copy()
            except Exception:
                pass

    if best_points is not None:
        # Final check for validity
        if np.all(np.isfinite(best_points)) and best_points.shape == (16, 2):
            return best_points

    # Fallback: return the first candidate (normalized)
    return normalize_points(all_candidates[0])

# EVOLVE-BLOCK-END
