# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
import time
from scipy.optimize import minimize
import itertools

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) of the 16 points.
    """
    start_time = time.time()
    deadline = start_time + 620.0  # Safety margin

    n_points = 16

    def compute_score(points):
        if points.shape != (n_points, 2):
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
        # Soft min: -1/beta * log(sum(exp(-beta * d)))
        # To avoid overflow, we can use log-sum-exp trick, but beta*d might be large.
        # d is in [0, 1]. beta is typically 10-100.
        # exp(-beta * d) is small.
        # Let's just compute it directly, it's safe for beta < 1000.
        vals = np.exp(-beta * d)
        sum_vals = np.sum(vals)
        if sum_vals == 0:
            return 0.0
        soft_min_val = -1.0 / beta * np.log(sum_vals)
        return -soft_min_val

    def optimize_candidate(points, time_budget=5.0):
        """
        Optimizes a candidate using L-BFGS-B with increasing beta (temperature).
        """
        points = points.copy()
        points = normalize_points(points)
        
        best_points = points.copy()
        best_score = compute_score(best_points)
        
        # Schedule of betas to sharpen the minimum
        betas = [5, 10, 20, 50, 100, 200, 500]
        
        start_opt = time.time()
        
        for beta in betas:
            if time.time() - start_opt > time_budget:
                break
            
            # Use L-BFGS-B for gradient-based optimization
            # We provide the objective function
            def obj(x):
                return soft_min_objective(x, beta)
            
            try:
                res = minimize(
                    obj, 
                    points.flatten(), 
                    method='L-BFGS-B', 
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if res.success or res.fun < 1e-10: # Check if it converged or is very good
                    opt_points = res.x.reshape(n_points, 2)
                    opt_points = normalize_points(opt_points)
                    opt_score = compute_score(opt_points)
                    
                    if opt_score > best_score:
                        best_score = opt_score
                        best_points = opt_points.copy()
                        
                    # Update current points for next beta stage
                    points = best_points.copy()
                    
            except Exception as e:
                # If L-BFGS-B fails, try a quick Nelder-Mead step?
                # For now, just keep the previous best.
                pass
                
            # If time is running out, break
            if time.time() - start_opt > time_budget:
                break
                
        return best_points, best_score

    best_points = None
    best_score = -np.inf

    # Generate diverse initial candidates
    candidates = []

    # 1. Triangular Lattice
    s = 1.0
    h = s * np.sqrt(3) / 2
    lattice_points = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            x = i * s + (j % 2) * s / 2
            y = j * h
            lattice_points.append([x, y])
    lattice_points = np.array(lattice_points)
    centroid = np.mean(lattice_points, axis=0)
    dists_to_centroid = np.sum((lattice_points - centroid) ** 2, axis=1)
    idx = np.argsort(dists_to_centroid)[:n_points]
    candidates.append(lattice_points[idx])

    # 2. Square Grid
    grid = np.array([np.meshgrid(np.linspace(-1.5, 1.5, 4), np.linspace(-1.5, 1.5, 4))]).reshape(2, -1).T
    candidates.append(grid)

    # 3. Random Points (Multiple seeds)
    for seed in range(30):
        np.random.seed(seed * 100 + 7)
        random_points = np.random.randn(n_points, 2)
        candidates.append(random_points)

    # 4. Concentric Circles
    # 4 inner, 12 outer
    inner_angles = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    outer_angles = np.linspace(0, 2 * np.pi, 12, endpoint=False) + np.pi/12
    inner_radius = 0.4
    outer_radius = 1.0
    inner_points = np.column_stack([inner_radius * np.cos(inner_angles), inner_radius * np.sin(inner_angles)])
    outer_points = np.column_stack([outer_radius * np.cos(outer_angles), outer_radius * np.sin(outer_angles)])
    two_circle = np.vstack([inner_points, outer_points])
    candidates.append(two_circle)
    
    # 5. 6 inner, 10 outer
    inner_angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    outer_angles = np.linspace(0, 2 * np.pi, 10, endpoint=False) + np.pi/10
    inner_radius = 0.45
    outer_radius = 1.0
    inner_points = np.column_stack([inner_radius * np.cos(inner_angles), inner_radius * np.sin(inner_angles)])
    outer_points = np.column_stack([outer_radius * np.cos(outer_angles), outer_radius * np.sin(outer_angles)])
    two_circle_2 = np.vstack([inner_points, outer_points])
    candidates.append(two_circle_2)
    
    # 6. Regular 16-gon
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    poly_points = np.column_stack([np.cos(angles), np.sin(angles)])
    candidates.append(poly_points)

    # Evaluate and optimize candidates
    processed = []
    
    # We have ~36 candidates. 
    # Budget per candidate: ~5s. Total ~180s.
    # Remaining time for polishing or more random restarts.
    
    for i, cand in enumerate(candidates):
        if time.time() > deadline - 10:
            break
            
        # Optimize each candidate
        opt_cand, opt_score = optimize_candidate(cand, time_budget=5.0)
        
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_cand.copy()
            
        processed.append((opt_score, opt_cand))

    # Sort processed by score
    processed.sort(key=lambda x: x[0], reverse=True)
    
    # Refine the top candidates with higher precision (higher beta, more iterations)
    # Use the remaining time
    if best_points is None:
        best_points = candidates[0]
        
    # Take the top 5 and give them more time with higher beta
    top_candidates = processed[:5]
    for score, cand in top_candidates:
        if time.time() > deadline - 5:
            break
            
        # Run a second pass with higher betas
        def optimize_candidate_high(cand, time_budget=10.0):
            points = cand.copy()
            points = normalize_points(points)
            best_points = points.copy()
            best_score = compute_score(best_points)
            
            betas = [100, 200, 500, 1000, 2000]
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
                        options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
                    )
                    
                    if res.success or res.fun < 1e-12:
                        opt_points = res.x.reshape(n_points, 2)
                        opt_points = normalize_points(opt_points)
                        opt_score = compute_score(opt_points)
                        
                        if opt_score > best_score:
                            best_score = opt_score
                            best_points = opt_points.copy()
                        points = best_points.copy()
                        
                except Exception:
                    pass
                
                if time.time() - start_opt > time_budget:
                    break
            return best_points, best_score

        opt_cand, opt_score = optimize_candidate_high(cand, time_budget=10.0)
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_cand.copy()

    return best_points

# EVOLVE-BLOCK-END
