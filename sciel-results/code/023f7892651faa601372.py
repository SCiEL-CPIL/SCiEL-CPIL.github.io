# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import time
from scipy.optimize import minimize, differential_evolution
import random
import sys

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    start_time = time.time()
    deadline = start_time + 600.0  # Safety margin, evaluator kills at 630s

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

    def objective_soft_min(x, beta):
        """
        Objective function to minimize: -soft_min(distances).
        The points are normalized to unit diameter internally.
        """
        points = x.reshape(n_points, 2)
        points = normalize_points(points)
        d = pdist(points)
        if np.max(d) < 1e-12:
            return 0.0
        vals = np.exp(-beta * d)
        sum_vals = np.sum(vals)
        if sum_vals == 0:
            return 0.0
        soft_min_val = -1.0 / beta * np.log(sum_vals)
        return -soft_min_val

    def objective_value_only(x, beta):
        """Just the objective value for faster evaluation."""
        points = x.reshape(n_points, 2)
        d = pdist(points)
        d_max = np.max(d)
        if d_max < 1e-12:
            return 0.0
        d_norm = d / d_max
        vals = np.exp(-beta * d_norm)
        sum_vals = np.sum(vals)
        if sum_vals == 0:
            return 0.0
        return -1.0 / beta * np.log(sum_vals)

    def optimize_candidate(points, time_budget=5.0, betas=None):
        """
        Optimizes a candidate using L-BFGS-B with increasing beta (temperature).
        """
        if betas is None:
            betas = [5, 10, 20, 50, 100, 200, 500]
            
        points = points.copy()
        points = normalize_points(points)
        
        best_points = points.copy()
        best_score = compute_score(best_points)
        
        start_opt = time.time()
        
        for beta in betas:
            if time.time() - start_opt > time_budget:
                break
            
            def obj(x):
                return objective_value_only(x, beta)
            
            try:
                res = minimize(
                    obj, 
                    points.flatten(), 
                    method='L-BFGS-B', 
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 1000}
                )
                
                # Check result
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
                
            # If time is running out, break
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
    # Generate a large enough grid
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
    # 4x4 grid
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

    # 1f. Random Points (Multiple seeds)
    for seed in range(50):
        np.random.seed(seed * 100 + 7)
        random_points = np.random.randn(n_points, 2)
        # Normalize roughly
        random_points = random_points / np.max(pdist(random_points))
        all_candidates.append(random_points)

    # 1g. Jittered Lattice
    base_lattice = lattice_points[idx].copy()
    for j_seed in range(10):
        np.random.seed(1000 + j_seed)
        jittered = base_lattice + np.random.randn(n_points, 2) * 0.05
        all_candidates.append(jittered)

    # 1h. Hexagonal arrangement
    hex_angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    hex_points = np.column_stack([np.cos(hex_angles), np.sin(hex_angles)])
    center_point = np.array([[0.0, 0.0]])
    ring2_angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    ring2_points = np.column_stack([2.0 * np.cos(ring2_angles), 2.0 * np.sin(ring2_angles)])
    all_candidates.append(np.vstack([hex_points, center_point, ring2_points[:9]]))

    # 1i. Fibonacci spiral distribution
    golden_angle = np.pi * (3 - np.sqrt(5))
    spiral_points = []
    for i in range(16):
        r = np.sqrt(i + 1) / 4
        theta = i * golden_angle
        spiral_points.append([r * np.cos(theta), r * np.sin(theta)])
    all_candidates.append(np.array(spiral_points))

    # 2. Phase 1: Global Search with Differential Evolution (if time permits)
    # DE is good for finding a good basin in 32D.
    if time.time() < deadline - 300:
        def de_objective(x):
            points = x.reshape(n_points, 2)
            return -compute_score(points)
        
        # Bounds: assume points are within [-10, 10]
        bounds = [(-10, 10)] * (n_points * 2)
        
        try:
            # popsize 15, maxiter 100 is a reasonable start
            # tol 0.01 to stop early if converged
            de_res = differential_evolution(
                de_objective, 
                bounds, 
                maxiter=150, 
                popsize=15, 
                tol=0.01,
                seed=42,
                polish=False # We will polish later
            )
            if de_res.success or de_res.fun < 0: # fun is -score
                de_points = de_res.x.reshape(n_points, 2)
                de_points = normalize_points(de_points)
                de_score = compute_score(de_points)
                all_candidates.append(de_points)
                if de_score > best_score:
                    best_score = de_score
                    best_points = de_points.copy()
        except Exception:
            pass

    # 3. Evaluate and Optimize Candidates in Batches
    
    processed = []
    
    # Sort candidates by initial score to prioritize good ones
    scored_candidates = []
    for cand in all_candidates:
        s = compute_score(cand)
        scored_candidates.append((s, cand))
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Optimize the top N candidates with more time, rest with less
    for i, (init_score, cand) in enumerate(scored_candidates):
        if time.time() > deadline - 60:
            break
        
        # Determine budget
        if i < 5:
            budget = 10.0
        elif i < 15:
            budget = 5.0
        else:
            budget = 2.0
            
        opt_cand, opt_score = optimize_candidate(cand, time_budget=budget)
        
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_cand.copy()
            
        processed.append((opt_score, opt_cand))

    # Sort processed by score
    processed.sort(key=lambda x: x[0], reverse=True)
    
    # 4. Refine the top candidates with higher precision
    
    # Take the top 5 and give them more time with higher beta
    top_candidates = processed[:5]
    for score, cand in top_candidates:
        if time.time() > deadline - 30:
            break
            
        # Run a second pass with higher betas
        def optimize_candidate_high(cand, time_budget=20.0):
            points = cand.copy()
            points = normalize_points(points)
            best_points = points.copy()
            best_score = compute_score(best_points)
            
            betas = [100, 200, 500, 1000, 2000, 5000]
            start_opt = time.time()
            
            for beta in betas:
                if time.time() - start_opt > time_budget:
                    break
                
                def obj(x):
                    return objective_value_only(x, beta)
                
                try:
                    res = minimize(
                        obj, 
                        points.flatten(), 
                        method='L-BFGS-B', 
                        options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 2000}
                    )
                    
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

        opt_cand, opt_score = optimize_candidate_high(cand, time_budget=20.0)
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_cand.copy()

    # 5. Final Polish: Nelder-Mead on the very best point
    if best_points is not None and time.time() < deadline - 10:
        def nm_objective(x):
            points = x.reshape(n_points, 2)
            points = normalize_points(points)
            d = pdist(points)
            d_min = np.min(d)
            d_max = np.max(d)
            if d_max == 0:
                return 0.0
            return -(d_min / d_max) ** 2
        
        # Normalize before NM
        bp = normalize_points(best_points).flatten()
        
        try:
            res = minimize(
                nm_objective, 
                bp, 
                method='Nelder-Mead', 
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            nm_points = res.x.reshape(n_points, 2)
            nm_points = normalize_points(nm_points)
            nm_score = compute_score(nm_points)
            
            if nm_score > best_score:
                best_score = nm_score
                best_points = nm_points.copy()
                
        except Exception:
            pass

    # Ensure valid output
    if best_points is None:
        # Fallback to first candidate
        best_points = all_candidates[0]
    
    return best_points

# EVOLVE-BLOCK-END
