# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import time
from scipy.optimize import minimize
import random
import sys

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin, evaluator kills at 630s

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

    def soft_min_objective_and_grad(x, beta):
        """
        Computes the soft-min objective and its gradient.
        We minimize -soft_min(distances) to maximize the minimum distance.
        """
        points = x.reshape(n_points, 2)
        d = pdist(points)
        d_max = np.max(d)
        if d_max < 1e-12:
            return 0.0, np.zeros_like(x)
        
        # Normalize distances to [0, 1] to make beta scale invariant
        d_norm = d / d_max
        
        # Calculate weights for soft min
        exp_vals = np.exp(-beta * d_norm)
        sum_exp = np.sum(exp_vals)
        
        if sum_exp == 0:
            return 0.0, np.zeros_like(x)
            
        # Value: -1/beta * log(sum_exp)
        val = -1.0 / beta * np.log(sum_exp)
        
        # Gradient computation
        grad = np.zeros_like(x)
        
        idx = 0
        for i in range(n_points):
            for j in range(i + 1, n_points):
                d_ij = d[idx]
                if d_ij > 1e-12:
                    weight = exp_vals[idx] / sum_exp
                    vec_ij = points[i] - points[j]
                    grad[i] += weight * (vec_ij / d_ij) / d_max
                    grad[j] += weight * (-vec_ij / d_ij) / d_max
                idx += 1
        
        return val, grad

    def epigraph_objective_and_grad(x, beta):
        """
        Uses an epigraph formulation: minimize t subject to d_ij >= t for all i,j.
        With soft constraints: minimize t + (1/beta) * log(sum(exp(-beta * (d_ij - t))))
        This is a smooth approximation that handles both min and max simultaneously.
        """
        points = x.reshape(n_points, 2)
        d = pdist(points)
        d_max = np.max(d)
        if d_max < 1e-12:
            return 0.0, np.zeros_like(x)
        
        d_norm = d / d_max
        # t is the soft-min value
        t = np.min(d_norm)
        
        # For numerical stability, use a different approach:
        # Minimize -soft_min(d_norm) which is what we had before, but let's try
        # a different parameterization
        
        # Actually, let's stick with the proven soft-min approach but with better beta scheduling
        return soft_min_objective_and_grad(x, beta)

    def optimize_candidate_gradient(points, time_budget=5.0, betas=None):
        """
        Optimizes a candidate using L-BFGS-B with analytic gradient and increasing beta.
        """
        if betas is None:
            betas = [10, 50, 100, 200, 500, 1000, 2000, 5000]
            
        points = points.copy()
        points = normalize_points(points)
        
        best_points = points.copy()
        best_score = compute_score(best_points)
        
        start_opt = time.time()
        
        for beta in betas:
            if time.time() - start_opt > time_budget:
                break
            
            def obj_and_grad(x):
                val, grad = soft_min_objective_and_grad(x, beta)
                return val, grad
            
            try:
                res = minimize(
                    obj_and_grad, 
                    points.flatten(), 
                    method='L-BFGS-B', 
                    jac=True,
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 2000}
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

    def high_precision_polish(points, time_budget=30.0):
        """
        Final high-precision polish using very high beta values and tight tolerances.
        """
        points = points.copy()
        points = normalize_points(points)
        
        best_points = points.copy()
        best_score = compute_score(best_points)
        
        start_polish = time.time()
        
        # Use very high beta values for near-exact min optimization
        high_betas = [5000, 10000, 20000, 50000, 100000]
        
        for beta in high_betas:
            if time.time() - start_polish > time_budget:
                break
            
            def obj_and_grad(x):
                val, grad = soft_min_objective_and_grad(x, beta)
                return val, grad
            
            try:
                res = minimize(
                    obj_and_grad, 
                    best_points.flatten(), 
                    method='L-BFGS-B', 
                    jac=True,
                    options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-12, 'maxfun': 5000}
                )
                
                opt_points = res.x.reshape(n_points, 2)
                opt_points = normalize_points(opt_points)
                opt_score = compute_score(opt_points)
                
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = opt_points.copy()
                    
            except Exception:
                pass
                
            if time.time() - start_polish > time_budget:
                break
        
        # Additional Nelder-Mead polish with very tight tolerance
        if time.time() - start_polish < time_budget - 5:
            def nm_objective(x):
                pts = x.reshape(n_points, 2)
                pts = normalize_points(pts)
                d = pdist(pts)
                d_min = np.min(d)
                d_max = np.max(d)
                if d_max == 0:
                    return 0.0
                return -(d_min / d_max) ** 2
            
            try:
                res = minimize(
                    nm_objective, 
                    best_points.flatten(), 
                    method='Nelder-Mead', 
                    options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                )
                nm_points = res.x.reshape(n_points, 2)
                nm_points = normalize_points(nm_points)
                nm_score = compute_score(nm_points)
                
                if nm_score > best_score:
                    best_score = nm_score
                    best_points = nm_points.copy()
                    
            except Exception:
                pass
        
        return best_points, best_score

    best_points = None
    best_score = -np.inf
    all_candidates = []

    # 1. Generate Diverse Initial Candidates
    
    # 1a. Square Grid (4x4)
    grid_coords = []
    for i in range(4):
        for j in range(4):
            grid_coords.append([i - 1.5, j - 1.5])
    all_candidates.append(np.array(grid_coords))

    # 1b. Triangular Lattice
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

    # 1e. Two concentric rings (8+8)
    ring1_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    ring2_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False) + np.pi/8
    r1 = 0.5
    r2 = 1.0
    ring1_points = np.column_stack([r1 * np.cos(ring1_angles), r1 * np.sin(ring1_angles)])
    ring2_points = np.column_stack([r2 * np.cos(ring2_angles), r2 * np.sin(ring2_angles)])
    all_candidates.append(np.vstack([ring1_points, ring2_points]))

    # 1f. Three concentric rings (4+6+6)
    ring1a = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    ring2a = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi/6
    ring3a = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    r1f = 0.3
    r2f = 0.7
    r3f = 1.0
    r1pts = np.column_stack([r1f * np.cos(ring1a), r1f * np.sin(ring1a)])
    r2pts = np.column_stack([r2f * np.cos(ring2a), r2f * np.sin(ring2a)])
    r3pts = np.column_stack([r3f * np.cos(ring3a), r3f * np.sin(ring3a)])
    all_candidates.append(np.vstack([r1pts, r2pts, r3pts]))

    # 1g. Center + 15 on circle
    center = np.array([[0, 0]])
    circ_angles = np.linspace(0, 2 * np.pi, 15, endpoint=False)
    circ_pts = np.column_stack([np.cos(circ_angles), np.sin(circ_angles)])
    all_candidates.append(np.vstack([center, circ_pts]))

    # 1h. Random Points with different seeds
    for seed in range(30):
        np.random.seed(seed * 100 + 7)
        random_points = np.random.randn(n_points, 2)
        random_points = normalize_points(random_points)
        all_candidates.append(random_points)

    # 1i. Random with structure: random radii and angles
    for seed in range(15):
        np.random.seed(seed * 200 + 13)
        radii = np.sort(np.random.uniform(0.1, 1.0, n_points))
        angles = np.sort(np.random.uniform(0, 2 * np.pi, n_points))
        structured_pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        structured_pts = normalize_points(structured_pts)
        all_candidates.append(structured_pts)

    # 2. Evaluate and Optimize Candidates
    
    scored_candidates = []
    for cand in all_candidates:
        s = compute_score(cand)
        scored_candidates.append((s, cand))
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Optimize top candidates with more time
    for i, (init_score, cand) in enumerate(scored_candidates):
        if time.time() > deadline - 90:
            break
        
        if i < 3:
            budget = 20.0
        elif i < 8:
            budget = 10.0
        elif i < 15:
            budget = 5.0
        else:
            budget = 2.0
            
        opt_cand, opt_score = optimize_candidate_gradient(cand, time_budget=budget)
        
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_cand.copy()

    # 3. Final high-precision polish
    if best_points is not None and time.time() < deadline - 15:
        polish_budget = min(60.0, deadline - time.time() - 10)
        if polish_budget > 10:
            polished_points, polished_score = high_precision_polish(best_points, time_budget=polish_budget)
            if polished_score > best_score:
                best_score = polished_score
                best_points = polished_points.copy()

    # Ensure valid output
    if best_points is None:
        best_points = all_candidates[0]
    
    return best_points

# EVOLVE-BLOCK-END
