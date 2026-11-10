# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import time
from scipy.optimize import minimize
import random

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin for 630s limit, leave 40s buffer

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
        
        exponents = -beta * d
        m = np.max(exponents)
        if m < -700:
            return 0.0
        sum_exp = np.sum(np.exp(exponents - m))
        if sum_exp == 0:
            return 0.0
        soft_min_val = -(m + np.log(sum_exp)) / beta
        return -soft_min_val

    def ratio_objective(x):
        """
        Direct objective: minimize -(d_min/d_max)^2 using smoothed min
        """
        points = x.reshape(n_points, 2)
        points = normalize_points(points)
        d = pdist(points)
        if d.size == 0 or np.max(d) < 1e-12:
            return 0.0
        
        d_max = np.max(d)
        beta = 5000
        exponents = -beta * d
        m = np.max(exponents)
        if m < -700:
            return 0.0
        sum_exp = np.sum(np.exp(exponents - m))
        if sum_exp == 0:
            return 0.0
        soft_min_val = -(m + np.log(sum_exp)) / beta
        
        ratio = soft_min_val / d_max
        return -(ratio * ratio)

    def optimize_candidate_powell(points, time_budget=5.0):
        """
        Optimizes a candidate using Powell with increasing beta.
        """
        points = points.copy()
        points = normalize_points(points)
        
        best_points = points.copy()
        best_score = compute_score(best_points)
        
        betas = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
        
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
                    method='Powell', 
                    options={'maxiter': 300, 'maxfev': 8000, 'xtol': 1e-12, 'ftol': 1e-12}
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

    def optimize_candidate_nm(points, time_budget=5.0):
        """
        Refine using Nelder-Mead on the soft-min with high beta.
        """
        points = points.copy()
        points = normalize_points(points)
        best_points = points.copy()
        best_score = compute_score(best_points)
        
        betas = [1000, 2000, 5000, 10000]
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
                    method='Nelder-Mead', 
                    options={'maxiter': 8000, 'maxfev': 15000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True}
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

    def final_polish(points, time_budget=30.0):
        """
        Final high-precision polish using the direct ratio objective.
        """
        points = points.copy()
        points = normalize_points(points)
        best_points = points.copy()
        best_score = compute_score(best_points)
        
        start_opt = time.time()
        
        for iteration in range(5):
            if time.time() - start_opt > time_budget:
                break
            
            def obj(x):
                return ratio_objective(x)
            
            try:
                res = minimize(
                    obj, 
                    best_points.flatten(), 
                    method='Nelder-Mead', 
                    options={'maxiter': 10000, 'maxfev': 20000, 'xatol': 1e-15, 'fatol': 1e-15, 'adaptive': True}
                )
                
                opt_points = res.x.reshape(n_points, 2)
                opt_points = normalize_points(opt_points)
                opt_score = compute_score(opt_points)
                
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = opt_points.copy()
            except Exception:
                pass
            
            if time.time() - start_opt > time_budget:
                break
        
        return best_points, best_score

    best_points = None
    best_score = -np.inf

    # Generate diverse initial candidates
    candidates = []

    # 1. Random Points (Multiple seeds)
    for seed in range(20):
        np.random.seed(seed * 100 + 7)
        random_points = np.random.randn(n_points, 2)
        candidates.append(random_points)

    # 2. Concentric Circles
    splits = [(4, 12), (6, 10), (8, 8), (2, 14)]
    for n_inner, n_outer in splits:
        inner_angles = np.linspace(0, 2 * np.pi, n_inner, endpoint=False)
        outer_angles = np.linspace(0, 2 * np.pi, n_outer, endpoint=False) + np.pi/n_outer
        for r_inner in [0.3, 0.4, 0.5, 0.6, 0.7]:
            inner_points = np.column_stack([r_inner * np.cos(inner_angles), r_inner * np.sin(inner_angles)])
            outer_points = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
            two_circle = np.vstack([inner_points, outer_points])
            candidates.append(two_circle)

    # 3. Triangular Lattice (subset)
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
    candidates.append(lattice_points[idx])

    # 4. Square Grid
    grid = np.array([np.meshgrid(np.linspace(-1.5, 1.5, 4), np.linspace(-1.5, 1.5, 4))]).reshape(2, -1).T
    candidates.append(grid)

    # 5. Regular 16-gon
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    poly_points = np.column_stack([np.cos(angles), np.sin(angles)])
    candidates.append(poly_points)

    # 6. Known good structure: 1 center, 6 around it, 9 outer?
    center = np.array([[0.0, 0.0]])
    ring1_angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    ring1_points = np.column_stack([0.5 * np.cos(ring1_angles), 0.5 * np.sin(ring1_angles)])
    ring2_angles = np.linspace(0, 2 * np.pi, 9, endpoint=False) + np.pi/9
    ring2_points = np.column_stack([np.cos(ring2_angles), np.sin(ring2_angles)])
    flower = np.vstack([center, ring1_points, ring2_points])
    candidates.append(flower)
    
    # 7. Another flower: 1 center, 5 inner, 10 outer
    center = np.array([[0.0, 0.0]])
    ring1_angles = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    ring1_points = np.column_stack([0.6 * np.cos(ring1_angles), 0.6 * np.sin(ring1_angles)])
    ring2_angles = np.linspace(0, 2 * np.pi, 10, endpoint=False) + np.pi/10
    ring2_points = np.column_stack([np.cos(ring2_angles), np.sin(ring2_angles)])
    flower2 = np.vstack([center, ring1_points, ring2_points])
    candidates.append(flower2)
    
    # 8. Two rings: 8 inner, 8 outer
    ring1_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    ring1_points = np.column_stack([0.5 * np.cos(ring1_angles), 0.5 * np.sin(ring1_angles)])
    ring2_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False) + np.pi/8
    ring2_points = np.column_stack([np.cos(ring2_angles), np.sin(ring2_angles)])
    two_rings = np.vstack([ring1_points, ring2_points])
    candidates.append(two_rings)

    # 9. Three rings: 1 center, 5 middle, 10 outer
    center = np.array([[0.0, 0.0]])
    ring1_angles = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    ring1_points = np.column_stack([0.4 * np.cos(ring1_angles), 0.4 * np.sin(ring1_angles)])
    ring2_angles = np.linspace(0, 2 * np.pi, 5, endpoint=False) + np.pi/5
    ring2_points = np.column_stack([0.7 * np.cos(ring2_angles), 0.7 * np.sin(ring2_angles)])
    ring3_angles = np.linspace(0, 2 * np.pi, 10, endpoint=False) + np.pi/10
    ring3_points = np.column_stack([np.cos(ring3_angles), np.sin(ring3_angles)])
    three_rings = np.vstack([center, ring1_points, ring2_points, ring3_points])
    candidates.append(three_rings)

    # Evaluate and optimize candidates
    processed = []
    
    # Budget per candidate: ~3s for Powell.
    # Total candidates ~ 20 + 20 + 1 + 1 + 1 + 1 + 1 + 1 + 1 = 47.
    # 47 * 3s = 141s.
    # Remaining time for NM polishing and final polish.
    
    for i, cand in enumerate(candidates):
        if time.time() > deadline - 120:
            break
            
        # Optimize each candidate
        opt_cand, opt_score = optimize_candidate_powell(cand, time_budget=3.0)
        
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_cand.copy()
            
        processed.append((opt_score, opt_cand))

    # Sort processed by score
    processed.sort(key=lambda x: x[0], reverse=True)
    
    # Refine the top candidates with Nelder-Mead
    top_candidates = processed[:10]
    for score, cand in top_candidates:
        if time.time() > deadline - 60:
            break
            
        # Run NM polishing
        opt_cand, opt_score = optimize_candidate_nm(cand, time_budget=5.0)
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_cand.copy()
            
    # Final high-precision polish on the best candidate
    if best_points is not None and time.time() < deadline - 35:
        opt_cand, opt_score = final_polish(best_points, time_budget=30.0)
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_cand.copy()

    # If we still have time, try a few more random restarts with direct polish
    while time.time() < deadline - 15:
        np.random.seed(int(time.time() * 1000) % 10000)
        random_points = np.random.randn(n_points, 2)
        opt_cand, opt_score = optimize_candidate_powell(random_points, time_budget=2.0)
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_cand.copy()
        opt_cand, opt_score = final_polish(best_points, time_budget=10.0)
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_cand.copy()

    if best_points is None:
        # Fallback
        best_points = np.random.randn(n_points, 2)

    return best_points

# EVOLVE-BLOCK-END
