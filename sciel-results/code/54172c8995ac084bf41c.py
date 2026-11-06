"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    def score(pts):
        """Calculate the raw score Delta/n^n for a set of 8 points."""
        if len(pts) != 8:
            return 0.0
        pts = np.array(pts)
        
        # Calculate all pairwise distances
        dists = []
        for i in range(8):
            for j in range(8):
                if i != j:
                    dists.append(np.linalg.norm(pts[i] - pts[j]))
        dists = np.array(dists)
        
        # Diameter is the max distance
        d = np.max(dists)
        if d == 0:
            return 0.0
        
        # Scale to diameter 2
        scaled_dists = dists * (2.0 / d)
        
        # Product of distances
        # To avoid underflow/overflow, use log-sum-exp trick or just product if small
        # For n=8, 56 pairs. Product is fine in float64 usually, but log is safer.
        log_prod = np.sum(np.log(scaled_dists))
        prod = np.exp(log_prod)
        
        # Score is Delta / n^n
        # n=8, so n^n = 8^8 = 16777216
        score_val = prod / (8.0 ** 8)
        
        return score_val

    def objective(params):
        """Objective function for minimization (negative score)."""
        pts = params.reshape(8, 2)
        # Ensure distinct points to avoid division by zero in scaling
        # If points are identical, distance is 0, score is 0 (or -inf in log)
        # We want to maximize score, so minimize -score.
        # If score is 0, -score is 0. But we want to avoid 0.
        # Actually, if distance is 0, log(0) is -inf, score is 0.
        # Let's add a small penalty for close points to help the optimizer.
        min_dist = np.inf
        for i in range(8):
            for j in range(i+1, 8):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < min_dist:
                    min_dist = d
        if min_dist < 1e-9:
            return 1e10 # Huge penalty
        
        s = score(pts)
        return -s

    # Initial bounds: coordinates between -10 and 10
    bounds = [(-10, 10)] * 16
    
    best_score = 0.0
    best_points = np.zeros((8, 2))
    
    # Regular octagon as a fallback
    reg_oct = np.array([[np.cos(2 * np.pi * k / 8), np.sin(2 * np.pi * k / 8)] for k in range(8)])
    best_score = score(reg_oct)
    best_points = reg_oct.copy()

    # Global Search: Differential Evolution
    # Maximize score -> Minimize -score
    # DE can be slow, so limit maxiter and popsize
    # The deadline is 140s. DE might take a while.
    # Let's try a moderate population and iterations.
    
    try:
        res = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=100, 
            popsize=15, 
            tol=1e-5,
            init='sobol',
            polish=False # We will polish manually
        )
        
        if time.time() > deadline:
             return best_points.tolist()
             
        if res.success or res.fun < 0: # If it found something better than 0 score (i.e. -score < 0)
            cand_pts = res.x.reshape(8, 2)
            cand_score = score(cand_pts)
            if cand_score > best_score:
                best_score = cand_score
                best_points = cand_pts.copy()
                
    except Exception:
        pass

    # Local Refinement: L-BFGS-B
    # Start from the best point found by DE (or regular octagon if DE failed)
    try:
        # L-BFGS-B requires gradients or approximates them. 
        # It's unconstrained in terms of bounds, but we can use bounds.
        res_local = minimize(
            objective, 
            best_points.flatten(), 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if time.time() > deadline:
             return best_points.tolist()
             
        if res_local.success:
            cand_pts = res_local.x.reshape(8, 2)
            cand_score = score(cand_pts)
            if cand_score > best_score:
                best_score = cand_score
                best_points = cand_pts.copy()
                
    except Exception:
        pass
    
    # Final check
    if time.time() > deadline:
        return best_points.tolist()

    return best_points.tolist()

# EVOLVE-BLOCK-END
