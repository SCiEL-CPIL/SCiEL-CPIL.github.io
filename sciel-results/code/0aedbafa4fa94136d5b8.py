import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n = 8
    # Regular octagon
    theta0 = 2 * np.pi * np.arange(n) / n
    points0 = np.column_stack([np.cos(theta0), np.sin(theta0)])
    
    def get_score(points):
        # points: 8x2
        d = np.linalg.norm(points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=2)
        mask = ~np.eye(n, dtype=bool)
        dists = d[mask]
        if len(dists) == 0:
            return 0.0
        max_d = np.max(dists)
        if max_d < 1e-9:
            return 0.0
        scaled_dists = dists * (2.0 / max_d)
        if np.any(scaled_dists <= 1e-9):
            return 0.0
        log_delta = np.sum(np.log(scaled_dists))
        # Delta = exp(log_delta)
        # Score = Delta / n^n
        # log(Score) = log_delta - n * log(n)
        log_score = log_delta - n * np.log(n)
        return np.exp(log_score)

    best_score = get_score(points0)
    best_points = points0
    
    # 1. Angle Optimization
    # Fix theta_0 = 0. Optimize theta_1 ... theta_7.
    def angle_objective(thetas):
        # thetas: 7 angles
        # Add 0 at start
        all_thetas = np.concatenate(([0.0], thetas))
        # Sort to ensure unique representation? Not strictly needed for distance calc,
        # but helps if we assume ordered points.
        # Actually, distance calc is invariant to permutation.
        # But DE might find equivalent permutations.
        x = np.cos(all_thetas)
        y = np.sin(all_thetas)
        pts = np.column_stack([x, y])
        return -get_score(pts)
        
    bounds = [(0, 2 * np.pi) for _ in range(7)]
    
    if time.time() < deadline - 50:
        try:
            res = differential_evolution(angle_objective, bounds, maxiter=100, popsize=20, tol=1e-10, seed=42, polish=True)
            if res.success:
                score = -res.fun
                if score > best_score:
                    best_score = score
                    thetas = np.concatenate(([0.0], res.x))
                    best_points = np.column_stack([np.cos(thetas), np.sin(thetas)])
        except Exception:
            pass

    # 2. 2D Local Optimization
    # Use L-BFGS-B with penalty for diameter > 2
    # This is often faster and more robust than SLSQP for many constraints
    
    def objective_2d_penalized(params):
        pts = params.reshape(8, 2)
        d = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
        mask = ~np.eye(8, dtype=bool)
        dists = d[mask]
        max_d = np.max(dists)
        
        # Penalty for exceeding diameter 2
        if max_d > 2.0:
            penalty = 100.0 * (max_d - 2.0)**2
        else:
            penalty = 0.0
            
        # Calculate score
        if max_d < 1e-9:
            return 1000.0 + penalty
        scaled_dists = dists * (2.0 / max_d)
        if np.any(scaled_dists <= 1e-9):
            return 1000.0 + penalty
            
        log_delta = np.sum(np.log(scaled_dists))
        log_score = log_delta - 8 * np.log(8)
        score = np.exp(log_score)
        return -score + penalty

    # Run several local optimizations from different starting points
    # 1. Regular octagon
    # 2. Perturbed regular octagon
    # 3. Square + 4 points?
    
    starts = [points0.flatten()]
    
    # Add a few random perturbations
    for _ in range(4):
        noise = np.random.normal(0, 0.2, size=16)
        starts.append(points0.flatten() + noise)
        
    # Add a "square-like" configuration: 4 points at square corners, 4 at midpoints of arcs?
    # Square corners: (1,0), (0,1), (-1,0), (0,-1) -> angles 0, pi/2, pi, 3pi/2
    # Midpoints: pi/4, 3pi/4, 5pi/4, 7pi/4
    # This is the regular octagon.
    
    # Try a configuration with 4 points on x-axis? No, distinct points.
    
    for x0 in starts:
        if time.time() > deadline - 10:
            break
        try:
            res = minimize(objective_2d_penalized, x0, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-12})
            if res.success:
                # Check if constraint is satisfied
                pts = res.x.reshape(8, 2)
                d = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
                mask = ~np.eye(8, dtype=bool)
                max_d = np.max(d[mask])
                if max_d <= 2.0 + 1e-6:
                    score = get_score(pts)
                    if score > best_score:
                        best_score = score
                        best_points = pts
        except Exception:
            pass

    return best_points.tolist()
