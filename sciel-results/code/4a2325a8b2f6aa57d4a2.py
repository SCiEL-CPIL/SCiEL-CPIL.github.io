import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

def construct():
    n = 8
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit

    def score_from_points(points):
        # points: (n, 2)
        diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        D = np.max(dists)
        if D < 1e-12:
            return 0.0
        scaled_dists = dists * (2.0 / D)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        prod = np.prod(scaled_dists[mask])
        return prod / (n ** n)

    def regular_octagon():
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pts = np.column_stack([np.cos(angles), np.sin(angles)])
        return pts

    best_score = score_from_points(regular_octagon())
    best_points = regular_octagon().copy()

    # New Strategy: Optimize cyclic octagon angles
    # Variables: alpha_1 ... alpha_7, alpha_8 = 2*pi - sum(alpha_1...alpha_7)
    # All alpha_i > 0
    
    def cyclic_objective(alpha_vars):
        # alpha_vars: array of 7 floats
        alphas = np.zeros(n)
        alphas[:7] = alpha_vars
        alphas[7] = 2 * np.pi - np.sum(alpha_vars)
        
        # Check positivity
        if np.any(alphas <= 1e-9):
            return 1e10  # Large penalty
        
        # Cumulative angles
        psi = np.cumsum(alphas)
        psi[0] = 0 # Start at 0
        
        # Points on unit circle
        pts = np.column_stack([np.cos(psi), np.sin(psi)])
        
        # Distances
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        unique_dists = dists[mask]
        
        # Diameter of unit circle config
        D1 = np.max(dists)
        
        # Log objective: sum(log(d)) - (n(n-1)/2) * log(D1)
        # We want to maximize this. So minimize negative.
        log_prod = np.sum(np.log(unique_dists))
        log_D1 = np.log(D1)
        
        return -(log_prod - (n * (n - 1) / 2.0) * log_D1)

    # Bounds for angles
    # To ensure alpha_8 > 0, sum(alpha_1..7) < 2*pi
    # Also alpha_i > 0
    # We can use bounds [1e-3, 2*pi - 7*1e-3] for each, but DE doesn't handle sum constraint well directly.
    # Instead, we can just use broad bounds and rely on the penalty or clipping.
    # Or better, use a transformation.
    # Let's just use DE with bounds [1e-4, 2*pi] and check validity.
    
    bounds = [(1e-4, 2 * np.pi)] * 7
    
    if time.time() < deadline - 10:
        try:
            de_res = differential_evolution(
                cyclic_objective,
                bounds=bounds,
                maxiter=500,
                tol=1e-10,
                seed=42,
                popsize=30,
                mutation=(0.5, 1.5),
                recombination=0.8,
                polish=True
            )
            
            if de_res.fun < 1e9: # Valid solution found
                alpha_vars = de_res.x
                alphas = np.zeros(n)
                alphas[:7] = alpha_vars
                alphas[7] = 2 * np.pi - np.sum(alpha_vars)
                
                if np.all(alphas > 1e-9):
                    psi = np.cumsum(alphas)
                    psi[0] = 0
                    pts_unit = np.column_stack([np.cos(psi), np.sin(psi)])
                    
                    # Calculate diameter and scale
                    diffs = pts_unit[:, np.newaxis, :] - pts_unit[np.newaxis, :, :]
                    dists = np.linalg.norm(diffs, axis=2)
                    D1 = np.max(dists)
                    R = 2.0 / D1
                    pts_scaled = pts_unit * R
                    
                    s = score_from_points(pts_scaled)
                    if s > best_score:
                        best_score = s
                        best_points = pts_scaled.copy()
                        
        except Exception as e:
            pass

    # Fallback: If cyclic search didn't improve, stick with best found (likely regular octagon)
    # We can also try a quick local refinement on the free coordinates if time permits, 
    # but the directive asks for the structural change. The cyclic representation is the change.
    
    return best_points.tolist()
