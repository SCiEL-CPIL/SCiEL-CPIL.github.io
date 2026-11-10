# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize

def min_max_dist_dim2_16() -> np.ndarray:
    n = 16
    start_time = time.time()
    deadline = start_time + 590  # Safety margin before 630s kill

    # We fix p[0] = (0,0) and p[1] = (1,0) to remove translation, rotation, and scale invariance.
    # We optimize the remaining 14 points (28 coordinates).
    
    def build_points(params):
        points = np.zeros((n, 2))
        points[0] = [0.0, 0.0]
        points[1] = [1.0, 0.0]
        # params shape is (14, 2)
        points[2:] = params
        return points

    def soft_min(dists, tau=0.1):
        # log-sum-exp trick for stability
        a = -dists / tau
        a_max = np.max(a)
        return -tau * (a_max + np.log(np.sum(np.exp(a - a_max)) / len(dists)))

    def soft_max(dists, tau=0.1):
        a = dists / tau
        a_max = np.max(a)
        return tau * (a_max + np.log(np.sum(np.exp(a - a_max)) / len(dists)))

    def objective(params):
        points = build_points(params)
        dists = pdist(points)
        d_min = soft_min(dists)
        d_max = soft_max(dists)
        
        if d_min <= 0 or d_max <= 0:
            return 1e10
            
        # Minimize log(d_max) - log(d_min)
        return np.log(d_max) - np.log(d_min)

    # Initial guess: Hexagonal lattice fragment
    # Generate a hexagonal lattice
    def hex_lattice_init():
        # Side length 1
        a = 1.0
        h = np.sqrt(3) / 2 * a
        pts = []
        # Center at (0,0) roughly, then we will shift/scale
        # Generate enough points and pick 16
        for i in range(-4, 5):
            for j in range(-4, 5):
                x = i * a + (j % 2) * a / 2
                y = j * h
                pts.append([x, y])
        pts = np.array(pts)
        # Pick 16 points that are close to each other to form a compact cluster
        # Calculate centroid
        centroid = np.mean(pts, axis=0)
        pts = pts - centroid
        # Sort by distance to centroid
        dists_to_cent = np.linalg.norm(pts, axis=1)
        idx = np.argsort(dists_to_cent)[:16]
        hex_pts = pts[idx]
        
        # We need to map these to the parameter space where p0=(0,0) and p1=(1,0).
        # This is a rigid transformation + scaling.
        # Find the two closest points in hex_pts to use as p0 and p1?
        # Or just use the first two?
        # Better: Find the pair with minimum distance, set that to 1.
        # Then rotate/translate.
        # This is complex. Simpler: Just use random initialization for DE, 
        # or use a grid initialization.
        
        # Let's just return a random initialization for DE, 
        # but DE is robust.
        # However, a good start helps.
        # Let's try to fit the hex lattice to p0=(0,0), p1=(1,0).
        # Find min dist pair in hex_pts
        d = pdist(hex_pts)
        min_d = np.min(d)
        # Find indices
        # This is getting complicated for a simple init.
        # Let's just use a random init for DE.
        return None

    # Bounds for the 14 free points
    # The cluster should be within a reasonable distance from origin.
    # If p0=(0,0) and p1=(1,0), the max distance is likely < 5.
    bounds = [(-5, 5)] * 28

    # Run Differential Evolution
    best_result = None
    try:
        de_res = differential_evolution(
            objective, 
            bounds=bounds, 
            seed=42, 
            maxiter=200, 
            tol=1e-8, 
            popsize=15,
            mutation=(0.5, 1.5),
            recombination=0.7
        )
        best_params = de_res.x
        best_score = de_res.fun
        best_result = build_points(best_params)
        
        # Check time
        if time.time() > deadline - 60:
            return best_result

        # Polish with L-BFGS-B
        res = minimize(
            objective, 
            best_params, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if res.fun < best_score:
            best_params = res.x
            best_score = res.fun
            best_result = build_points(best_params)

    except Exception as e:
        # Fallback to random if optimizer fails
        np.random.seed(42)
        params = np.random.uniform(-2, 2, (14, 2))
        best_result = build_points(params)

    # Final check and return
    # The return value must be shape (16, 2)
    return best_result

# EVOLVE-BLOCK-END
