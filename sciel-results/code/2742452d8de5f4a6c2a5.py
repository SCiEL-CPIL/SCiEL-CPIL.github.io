import numpy as np
import time
from scipy.optimize import minimize

N = 300

def construct():
    t0 = time.time()
    # 150s limit. Keep 10s margin.
    DEADLINE = t0 + 135.0

    def elapsed():
        return time.time() - t0

    def score(points):
        """
        Calculate the maximum radius for the given centers.
        points: (N, 2) array.
        """
        if points.shape[0] < 2:
            return 0.0
        
        # Calculate pairwise squared distances
        # Using broadcasting
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        
        # Set diagonal to infinity to ignore self-distance
        np.fill_diagonal(dist_sq, np.inf)
        
        min_dist_sq = np.min(dist_sq)
        min_dist = np.sqrt(min_dist_sq)
        
        # Calculate maximum distance from origin
        norms = np.linalg.norm(points, axis=1)
        max_norm = np.max(norms)
        
        # Radius is limited by both pair distances and boundary
        r_pair = 0.5 * min_dist
        r_bound = 1.0 - max_norm
        
        return min(r_pair, r_bound)

    def objective(params):
        """
        Objective function for minimization.
        We want to maximize score, so we minimize -score.
        """
        pts = params.reshape(N, 2)
        # Add a small penalty for non-finite values just in case
        if not np.all(np.isfinite(pts)):
            return 1e6
        s = score(pts)
        return -s

    def gen_hex_seed():
        """
        Generate a hexagonal lattice seed.
        """
        # Estimate initial spacing
        # r ~ sqrt(0.9 / N)
        r_est = np.sqrt(0.9 / N)
        spacing = 2.0 * r_est
        
        k = int(1.0 / spacing) + 3
        pts = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x*x + y*y <= 1.0:
                    pts.append([x, y])
        
        pts = np.array(pts)
        
        # If we have more than N points, select the ones closest to the center
        # This creates a "filled" disk shape
        if len(pts) > N:
            radii = np.linalg.norm(pts, axis=1)
            idx = np.argsort(radii)
            pts = pts[idx[:N]]
        elif len(pts) < N:
            # Shrink spacing and retry
            while len(pts) < N:
                spacing *= 0.95
                k = int(1.0 / spacing) + 3
                pts = []
                for j in range(-k, k + 1):
                    for i in range(-k, k + 1):
                        x = (i + 0.5 * (j % 2)) * spacing
                        y = j * spacing * np.sqrt(3.0) / 2.0
                        if x*x + y*y <= 1.0:
                            pts.append([x, y])
                pts = np.array(pts)
                if len(pts) > N:
                    radii = np.linalg.norm(pts, axis=1)
                    idx = np.argsort(radii)
                    pts = pts[idx[:N]]
        
        return pts

    # Initial Seed
    best_pts = gen_hex_seed()
    best_score = score(best_pts)
    
    # Optimization Loop
    # We use L-BFGS-B which is fast.
    # We perform multiple restarts to escape local minima.
    
    current_pts = best_pts.copy()
    
    while elapsed() < DEADLINE - 5:
        # Perturb the current solution for restart
        # Use a small noise to avoid getting stuck in the exact same local minimum
        noise_scale = 0.005
        current_pts = current_pts + np.random.normal(0, noise_scale, current_pts.shape)
        
        # Ensure points are roughly inside (clip to [-1, 1] to help optimizer)
        # The score function handles the actual constraint, but keeping them
        # in bounds helps the numerical gradient.
        current_pts = np.clip(current_pts, -1.0, 1.0)
        
        x0 = current_pts.flatten()
        
        try:
            # Run optimizer
            # maxiter=100 is usually enough for a local polish
            res = minimize(
                objective, 
                x0, 
                method='L-BFGS-B', 
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 20}
            )
            
            if res.x is not None:
                cand_pts = res.x.reshape(N, 2)
                
                # Validate
                if np.all(np.isfinite(cand_pts)):
                    cand_score = score(cand_pts)
                    
                    if cand_score > best_score:
                        best_score = cand_score
                        best_pts = cand_pts
                        # Update current to be the best found so far for next restart
                        # Or keep it as the result of this run?
                        # Using the best found is safer.
                        current_pts = best_pts.copy()
                        
        except Exception:
            pass

    return best_pts
