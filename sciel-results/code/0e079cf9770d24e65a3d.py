import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 300

def construct():
    t0 = time.time()
    # 150s limit. Keep a large safety margin (15s) to avoid timeout.
    DEADLINE = t0 + 130.0

    def elapsed():
        return time.time() - t0

    def score(points):
        """
        Calculate the maximum radius for the given centers.
        points: (N, 2) array.
        """
        if points.shape[0] < 2:
            return 0.0
        
        # Use cdist for faster pairwise distance calculation
        dist = cdist(points, points)
        
        # Set diagonal to infinity to ignore self-distance
        np.fill_diagonal(dist, np.inf)
        
        min_dist = np.min(dist)
        
        # Calculate maximum distance from origin
        norms = np.linalg.norm(points, axis=1)
        max_norm = np.max(norms)
        
        # Radius is limited by both pair distances and boundary
        r_pair = 0.5 * min_dist
        r_bound = 1.0 - max_norm
        
        return min(r_pair, r_bound)

    def objective(x):
        """
        Objective function for optimization.
        We minimize the sum of inverse distances to maximize separation.
        Also penalize points outside the unit circle.
        """
        pts = x.reshape(N, 2)
        
        # Calculate pairwise distances
        dist = cdist(pts, pts)
        np.fill_diagonal(dist, np.inf)
        
        # Avoid division by zero
        dist = np.maximum(dist, 1e-10)
        
        # Sum of inverse distances (repulsion)
        # Using power 2 is standard, higher powers make it steeper
        repulsion = np.sum(1.0 / (dist**2))
        
        # Boundary penalty
        norms = np.linalg.norm(pts, axis=1)
        boundary_violation = np.maximum(0, norms - 1.0)
        penalty = np.sum(boundary_violation**2)
        
        return repulsion + 100.0 * penalty

    def gen_ring_seed():
        """
        Generate a ring-based seed.
        Distribute points on concentric rings.
        """
        pts = []
        # Number of rings
        # Approximate number of rings needed
        # Area of circle ~ pi * r^2. 
        # Each circle area ~ pi * (r_c)^2.
        # N * pi * r_c^2 ~ pi * 1^2 => r_c ~ 1/sqrt(N)
        # Number of rings ~ 1 / r_c ~ sqrt(N) ~ 17
        num_rings = 18
        
        for k in range(1, num_rings + 1):
            r_ring = k / (num_rings + 0.5)
            # Number of points on this ring
            # Circumference ~ 2 * pi * r_ring
            # Spacing ~ 2 * r_c
            # n_pts ~ 2 * pi * r_ring / (2 * r_c)
            r_c_est = 1.0 / np.sqrt(N)
            n_pts = int(2 * np.pi * r_ring / (2 * r_c_est))
            
            if n_pts < 1:
                n_pts = 1
            
            # Angular positions
            angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
            # Add a small offset for alternating rings to improve packing
            offset = (k % 2) * np.pi / n_pts
            
            for a in angles:
                x = r_ring * np.cos(a + offset)
                y = r_ring * np.sin(a + offset)
                pts.append([x, y])
        
        pts = np.array(pts)
        
        # If we have more than N points, select the ones closest to the center
        if len(pts) > N:
            radii = np.linalg.norm(pts, axis=1)
            idx = np.argsort(radii)
            pts = pts[idx[:N]]
        elif len(pts) < N:
            # Add random points or adjust
            # For simplicity, just pad with random points inside
            missing = N - len(pts)
            random_pts = np.random.uniform(-1, 1, (missing, 2))
            # Ensure they are inside
            norms = np.linalg.norm(random_pts, axis=1)
            random_pts = random_pts / np.maximum(norms, 1e-6) * 0.9
            pts = np.vstack([pts, random_pts])
        
        return pts

    # Initial Seed
    # Use ring seed
    best_pts = gen_ring_seed()
    best_score = score(best_pts)
    
    # Optimization Loop
    # We use L-BFGS-B on the repulsion objective.
    
    current_pts = best_pts.copy()
    
    # Number of restarts
    num_restarts = 20
    
    for i in range(num_restarts):
        if elapsed() > DEADLINE - 10.0:
            break
            
        # Perturb the current solution for restart
        # Use smaller noise for later restarts
        noise_scale = 0.05 / (1 + i)
        current_pts = current_pts + np.random.normal(0, noise_scale, current_pts.shape)
        
        # Ensure points are roughly inside
        norms = np.linalg.norm(current_pts, axis=1)
        mask = norms > 1.0
        if np.any(mask):
            current_pts[mask] = current_pts[mask] / norms[mask] * 0.99
        
        x0 = current_pts.flatten()
        
        try:
            # Run optimizer
            # maxiter is limited to ensure we don't run too long per restart
            # We have ~130s / 20 restarts = 6.5s per restart
            # L-BFGS-B is fast.
            res = minimize(
                objective, 
                x0, 
                method='L-BFGS-B', 
                options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            
            if res.x is not None:
                cand_pts = res.x.reshape(N, 2)
                
                # Validate
                if np.all(np.isfinite(cand_pts)):
                    cand_score = score(cand_pts)
                    
                    if cand_score > best_score:
                        best_score = cand_score
                        best_pts = cand_pts
                        current_pts = best_pts.copy()
                        
        except Exception:
            pass

    return best_pts
