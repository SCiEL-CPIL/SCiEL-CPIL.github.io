"""Rewritten program to maximize minimum triangle area over convex hull area for N=14."""
import numpy as np
import scipy.spatial
import scipy.optimize
import time
from itertools import combinations

N = 14

def construct():
    start_time = time.time()
    time_limit = 120.0  # Leave margin for evaluation overhead

    # Initial guess: Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    points = np.vstack([np.cos(angles), np.sin(angles)]).T

    # Helper to calculate the score
    def get_score(pts):
        # Check for degeneracy
        if len(np.unique(np.round(pts, decimals=8), axis=0)) < N:
            return 0.0
            
        # Calculate min triangle area
        # Vectorized approach for speed
        n = len(pts)
        # Generate indices for all unique triples i < j < k
        # For N=14, this is 364 triples. 
        # Using combinations is simple and fast enough for 14 points.
        # Vectorizing with meshgrid creates 14^3 = 2744 points, filtering to 364.
        # This is very fast in numpy.
        
        idx = np.arange(n)
        # i, j, k
        i, j, k = np.meshgrid(idx, idx, idx, indexing='ij')
        mask = (i < j) & (j < k)
        i, j, k = i[mask], j[mask], k[mask]
        
        pi = pts[i]
        pj = pts[j]
        pk = pts[k]
        
        v1 = pj - pi
        v2 = pk - pi
        
        # Cross product magnitude (z-component)
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        min_area = np.min(areas)
        
        # Calculate convex hull area
        try:
            hull = scipy.spatial.ConvexHull(pts)
            hull_area = hull.volume
        except Exception:
            return 0.0
            
        if hull_area <= 1e-10:
            return 0.0
            
        return min_area / hull_area

    def objective(pts):
        return -get_score(pts)

    best_pts = points
    best_score = get_score(points)
    
    # Optimization settings
    options = {
        'xatol': 1e-9, 
        'fatol': 1e-9, 
        'maxiter': 2000, 
        'maxfev': 2000,
        'adaptive': True
    }
    
    # Multi-start optimization
    # We try the regular polygon and several perturbations
    restarts = 5
    
    for r in range(restarts):
        if time.time() - start_time > time_limit:
            break
            
        if r == 0:
            current_pts = points.copy()
        else:
            # Add noise to the regular polygon
            # Noise scale should be small enough to stay in convex position
            # but large enough to escape local minima
            noise_scale = 0.05 * (r + 1) / 5.0
            noise = np.random.normal(0, noise_scale, size=(N, 2))
            current_pts = points + noise
            
            # Re-center
            current_pts -= np.mean(current_pts, axis=0)
            
            # Normalize scale to keep numbers stable (score is scale invariant)
            dists = np.linalg.norm(current_pts, axis=1)
            if np.min(dists) > 1e-6:
                current_pts = current_pts / np.mean(dists)
        
        # Run Nelder-Mead
        try:
            res = scipy.optimize.minimize(
                objective, 
                current_pts, 
                method='Nelder-Mead', 
                options=options
            )
            
            # Check if the result is better
            curr_score = -res.fun
            if curr_score > best_score:
                best_score = curr_score
                best_pts = res.x
        except Exception:
            continue

    return best_pts
