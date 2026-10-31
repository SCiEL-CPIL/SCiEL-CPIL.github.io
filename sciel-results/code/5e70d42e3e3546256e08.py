import numpy as np
from scipy.optimize import minimize
import itertools

def construct():
    # Initial configuration: 16 points from a hexagonal lattice
    # This is the same setup as the seed, but we will optimize from here.
    
    sqrt3 = np.sqrt(3)
    
    # Generate 19 hex points
    points = []
    # Center
    points.append((0.0, 0.0))
    
    # Ring 1
    dirs = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    for q, r in dirs:
        x = q + r/2.0
        y = sqrt3/2.0 * r
        points.append((x, y))
        
    # Ring 2
    for q in range(-2, 3):
        for r in range(-2, 3):
            if max(abs(q), abs(r), abs(q+r)) == 2:
                x = q + r/2.0
                y = sqrt3/2.0 * r
                points.append((x, y))
                
    # points has 19 entries. Indices 0-18.
    # We want to pick 16. The seed program iterated over removing 3 from the 12 outer ones.
    # Let's just pick the best one quickly or use a heuristic.
    # Actually, we can just run the optimizer on the full 19 and let it collapse? No, we need exactly 16.
    
    # Let's find the best 16-subset of the 19 hex points to use as a starting point.
    best_min_max_dist = float('inf')
    best_subset = None
    
    outer_indices = list(range(7, 19))
    
    # To speed up, we can just pick a symmetric subset.
    # Removing 3 points from ring 2.
    # Let's just try all 220 combinations. It's fast.
    for remove_combo in itertools.combinations(outer_indices, 3):
        remove_set = set(remove_combo)
        current_points = []
        for i in range(19):
            if i not in remove_set:
                current_points.append(points[i])
                
        pts = np.array(current_points)
        # Calculate pairwise distances
        dists = []
        n = len(pts)
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(pts[i] - pts[j])
                dists.append(d)
        d_min = min(dists)
        d_max = max(dists)
        ratio_sq = (d_max / d_min)**2
        
        if ratio_sq < best_min_max_dist:
            best_min_max_dist = ratio_sq
            best_subset = pts.copy()
            
    # Now we have a good starting point `best_subset` (16x2).
    # We will use scipy.optimize.minimize to refine it.
    
    def objective(flat_pts):
        pts = flat_pts.reshape(16, 2)
        # Calculate pairwise distances
        # Using broadcasting for speed
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        # Mask out diagonal (0 distance)
        np.fill_diagonal(dists, np.inf)
        
        d_min = np.min(dists)
        d_max = np.max(dists)
        
        if d_min < 1e-6:
            return 1e10
            
        return (d_max / d_min)**2

    # Flatten the initial points
    x0 = best_subset.flatten()
    
    # Run optimization
    # Use Nelder-Mead or Powell. Powell is often better for this.
    # We might need multiple restarts or a good method.
    # Let's try Powell.
    
    res = minimize(objective, x0, method='Powell', options={'maxiter': 1000, 'ftol': 1e-12})
    
    if res.success:
        best_pts = res.x.reshape(16, 2)
    else:
        best_pts = best_subset
        
    # Verify distinctness
    # Check if any distance is too small
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 1e-5:
        return best_subset
        
    return best_pts
