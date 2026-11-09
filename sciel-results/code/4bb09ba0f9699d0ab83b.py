"""Improved program using FCC lattice packing with local optimization."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize
import time

def _score(points):
    """
    Compute the score (radius) for a set of points.
    Higher is better.
    Returns -inf if invalid.
    """
    if len(points) != N:
        return -np.inf
    if not np.all(np.isfinite(points)):
        return -np.inf
    
    # Check if inside cube [-0.5, 0.5]
    # Distance to boundary
    dist_to_boundary = 0.5 - np.abs(points)
    min_boundary = np.min(dist_to_boundary)
    
    if min_boundary < -1e-9: # Allow tiny floating point error
        return -np.inf
    
    # Pairwise distances
    # Use cKDTree for efficiency
    tree = cKDTree(points)
    # We need the minimum distance among all pairs
    # query_pairs returns pairs within a given distance. 
    # We don't know the min distance, so we can estimate it or just compute all pairs if N is small.
    # N=100, so 100*99/2 = 4950 pairs. This is fast.
    # dists = tree.query_pairs(1.0) # 1.0 is large enough
    # But calculating norms for all pairs explicitly is also fast for N=100.
    
    # Explicit calculation for simplicity and speed in this specific case
    # dists = np.linalg.norm(points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=2)
    # np.fill_diagonal(dists, np.inf)
    # min_pair = np.min(dists)
    
    # Vectorized distance matrix
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    radius = min(min_pair / 2.0, min_boundary)
    return radius

def _neg_score(points_flat):
    """
    Objective function for minimization.
    We minimize -score.
    """
    points = points_flat.reshape(N, 3)
    s = _score(points)
    if s == -np.inf:
        return 1e6
    return -s

def construct():
    # Generate FCC lattice points
    # FCC lattice vectors:
    # a1 = (0, 1, 1) * a/2
    # a2 = (1, 0, 1) * a/2
    # a3 = (1, 1, 0) * a/2
    # Points are (i*a1 + j*a2 + k*a3) for integers i,j,k.
    # Alternatively, points are (x, y, z) where x,y,z are multiples of a/2 and x+y+z is an even multiple of a/2?
    # Standard FCC: 
    # (0,0,0), (0, a/2, a/2), (a/2, 0, a/2), (a/2, a/2, 0) in the unit cell of side a.
    # So coordinates are (i*a, j*a, k*a) + offsets.
    
    # Let's generate a grid of candidate FCC points.
    # We will try a range of lattice constants 'a' and shifts.
    
    best_points = None
    best_score_val = 0.0
    
    # Time limit check
    start_time = time.time()
    time_limit = 110.0 # Leave 10s buffer
    
    # Range of lattice constants
    # Theoretical max radius for 100 spheres in unit cube is ~0.111.
    # Min pair dist ~ 0.222.
    # FCC nearest neighbor dist = a / sqrt(2).
    # So a/sqrt(2) ~ 0.222 => a ~ 0.314.
    # Let's search a in [0.25, 0.35].
    
    a_values = np.linspace(0.28, 0.34, 13)
    
    # Shifts: dx, dy, dz in [0, a/2)
    # Symmetry suggests we only need to check shifts in [0, a/2]
    # To keep it fast, let's try a few discrete shifts or just 0.
    # Actually, optimizing the shift is part of the local refinement.
    # We can start with shift (0,0,0) and let the optimizer move the whole lattice?
    # No, the optimizer moves individual points.
    # A better initial guess is to shift the lattice so that the selected points are centered.
    
    for a in a_values:
        if time.time() - start_time > time_limit:
            break
            
        # Generate FCC points
        # Lattice constant a
        # Points: (i*a + ox, j*a + oy, k*a + oz) where (i,j,k) are integers
        # and (ox, oy, oz) are from {0, a/2} such that sum of indices is even?
        # Simpler: Generate all points (x,y,z) where x,y,z are in {0, a/2, a, 3a/2, ...}
        # and (x+y+z)/a is an integer? No.
        
        # Let's use the standard definition:
        # FCC points are (i*a, j*a, k*a) + (0,0,0)
        # (i*a, j*a, k*a) + (a/2, a/2, 0)
        # (i*a, j*a, k*a) + (a/2, 0, a/2)
        # (i*a, j*a, k*a) + (0, a/2, a/2)
        
        # We want points in [-0.5, 0.5].
        # Let's generate a range of i,j,k from -3 to 3.
        
        points_list = []
        offsets = [
            (0.0, 0.0, 0.0),
            (a/2.0, a/2.0, 0.0),
            (a/2.0, 0.0, a/2.0),
            (0.0, a/2.0, a/2.0)
        ]
        
        for i in range(-3, 4):
            for j in range(-3, 4):
                for k in range(-3, 4):
                    for ox, oy, oz in offsets:
                        x = i * a + ox
                        y = j * a + oy
                        z = k * a + oz
                        points_list.append((x, y, z))
        
        points_array = np.array(points_list)
        
        # Filter points inside the cube (with a small margin to be safe for selection, 
        # but we can select points slightly outside if we plan to move them in? 
        # No, the score function penalizes points outside. 
        # It's better to select points that are strictly inside or on the boundary.)
        
        mask = np.all(np.abs(points_array) <= 0.5, axis=1)
        candidates = points_array[mask]
        
        if len(candidates) < N:
            continue
            
        # Select the best N candidates.
        # Heuristic: Points with largest minimum distance to boundary are "safer".
        # However, we want to maximize the *global* minimum distance.
        # A good initial selection is the N points closest to the origin (center), 
        # as they are furthest from the boundary on average.
        # Or, select based on distance to boundary.
        
        dist_to_boundary = np.min(0.5 - np.abs(candidates), axis=1)
        # Sort by distance to boundary, descending
        order = np.argsort(dist_to_boundary)[::-1]
        
        # We might want to try a few different selection strategies or just this one.
        # Let's take the top N.
        selected = candidates[order[:N]]
        
        # Refine
        # Flatten for optimizer
        x0 = selected.flatten()
        
        # Use Nelder-Mead for local optimization
        # It doesn't require gradients and works well for this.
        try:
            res = minimize(_neg_score, x0, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.success or res.fun < _neg_score(x0):
                refined_points = res.x.reshape(N, 3)
                current_score = _score(refined_points)
                
                if current_score > best_score_val:
                    best_score_val = current_score
                    best_points = refined_points
        except Exception:
            pass
            
        # Check time
        if time.time() - start_time > time_limit:
            break

    # Fallback if optimization failed or found nothing
    if best_points is None or not np.all(np.isfinite(best_points)):
        # Simple cubic fallback
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_points = grid[:N]
        
    return best_points
# EVOLVE-BLOCK-END
