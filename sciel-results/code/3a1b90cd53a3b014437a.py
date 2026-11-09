import numpy as np
import time
from scipy.spatial.distance import pdist

N = 300

# EVOLVE-BLOCK-START

def _get_score(points):
    """
    Computes the score (common radius r) for a given set of points.
    r = min( min_{i<j} ||c_i - c_j|| / 2, min_i (1 - ||c_i||) )
    """
    if points is None or len(points) != N:
        return 0.0
    
    if not np.all(np.isfinite(points)):
        return 0.0

    norms = np.linalg.norm(points, axis=1)
    boundary_r = 1.0 - np.max(norms)
    
    if boundary_r < 0:
        return 0.0

    d = pdist(points)
    if len(d) == 0:
        pair_r = float('inf')
    else:
        pair_r = np.min(d) / 2.0

    r = min(pair_r, boundary_r)
    return max(0.0, r)

def _hexagonal_lattice(spacing):
    """
    Generate a hexagonal lattice of points inside the unit disk.
    """
    k = int(1.5 / spacing) + 5
    pts = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if np.hypot(x, y) <= 1.0:
                pts.append((x, y))
    
    if len(pts) < N:
        return None
    
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists, kind="stable")[:N]
    return pts[idx]

def _refine(points, time_limit):
    """
    Refine the configuration using gradient ascent on active constraints.
    """
    start_time = time.time()
    best_score = _get_score(points)
    best_points = points.copy()
    
    # Adaptive step size
    step = 0.01
    
    while time.time() < start_time + time_limit:
        points = best_points.copy()
        
        # Compute pairwise distances
        d = pdist(points)
        if len(d) == 0:
            break
        min_d = np.min(d)
        pair_r = min_d / 2.0
        
        # Compute boundary clearance
        norms = np.linalg.norm(points, axis=1)
        max_norm = np.max(norms)
        boundary_r = 1.0 - max_norm
        
        current_r = min(pair_r, boundary_r)
        
        if current_r > best_score:
            best_score = current_r
            best_points = points.copy()
        
        # Determine active constraint
        # We need the indices of the active constraints
        
        # Find pair with min distance
        # pdist returns condensed distance matrix. 
        # i, j such that d[i,j] is min.
        # We can reconstruct or use a helper.
        # For efficiency, let's just find the min in the condensed matrix and map back.
        min_idx = np.argmin(d)
        # Map condensed index to (i, j)
        # n = N
        # i = floor((-1 + sqrt(1 + 8*min_idx)) / 2)
        # j = i + 1 + (min_idx - i*(2*n - i - 1)//2) -- this is messy.
        
        # Simpler: Reconstruct full matrix for N=300 is 90000 floats. Fast.
        # But we already have d.
        # Let's just use the gradient of the soft-min or active constraint.
        
        # If pair_r < boundary_r:
        #   Push the closest pair apart.
        # Else:
        #   Pull the outermost point in.
        
        # To find the pair efficiently without full matrix reconstruction every step:
        # We can store the indices from the previous step and check if they are still min.
        # But for simplicity and robustness with N=300, reconstruction is fine.
        
        # Reconstruct full distance matrix
        # scipy.spatial.distance.squareform
        from scipy.spatial.distance import squareform
        D = squareform(d)
        np.fill_diagonal(D, np.inf)
        
        # Find min pair
        min_val, min_idx_flat = np.unravel_index(np.argmin(D), D.shape)
        i, j = min_idx_flat
        
        # Find max norm point
        k = np.argmax(norms)
        
        if pair_r < boundary_r:
            # Internal constraint active
            # Move i away from j, and j away from i
            vec = points[i] - points[j]
            norm_vec = np.linalg.norm(vec)
            if norm_vec > 1e-9:
                grad_i = vec / norm_vec
                grad_j = -vec / norm_vec
            else:
                grad_i = np.array([1.0, 0.0])
                grad_j = -grad_i
            
            # Apply small repulsion to all points to prevent clustering
            # Only for points close to i or j?
            # For now, just update i and j
            points[i] += step * grad_i
            points[j] += step * grad_j
            
        else:
            # Boundary constraint active
            # Move k towards center
            vec = -points[k]
            norm_vec = np.linalg.norm(vec)
            if norm_vec > 1e-9:
                grad_k = vec / norm_vec
            else:
                grad_k = np.array([0.0, 0.0])
            
            points[k] += step * grad_k
            
        # Project all points into the unit circle
        # If a point is outside, move it to the boundary
        # This ensures validity
        norms_new = np.linalg.norm(points, axis=1)
        mask = norms_new > 1.0
        if np.any(mask):
            points[mask] = points[mask] / norms_new[mask]
            
        # Occasionally add noise to escape local minima
        if np.random.random() < 0.01:
            points += np.random.normal(0, step/10, points.shape)
            norms_new = np.linalg.norm(points, axis=1)
            mask = norms_new > 1.0
            if np.any(mask):
                points[mask] = points[mask] / norms_new[mask]
                
        # Decay step size slowly
        step *= 0.999

    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    candidates = []
    
    # 1. Generate initial candidates from hexagonal lattices with various spacings
    # Spacing corresponds to 2*r. 
    # Target r ~ 0.053 -> spacing ~ 0.106
    # Try a range of spacings
    spacings = np.linspace(0.05, 0.12, 15)
    
    for spacing in spacings:
        pts = _hexagonal_lattice(spacing)
        if pts is not None:
            r = _get_score(pts)
            if r > 0:
                candidates.append((pts, r))
                
    if not candidates:
        # Fallback: Random points
        angles = 2 * np.pi * np.random.rand(N)
        radii = np.sqrt(np.random.rand(N)) * 0.9
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        return pts
        
    # Sort candidates by initial score
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    best_points = candidates[0][0].copy()
    best_score = candidates[0][1]
    
    # 2. Refine the top candidate
    # Allocate most time to the best candidate
    time_remaining = deadline - time.time()
    if time_remaining > 5:
        refined_points, refined_score = _refine(best_points, time_remaining - 5)
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_points
    
    # 3. If time remains, try to refine the second best candidate
    if len(candidates) > 1 and time.time() < deadline - 5:
        time_remaining = deadline - time.time()
        if time_remaining > 5:
            refined_points2, refined_score2 = _refine(candidates[1][0].copy(), time_remaining - 5)
            if refined_score2 > best_score:
                best_score = refined_score2
                best_points = refined_points2
                
    return best_points

# EVOLVE-BLOCK-END
