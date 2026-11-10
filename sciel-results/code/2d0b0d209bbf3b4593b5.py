# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random

def erdos_szekeres_problem7() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 620  # Safety margin
    
    n = 33
    WEIGHT = 3654.0 # C(29, 3)

    # Precompute all 4-combinations
    all_combinations = np.array(list(zip(*np.meshgrid(
        np.arange(n), np.arange(n), np.arange(n), np.arange(n), indexing='ij'
    )))).reshape(-1, 4)
    mask = (all_combinations[:, 0] < all_combinations[:, 1]) & \
           (all_combinations[:, 1] < all_combinations[:, 2]) & \
           (all_combinations[:, 2] < all_combinations[:, 3])
    all_combinations = all_combinations[mask]
    
    # Precompute combination indices for each point
    comb_indices = []
    for i in range(n):
        mask_i = np.any(all_combinations == i, axis=1)
        comb_indices.append(np.where(mask_i)[0])

    # Vectorized convexity check for 4-tuples
    # Returns boolean array: True if convex
    def is_convex_batch(points, indices):
        p0 = points[indices[:, 0]]
        p1 = points[indices[:, 1]]
        p2 = points[indices[:, 2]]
        p3 = points[indices[:, 3]]
        
        v1 = p1 - p0
        v2 = p2 - p0
        v3 = p3 - p0
        
        a012 = np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
        a013 = np.abs(v1[:, 0] * v3[:, 1] - v1[:, 1] * v3[:, 0])
        a023 = np.abs(v2[:, 0] * v3[:, 1] - v2[:, 1] * v3[:, 0])
        
        v21 = p2 - p1
        v31 = p3 - p1
        a123 = np.abs(v21[:, 0] * v31[:, 1] - v21[:, 1] * v31[:, 0])
        
        tol = 1e-8
        cond0 = np.abs(a123 - (a012 + a013 + a023)) < tol
        cond1 = np.abs(a023 - (a012 + a013 + a123)) < tol
        cond2 = np.abs(a013 - (a012 + a023 + a123)) < tol
        cond3 = np.abs(a012 - (a013 + a023 + a123)) < tol
        
        is_non_convex = cond0 | cond1 | cond2 | cond3
        return ~is_non_convex

    def is_valid(points):
        if points.shape != (n, 2):
            return False
        if np.any(np.isnan(points)) or np.any(np.isinf(points)):
            return False
        if np.any(points < -1e-6) or np.any(points > 1 + 1e-6):
            return False
        
        diffs = points[:, None, :] - points[None, :, :]
        dist_sq = np.sum(diffs**2, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        if np.min(dist_sq) < 1e-12:
            return False
        
        # Collinearity check
        # Vectorized triple check is complex, but 5456 is small.
        # However, we can vectorize the area calculation for all triples.
        # i, j, k
        i_idx, j_idx, k_idx = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing='ij')
        i_flat = i_idx.ravel()
        j_flat = j_idx.ravel()
        k_flat = k_idx.ravel()
        
        valid_triples = (i_flat < j_flat) & (j_flat < k_flat)
        i_v = i_flat[valid_triples]
        j_v = j_flat[valid_triples]
        k_v = k_flat[valid_triples]
        
        p1 = points[i_v]
        p2 = points[j_v]
        p3 = points[k_v]
        
        # Area2 = x1(y2-y3) + x2(y3-y1) + x3(y1-y2)
        areas = np.abs(p1[:, 0] * (p2[:, 1] - p3[:, 1]) + 
                       p2[:, 0] * (p3[:, 1] - p1[:, 1]) + 
                       p3[:, 0] * (p1[:, 1] - p2[:, 1]))
        
        if np.any(areas < 1e-9):
            return False
        return True

    def check_validity_fast(points, idx, new_p):
        diffs = points - new_p
        dist_sq = np.sum(diffs**2, axis=1)
        dist_sq[idx] = np.inf
        if np.min(dist_sq) < 1e-12:
            return False
            
        i_idx, j_idx = np.triu_indices(n, k=1)
        mask = (i_idx != idx) & (j_idx != idx)
        i_idx = i_idx[mask]
        j_idx = j_idx[mask]
        
        p_i = points[i_idx]
        p_j = points[j_idx]
        
        v1 = p_i - new_p
        v2 = p_j - new_p
        areas = np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
        
        if np.any(areas < 1e-9):
            return False
            
        return True

    # Grid initialization strategy
    # Use a 6x6 grid, pick 33 points.
    # To maximize collinearity, we want to keep the grid structure intact.
    # Removing 3 points from a 6x6 grid.
    
    best_points = None
    best_flags = None
    best_count = float('inf')
    
    # Grid coordinates
    grid_coords = np.array([[x, y] for y in np.linspace(0.1, 0.9, 6) for x in np.linspace(0.1, 0.9, 6)])
    
    # Try a few different removal patterns to break symmetry but keep density
    removal_patterns = [
        [0, 1, 2], # Top left
        [33, 34, 35], # Bottom right
        [17, 18, 19], # Middle row center
        [5, 11, 17], # Diagonal
        [10, 16, 22] # Other diagonal
    ]
    
    for pattern in removal_patterns:
        keep_mask = np.ones(36, dtype=bool)
        keep_mask[pattern] = False
        pts = grid_coords[keep_mask].copy()
        
        # Add tiny jitter to ensure validity (no exact collinearity if grid is too perfect? 
        # Wait, grid points ARE collinear. The validity check REQUIRES non-collinearity (area >= 1e-9).
        # So we MUST jitter.
        # But we want to preserve the "grid-like" structure.
        # Small jitter.
        pts += np.random.uniform(-0.001, 0.001, size=pts.shape)
        pts = np.clip(pts, 0.0, 1.0)
        
        if not is_valid(pts):
            continue
            
        flags = is_convex_batch(pts, all_combinations)
        count = np.sum(flags) * WEIGHT
        
        if count < best_count:
            best_count = count
            best_points = pts.copy()
            best_flags = flags.copy()
            
    if best_points is None:
        # Fallback random
        best_points = np.random.rand(n, 2) * 0.98 + 0.01
        if not is_valid(best_points):
             return np.random.rand(n, 2)
        best_flags = is_convex_batch(best_points, all_combinations)
        best_count = np.sum(best_flags) * WEIGHT

    # Optimization
    # Strategy: 
    # 1. Primary move: Snap to nearest grid point (if valid).
    # 2. Secondary move: Small random perturbation.
    
    T = 0.05
    T_min = 0.001
    cooling = 0.999
    step_size = 0.02
    
    # Precompute grid points for snapping
    # grid_coords is 36x2
    
    while time.time() < deadline:
        idx = random.randint(0, n - 1)
        old_p = best_points[idx]
        
        # Decide move type
        # 80% chance: try to snap to a nearby grid point
        # 20% chance: random perturbation
        
        if random.random() < 0.8:
            # Find 3 nearest grid points to old_p
            dists = np.sum((grid_coords - old_p)**2, axis=1)
            nearest_idxs = np.argsort(dists)[:5]
            
            # Try them in order of distance
            new_p = None
            for g_idx in nearest_idxs:
                candidate = grid_coords[g_idx].copy()
                # Add tiny jitter to candidate to ensure validity (since grid is collinear)
                candidate += np.random.uniform(-0.0005, 0.0005, 2)
                candidate = np.clip(candidate, 0.0, 1.0)
                
                if check_validity_fast(best_points, idx, candidate):
                    new_p = candidate
                    break
            
            if new_p is None:
                # If no grid point works, fall back to perturbation
                new_p = old_p + np.random.normal(0, step_size, 2)
                new_p = np.clip(new_p, 0.0, 1.0)
        else:
            new_p = old_p + np.random.normal(0, step_size, 2)
            new_p = np.clip(new_p, 0.0, 1.0)
            
        if not check_validity_fast(best_points, idx, new_p):
            continue
            
        comb_idx = comb_indices[idx]
        tuples = all_combinations[comb_idx]
        
        temp_pts = best_points.copy()
        temp_pts[idx] = new_p
        
        new_flags = is_convex_batch(temp_pts, tuples)
        old_flags = best_flags[comb_idx]
        
        delta_quads = np.sum(new_flags) - np.sum(old_flags)
        delta_count = delta_quads * WEIGHT
        
        if delta_count < 0 or random.random() < math.exp(-delta_count / T):
            best_points[idx] = new_p
            best_flags[comb_idx] = new_flags
            best_count += delta_count
            T = max(T_min, T * cooling)
        else:
            T = max(T_min, T * cooling)
            
        # Occasionally adjust step size
        if random.random() < 0.01:
            step_size = max(0.005, step_size * 0.9)

    return best_points

# EVOLVE-BLOCK-END
