"""Improved program for packing 100 equal spheres in a unit cube using simulated annealing."""
import numpy as np
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START
import numpy as np
import random
import math

def _compute_score(c):
    """
    Compute the packing score (minimum radius) for a configuration c.
    c: (N, 3) array of centers.
    Returns: min(min_pair_dist/2, min_boundary_dist)
    """
    # Pairwise distances
    # Use broadcasting to compute all pairwise distances
    # diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    # dists = np.sqrt(np.sum(diff**2, axis=2))
    # This creates an (N, N) matrix. For N=100, this is 10,000 floats. Very fast.
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Zero out diagonal to ignore self-distance
    np.fill_diagonal(dists, np.inf)
    
    min_pair_dist = np.min(dists)
    
    # Boundary distances
    # Distance to nearest face for each point
    # Face distances for x: 0.5 - |x|
    # Face distances for y: 0.5 - |y|
    # Face distances for z: 0.5 - |z|
    abs_c = np.abs(c)
    dists_x = 0.5 - abs_c[:, 0]
    dists_y = 0.5 - abs_c[:, 1]
    dists_z = 0.5 - abs_c[:, 2]
    
    # The distance to the boundary for a point is the min of its distances to the 6 faces.
    # Since the cube is symmetric and centered, dist to +x face is 0.5-x, to -x face is 0.5+x.
    # min(0.5-x, 0.5+x) = 0.5 - |x|.
    boundary_dists = np.min(np.column_stack([dists_x, dists_y, dists_z]), axis=1)
    min_boundary_dist = np.min(boundary_dists)
    
    return min(min_pair_dist / 2.0, min_boundary_dist)

def _simulated_annealing(c_init, max_iters, T0, T1, alpha, step_scale, verbose=False):
    """
    Run Simulated Annealing.
    """
    c = c_init.copy()
    current_score = _compute_score(c)
    best_c = c.copy()
    best_score = current_score
    
    T = T0
    step = step_scale
    
    for it in range(max_iters):
        # Generate move
        # Move 1 or 2 points
        n_move = 1 if random.random() < 0.8 else 2
        idxs = random.sample(range(N), n_move)
        
        c_new = c.copy()
        for idx in idxs:
            # Gaussian perturbation
            delta = np.random.randn(3) * step
            c_new[idx] += delta
            # Clamp to cube
            c_new[idx] = np.clip(c_new[idx], -0.5, 0.5)
        
        new_score = _compute_score(c_new)
        
        # Acceptance
        if new_score >= current_score:
            c = c_new
            current_score = new_score
        else:
            # Metropolis criterion
            delta_score = new_score - current_score
            # Avoid underflow in exp
            if delta_score > -50 * T: 
                prob = math.exp(delta_score / T)
                if random.random() < prob:
                    c = c_new
                    current_score = new_score
        
        if current_score > best_score:
            best_score = current_score
            best_c = c.copy()
            
        # Update temperature
        T *= alpha
        if T < T1:
            break
            
        # Adaptive step size: if we've been stuck, increase step; if improving, decrease
        # Simple heuristic: if no improvement in last 500 iters, increase step slightly
        # For simplicity, we'll just decay step size slightly with T
        step = step_scale * (T / T0 + 0.1)

    return best_c, best_score

def _local_refinement(c, max_iters=2000):
    """
    Deterministic local search to fix local minima.
    Focuses on the most violated constraints.
    """
    c = c.copy()
    current_score = _compute_score(c)
    best_c = c.copy()
    best_score = current_score
    
    for it in range(max_iters):
        # Compute constraints
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Find min pair
        min_pair_idx = np.argmin(dists)
        i, j = np.unravel_index(min_pair_idx, (N, N))
        min_pair_dist = dists[i, j]
        
        # Find min boundary
        abs_c = np.abs(c)
        boundary_dists = np.min(np.column_stack([
            0.5 - abs_c[:, 0], 0.5 + abs_c[:, 0],
            0.5 - abs_c[:, 1], 0.5 + abs_c[:, 1],
            0.5 - abs_c[:, 2], 0.5 + abs_c[:, 2]
        ]), axis=1)
        min_bound_idx = np.argmin(boundary_dists)
        min_bound_dist = boundary_dists[min_bound_idx]
        
        r_pair = min_pair_dist / 2.0
        r_bound = min_bound_dist
        
        if r_pair <= r_bound:
            # Pair is limiting
            # Try to move i and j apart
            v = c[i] - c[j]
            norm_v = np.linalg.norm(v)
            if norm_v < 1e-10:
                v = np.random.randn(3)
                norm_v = np.linalg.norm(v)
            v = v / norm_v
            
            # Move i in direction v, j in -v
            move_size = 0.005
            c_new = c.copy()
            c_new[i] = np.clip(c_new[i] + v * move_size, -0.5, 0.5)
            c_new[j] = np.clip(c_new[j] - v * move_size, -0.5, 0.5)
            
            new_score = _compute_score(c_new)
            if new_score > current_score:
                c = c_new
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_c = c.copy()
            else:
                # Try smaller move
                move_size = 0.001
                c_new = c.copy()
                c_new[i] = np.clip(c_new[i] + v * move_size, -0.5, 0.5)
                c_new[j] = np.clip(c_new[j] - v * move_size, -0.5, 0.5)
                new_score = _compute_score(c_new)
                if new_score > current_score:
                    c = c_new
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_c = c.copy()
        else:
            # Boundary is limiting
            idx = min_bound_idx
            # Determine which face is closest
            # We stored 6 distances per point. 
            # 0: -x, 1: +x, 2: -y, 3: +y, 4: -z, 5: +z
            # Wait, my column_stack order was:
            # 0: 0.5 - |x| (this is min(0.5-x, 0.5+x) effectively, but we need direction)
            # Let's re-calculate direction properly.
            x, y, z = c[idx]
            
            # Distance to -x face: x - (-0.5) = x + 0.5
            # Distance to +x face: 0.5 - x
            # etc.
            dist_neg_x = x + 0.5
            dist_pos_x = 0.5 - x
            dist_neg_y = y + 0.5
            dist_pos_y = 0.5 - y
            dist_neg_z = z + 0.5
            dist_pos_z = 0.5 - z
            
            min_d = min(dist_neg_x, dist_pos_x, dist_neg_y, dist_pos_y, dist_neg_z, dist_pos_z)
            
            move_size = 0.005
            c_new = c.copy()
            if min_d == dist_neg_x:
                c_new[idx, 0] = np.clip(c_new[idx, 0] + move_size, -0.5, 0.5)
            elif min_d == dist_pos_x:
                c_new[idx, 0] = np.clip(c_new[idx, 0] - move_size, -0.5, 0.5)
            elif min_d == dist_neg_y:
                c_new[idx, 1] = np.clip(c_new[idx, 1] + move_size, -0.5, 0.5)
            elif min_d == dist_pos_y:
                c_new[idx, 1] = np.clip(c_new[idx, 1] - move_size, -0.5, 0.5)
            elif min_d == dist_neg_z:
                c_new[idx, 2] = np.clip(c_new[idx, 2] + move_size, -0.5, 0.5)
            elif min_d == dist_pos_z:
                c_new[idx, 2] = np.clip(c_new[idx, 2] - move_size, -0.5, 0.5)
                
            new_score = _compute_score(c_new)
            if new_score > current_score:
                c = c_new
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_c = c.copy()
            else:
                move_size = 0.001
                c_new = c.copy()
                if min_d == dist_neg_x:
                    c_new[idx, 0] = np.clip(c_new[idx, 0] + move_size, -0.5, 0.5)
                elif min_d == dist_pos_x:
                    c_new[idx, 0] = np.clip(c_new[idx, 0] - move_size, -0.5, 0.5)
                elif min_d == dist_neg_y:
                    c_new[idx, 1] = np.clip(c_new[idx, 1] + move_size, -0.5, 0.5)
                elif min_d == dist_pos_y:
                    c_new[idx, 1] = np.clip(c_new[idx, 1] - move_size, -0.5, 0.5)
                elif min_d == dist_neg_z:
                    c_new[idx, 2] = np.clip(c_new[idx, 2] + move_size, -0.5, 0.5)
                elif min_d == dist_pos_z:
                    c_new[idx, 2] = np.clip(c_new[idx, 2] - move_size, -0.5, 0.5)
                    
                new_score = _compute_score(c_new)
                if new_score > current_score:
                    c = c_new
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_c = c.copy()
                    
    return best_c, best_score

def construct():
    """
    Construct the optimal arrangement.
    Uses multiple random restarts of Simulated Annealing followed by local refinement.
    """
    best_c = None
    best_score = 0.0
    
    # Number of restarts
    n_restarts = 5
    
    for restart in range(n_restarts):
        # Random initialization
        c_init = np.random.uniform(-0.5, 0.5, (N, 3))
        
        # Run SA
        c_sa, score_sa = _simulated_annealing(
            c_init, 
            max_iters=20000, 
            T0=0.05, 
            T1=1e-6, 
            alpha=0.999, 
            step_scale=0.05
        )
        
        # Local Refinement
        c_ref, score_ref = _local_refinement(c_sa, max_iters=1000)
        
        if score_ref > best_score:
            best_score = score_ref
            best_c = c_ref.copy()
            
        # Optional: Second round of SA from refined state with lower T
        if score_ref > 0.1: # Only if we have a decent start
            c_sa2, score_sa2 = _simulated_annealing(
                c_ref, 
                max_iters=10000, 
                T0=0.01, 
                T1=1e-7, 
                alpha=0.9995, 
                step_scale=0.01
            )
            c_ref2, score_ref2 = _local_refinement(c_sa2, max_iters=500)
            if score_ref2 > best_score:
                best_score = score_ref2
                best_c = c_ref2.copy()

    return best_c
# EVOLVE-BLOCK-END
