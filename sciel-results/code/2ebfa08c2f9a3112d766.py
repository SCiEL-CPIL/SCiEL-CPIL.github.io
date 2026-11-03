"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.spatial.distance import pdist

def compute_radius(centers):
    """
    Compute the maximum radius for a given set of centers.
    r = min( min_{i<j} ||c_i - c_j|| / 2, min_{i,k} (0.5 - |c_ik|) )
    """
    if not np.all(np.isfinite(centers)):
        return -1.0
        
    # Calculate pairwise distances using scipy for speed
    min_pair_dist = np.min(pdist(centers))
    
    # Calculate distance to boundaries
    abs_centers = np.abs(centers)
    dists_to_boundary = 0.5 - abs_centers
    min_boundary_dist = np.min(dists_to_boundary)
    
    # The radius is the minimum of the half-pair-distance and the boundary distance
    r = min(min_pair_dist / 2.0, min_boundary_dist)
    return r

def local_refinement(centers, max_iters=500, step_init=0.01):
    """
    Gradient-free local refinement using steepest ascent on the min-distance.
    Identifies the bottleneck and moves spheres to alleviate it.
    """
    current_radius = compute_radius(centers)
    step = step_init
    
    for _ in range(max_iters):
        # Identify bottleneck
        min_pair_dist = np.min(pdist(centers))
        abs_centers = np.abs(centers)
        dists_to_boundary = 0.5 - abs_centers
        min_boundary_dist = np.min(dists_to_boundary)
        
        if min_pair_dist / 2.0 < min_boundary_dist:
            # Bottleneck is a pair
            # Find the pair
            dists = pdist(centers)
            min_idx = np.argmin(dists)
            # Convert condensed index to (i, j)
            # n=50, index k corresponds to i=k//n? No.
            # scipy condensed index for n points:
            # (0,1) is 0, (0,2) is 1, ..., (0, n-1) is n-2, (1,2) is n-1, etc.
            # Formula: k = i*n - i*(i+1)//2 + j - i - 1
            # We need inverse.
            # For N=50, we can just use a lookup or brute force if needed, 
            # but N is small. Let's just find the pair by scanning if needed, 
            # or use a helper.
            
            # Helper to get pair from condensed index
            def get_pair(k, n):
                # i is the first index
                # k = i*n - i*(i+1)/2 + j - i - 1
                # Approximation: i is small
                # Let's just find the min pair by brute force for simplicity and safety, 
                # as N=50 is small.
                min_d = np.inf
                min_i, min_j = 0, 1
                for i in range(n):
                    for j in range(i+1, n):
                        d = np.linalg.norm(centers[i] - centers[j])
                        if d < min_d:
                            min_d = d
                            min_i, min_j = i, j
                return min_i, min_j

            i, j = get_pair(min_idx, N)
            
            # Move i and j away from each other
            vec = centers[i] - centers[j]
            norm = np.linalg.norm(vec)
            if norm > 1e-9:
                vec /= norm
            else:
                vec = np.array([1.0, 0.0, 0.0])
            
            old_i = centers[i].copy()
            old_j = centers[j].copy()
            
            centers[i] += vec * step
            centers[j] -= vec * step
            
            # Clip to boundary to ensure validity, though this might reduce radius
            # Better to check radius after move
            new_radius = compute_radius(centers)
            
            if new_radius > current_radius:
                current_radius = new_radius
                # Keep step or increase slightly
                step = min(step * 1.1, 0.05)
            else:
                # Revert
                centers[i] = old_i
                centers[j] = old_j
                step *= 0.8
                if step < 1e-6:
                    break
        else:
            # Bottleneck is boundary
            # Find the point closest to boundary
            min_b_idx = np.argmin(min_boundary_dist)
            # Which coordinate?
            min_coord_val = dists_to_boundary[min_b_idx]
            
            # Determine direction to move away from boundary
            # If center is positive and close to 0.5, move negative.
            # If center is negative and close to -0.5, move positive.
            # Actually, dists_to_boundary is 0.5 - |c|.
            # If |c| is large, dist is small.
            # We want to decrease |c|.
            
            c = centers[min_b_idx]
            move_vec = -c / np.linalg.norm(c) if np.linalg.norm(c) > 1e-9 else np.array([1.0, 0.0, 0.0])
            
            old_c = centers[min_b_idx].copy()
            centers[min_b_idx] += move_vec * step
            
            new_radius = compute_radius(centers)
            
            if new_radius > current_radius:
                current_radius = new_radius
                step = min(step * 1.1, 0.05)
            else:
                centers[min_b_idx] = old_c
                step *= 0.8
                if step < 1e-6:
                    break

    return centers

def construct():
    """
    Use Simulated Annealing with Local Refinement to maximize the packing radius.
    """
    rng = np.random.default_rng(42)
    
    # Time management
    start_time = time.time()
    time_limit = 110.0  # Leave 10s buffer for overhead
    
    # State
    best_centers = None
    best_radius = -1.0
    
    # Parameters
    T_initial = 0.02
    T_final = 1e-6
    max_iter_per_restart = 20000  # Reduced to allow more restarts or refinement
    num_restarts = 30
    
    # Calculate cooling alpha for each restart
    alpha = (T_final / T_initial) ** (1.0 / max_iter_per_restart)
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Initialization
        # Use a jittered grid to ensure good initial spread
        # 4x4x4 grid = 64 points. Pick 50.
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid_points = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        
        # Shuffle and pick N
        indices = rng.permutation(len(grid_points))
        centers = grid_points[indices[:N]].copy()
        
        # Add small noise to break symmetry
        centers += rng.normal(0, 0.005, size=centers.shape)
        centers = np.clip(centers, -0.49, 0.49)
        
        current_radius = compute_radius(centers)
        
        if current_radius > best_radius:
            best_radius = current_radius
            best_centers = centers.copy()
            
        T = T_initial
        no_improve_counter = 0
        max_no_improve = 2000
        
        for i in range(max_iter_per_restart):
            if time.time() - start_time > time_limit:
                break
                
            # Select random sphere
            idx = rng.integers(0, N)
            
            # Random move
            # Step size scales with T
            step_size = 0.01 + 0.01 * (T / T_initial)
            delta = rng.normal(0, step_size, size=3)
            
            old_pos = centers[idx].copy()
            centers[idx] += delta
            
            new_radius = compute_radius(centers)
            
            # Acceptance
            if new_radius > current_radius:
                current_radius = new_radius
                no_improve_counter = 0
                if new_radius > best_radius:
                    best_radius = new_radius
                    best_centers = centers.copy()
            else:
                # Probabilistic acceptance
                delta_r = new_radius - current_radius
                # If T is very small, exp(delta_r/T) is 0 for negative delta_r
                if delta_r < 0 and T > 1e-9:
                    prob = math.exp(delta_r / T)
                    if rng.random() < prob:
                        current_radius = new_radius
                    else:
                        # Reject
                        centers[idx] = old_pos
                else:
                    # Reject
                    centers[idx] = old_pos
                no_improve_counter += 1
            
            # Cool down
            T *= alpha
            
            if no_improve_counter > max_no_improve:
                break
        
        # Local Refinement after SA
        if time.time() - start_time < time_limit:
            refined_centers = local_refinement(centers, max_iters=200, step_init=0.005)
            refined_radius = compute_radius(refined_centers)
            if refined_radius > best_radius:
                best_radius = refined_radius
                best_centers = refined_centers.copy()

    return best_centers
# EVOLVE-BLOCK-END
