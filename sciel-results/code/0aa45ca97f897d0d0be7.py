"""Improved program for packing 100 equal spheres in a unit cube using simulated annealing."""
import numpy as np
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START
import numpy as np
import random
import math
from scipy.spatial.distance import cdist

def _compute_radius(c):
    """
    Compute the maximum radius for a given configuration c (N, 3).
    Returns the minimum of:
    1. Half the minimum pairwise distance.
    2. The minimum distance from any center to the cube boundary.
    """
    # Pairwise distances
    # cdist is efficient for small N
    dists = cdist(c, c)
    # Get upper triangle indices to avoid self-distance and double counting
    i, j = np.triu_indices(N, k=1)
    min_pair_dist = dists[i, j].min()
    
    # Boundary distances
    # Distance to nearest face for each point
    # Faces are at +/- 0.5
    # dist = min(0.5 - x, 0.5 + x, ...) 
    # Since c is in [-0.5, 0.5], abs(c) is in [0, 0.5]
    # dist to nearest face along axis k is 0.5 - abs(c[:, k])
    abs_c = np.abs(c)
    min_boundary_dist = (0.5 - abs_c).min()
    
    return min(min_pair_dist / 2.0, min_boundary_dist)

def _initial_grid():
    """
    Generate an initial grid configuration.
    For N=100, a 5x5x4 grid (or similar) fits well.
    We'll use a 5x5x5 grid and take the first 100 points, or adjust spacing.
    A 5x5x4 grid has exactly 100 points.
    """
    # 5x5x4 grid
    nx, ny, nz = 5, 5, 4
    # Spacing to fill [-0.5, 0.5]
    # Centers at -0.5 + (i+0.5)*dx
    dx = 1.0 / nx
    dy = 1.0 / ny
    dz = 1.0 / nz
    
    centers = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                x = -0.5 + (i + 0.5) * dx
                y = -0.5 + (j + 0.5) * dy
                z = -0.5 + (k + 0.5) * dz
                centers.append([x, y, z])
    
    c = np.array(centers[:N])
    return c

def construct():
    """
    Construct the optimal arrangement of 100 equal spheres in a unit cube.
    Uses Simulated Annealing with hybrid moves (random walk, swap, perturbation).
    """
    # Initialize
    c = _initial_grid()
    current_score = _compute_radius(c)
    best_c = c.copy()
    best_score = current_score
    
    # Annealing parameters
    T_start = 0.05
    T_end = 1e-6
    alpha = 0.9995 # Cooling factor
    max_iters = 200000 # Enough iterations to converge within time limit
    # 200k iterations * ~10us/iter (approx for cdist N=100) = 2 seconds. Safe.
    
    T = T_start
    
    # Precompute some constants
    # We will use random moves:
    # 1. Move one random point by a small Gaussian step
    # 2. Swap two random points
    # 3. Move two random points slightly (cooperative move)
    
    for it in range(max_iters):
        move_type = random.random()
        
        c_new = c.copy()
        
        if move_type < 0.6:
            # Move one point
            idx = random.randint(0, N - 1)
            step_size = 0.02 * (T / T_start + 0.1) # Scaled by temperature
            delta = np.random.randn(3) * step_size
            c_new[idx] = c_new[idx] + delta
            # Clamp to cube
            c_new[idx] = np.clip(c_new[idx], -0.5, 0.5)
            
        elif move_type < 0.8:
            # Swap two points
            i, j = random.sample(range(N), 2)
            c_new[i], c_new[j] = c_new[j].copy(), c_new[i].copy()
            
        else:
            # Move two points (cooperative)
            i, j = random.sample(range(N), 2)
            step_size = 0.01 * (T / T_start + 0.1)
            c_new[i] = c_new[i] + np.random.randn(3) * step_size
            c_new[j] = c_new[j] + np.random.randn(3) * step_size
            c_new[i] = np.clip(c_new[i], -0.5, 0.5)
            c_new[j] = np.clip(c_new[j], -0.5, 0.5)
            
        new_score = _compute_radius(c_new)
        
        # Acceptance criterion
        if new_score >= current_score:
            # Always accept improvement
            c = c_new
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_c = c.copy()
        else:
            # Accept with probability exp((new - curr) / T)
            delta_score = new_score - current_score
            # If delta is very negative and T is small, prob is 0
            if T > 1e-10:
                prob = math.exp(delta_score / T)
                if random.random() < prob:
                    c = c_new
                    current_score = new_score
        
        # Cool down
        T *= alpha
        
        # Optional: If T is very low, we can break early if no improvement for a long time
        # But for simplicity and robustness, we run full schedule.
        
    # Final local refinement: Greedy push of overlapping pairs
    # This helps squeeze out the last bit of radius
    for _ in range(500):
        r_pair = None
        r_bound = None
        dists = cdist(best_c, best_c)
        i, j = np.triu_indices(N, k=1)
        min_pair_dist = dists[i, j].min()
        min_pair_idx = np.argmin(dists[i, j])
        
        abs_c = np.abs(best_c)
        min_boundary_dist = (0.5 - abs_c).min()
        
        current_r = min(min_pair_dist / 2.0, min_boundary_dist)
        
        if min_pair_dist / 2.0 <= min_boundary_dist:
            # Pairwise limit
            pi, pj = i[min_pair_idx], j[min_pair_idx]
            diff = best_c[pi] - best_c[pj]
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                direction = np.random.randn(3)
                direction /= np.linalg.norm(direction)
            else:
                direction = diff / dist
                
            # Push apart
            move = 0.001
            c_test = best_c.copy()
            c_test[pi] = np.clip(c_test[pi] + direction * move, -0.5, 0.5)
            c_test[pj] = np.clip(c_test[pj] - direction * move, -0.5, 0.5)
            
            new_r = _compute_radius(c_test)
            if new_r > current_r:
                best_c = c_test
        else:
            # Boundary limit
            # Find point closest to boundary
            # dists to faces
            d_x = 0.5 - np.abs(best_c[:, 0])
            d_y = 0.5 - np.abs(best_c[:, 1])
            d_z = 0.5 - np.abs(best_c[:, 2])
            all_dists = np.column_stack([d_x, d_y, d_z])
            min_val = np.min(all_dists)
            min_idx = np.argmin(all_dists.flatten())
            pt_idx = min_idx // 3
            axis = min_idx % 3
            
            sign = 1 if best_c[pt_idx, axis] > 0 else -1
            c_test = best_c.copy()
            c_test[pt_idx, axis] = np.clip(c_test[pt_idx, axis] + sign * 0.001, -0.5, 0.5)
            
            new_r = _compute_radius(c_test)
            if new_r > current_r:
                best_c = c_test
            else:
                break # If we can't improve boundary, likely stuck or optimal locally
        
    return best_c
# EVOLVE-BLOCK-END
