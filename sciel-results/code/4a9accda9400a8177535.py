"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard: 120s limit, leave 10s margin for overhead/safety
    deadline = time.time() + 110.0
    
    # 1. Initialization: 4x4x4 grid, take first 50
    # The grid points are spaced by 0.25.
    # Coordinates: -0.375, -0.125, 0.125, 0.375
    ticks = np.array([-0.375, -0.125, 0.125, 0.375])
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    # Helper function to calculate the score (minimum radius)
    def calc_score(centers):
        # Min distance to walls
        # centers shape (N, 3)
        # wall dist for each axis
        d_x = 0.5 - np.abs(centers[:, 0])
        d_y = 0.5 - np.abs(centers[:, 1])
        d_z = 0.5 - np.abs(centers[:, 2])
        min_wall = np.min(np.min(np.stack([d_x, d_y, d_z]), axis=1))
        
        # Min distance between centers
        # Calculate pairwise distances
        # To save memory/time for N=50, we can just do it directly
        # dists[i, j] = |c_i - c_j|
        diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        # Set diagonal to inf
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        return min(min_wall, min_pair / 2.0)
    
    best_centers = centers.copy()
    best_score = calc_score(best_centers)
    current_centers = centers.copy()
    
    # Optimization parameters
    step_size = 0.05
    max_iterations = 20000
    
    # We will use a greedy local search:
    # For each point, move it in the direction that increases the minimum local distance.
    # Local minimum distance for point i is min(dist_to_nearest_neighbor, dist_to_nearest_wall)
    
    for it in range(max_iterations):
        if time.time() > deadline:
            break
            
        # Calculate score for current
        curr_score = calc_score(current_centers)
        if curr_score > best_score:
            best_score = curr_score
            best_centers = current_centers.copy()
            
        # Perform one step of local optimization
        # We update points one by one or in parallel?
        # Parallel update can cause oscillations. Sequential is safer but slower.
        # Let's try sequential with a small step.
        
        # To speed up, we can compute the "force" for all points at once based on current state,
        # then apply them. This is like a gradient step.
        
        # 1. Find nearest neighbor for each point
        diffs = current_centers[:, np.newaxis, :] - current_centers[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Index of nearest neighbor
        nn_idx = np.argmin(dists, axis=1)
        nn_dists = dists[np.arange(N), nn_idx]
        
        # 2. Find nearest wall for each point
        # d_x = 0.5 - |x|, etc.
        # The wall closest to point i is the one with min d_k.
        # Direction to move away from wall: sign of coordinate
        d_x = 0.5 - np.abs(current_centers[:, 0])
        d_y = 0.5 - np.abs(current_centers[:, 1])
        d_z = 0.5 - np.abs(current_centers[:, 2])
        
        # Stack wall distances
        wall_dists = np.stack([d_x, d_y, d_z], axis=1) # Shape (N, 3)
        min_wall_dist = np.min(wall_dists, axis=1)
        
        # Determine for each point if it is constrained by neighbor or wall
        # If min_wall_dist < nn_dist, constraint is wall. Move towards center.
        # Else, constraint is neighbor. Move away from neighbor.
        
        move_vectors = np.zeros_like(current_centers)
        
        # Case 1: Constrained by wall
        # Identify points where wall is closer
        wall_constrained = min_wall_dist < nn_dists
        
        for i in range(N):
            if wall_constrained[i]:
                # Move towards center in the axis where it is closest to wall
                # Find which axis is closest
                min_axis = np.argmin(wall_dists[i])
                # Direction is towards 0
                # If coord is positive, move negative. If negative, move positive.
                # So direction is -sign(coord)
                dir = -np.sign(current_centers[i, min_axis])
                if dir == 0:
                    dir = 1.0 # If exactly 0, pick arbitrary, though unlikely
                move_vectors[i, min_axis] += dir * step_size
            else:
                # Constrained by neighbor
                j = nn_idx[i]
                # Vector from j to i
                v = current_centers[i] - current_centers[j]
                dist = nn_dists[i]
                if dist > 1e-9:
                    v_norm = v / dist
                    move_vectors[i] += v_norm * step_size
                else:
                    # Coincident, random kick
                    move_vectors[i] += np.random.randn(3) * step_size
        
        # Apply moves
        current_centers += move_vectors
        
        # Clip to bounds (strictly inside)
        # The score function handles boundaries, but keeping them inside helps stability
        # Actually, if we clip, we might stay stuck on the boundary.
        # The move_vectors should push them in.
        # Let's clip to [-0.5, 0.5] to ensure validity, but the optimization should keep them inside.
        current_centers = np.clip(current_centers, -0.5, 0.5)
        
        # Random perturbation to escape local minima
        # Every 100 steps, do a small random jump
        if it % 100 == 0 and it > 0:
            # Reduce step size over time (annealing)
            step_size = max(0.001, step_size * 0.99)
            current_centers += np.random.randn(N, 3) * 0.01

    # Final check: ensure best_centers is valid
    # If best_score is very low or invalid, return current?
    # Usually best_centers is the one with highest score.
    
    return best_centers
# EVOLVE-BLOCK-END
