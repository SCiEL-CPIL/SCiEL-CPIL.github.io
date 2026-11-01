"""
Packing 50 equal spheres in a unit cube via Geometric Force-Directed Optimization.

Strategy:
1. Initialize with a structured grid (4x4x4 subset) which provides a good topological start.
2. Iteratively identify the "bottleneck" constraints: the pair of spheres with the minimum
   separation and the sphere with the minimum distance to the wall.
3. Apply repulsive forces to the centers involved in these bottleneck constraints to push
   them apart or away from the walls.
4. Use a cooling schedule to reduce the magnitude of moves.
5. This is a direct geometric relaxation of the max-min problem, avoiding the need for
   smooth surrogates or complex gradient calculations.
"""

import numpy as np
import time
import math

N = 50
DEADLINE = 145.0  # seconds, leaving 5s margin for the 150s limit

# EVOLVE-BLOCK-START

def _compute_radius(centers):
    """Compute the maximum common radius for a given set of centers."""
    if not np.all(np.isfinite(centers)):
        return -1.0
    
    # Distance to walls
    # centers is (N, 3)
    # Distance to face x=0.5 is 0.5 - c_x, etc.
    # Distance to nearest wall for sphere i is min(0.5 - c_i_x, 0.5 + c_i_x, ...) 
    # Since cube is [-0.5, 0.5]^3, distance to wall is 0.5 - max(|c_x|, |c_y|, |c_z|) ? 
    # No, the distance to the boundary of the cube for a point (x,y,z) inside [-0.5, 0.5]^3 is:
    # min(0.5 - x, 0.5 + x, 0.5 - y, 0.5 + y, 0.5 - z, 0.5 + z)
    # Which is equivalent to 0.5 - max(abs(x), abs(y), abs(z))
    
    wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
    
    # Distance between pairs
    # For N=50, O(N^2) is 2500 ops, very fast.
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    
    min_pair_dist = np.min(dists)
    
    # The radius is limited by the smallest wall distance and half the smallest pair distance
    return np.min(np.minimum(wall_dists, min_pair_dist / 2.0))

def _compute_radius_and_bottlenecks(centers):
    """
    Compute radius and identify the specific constraints (indices) that define the bottleneck.
    Returns:
    radius: float
    bottleneck_pairs: list of (i, j) tuples for pair constraints
    bottleneck_walls: list of (i, wall_axis, wall_sign) for wall constraints
    """
    if not np.all(np.isfinite(centers)):
        return -1.0, [], []
    
    wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    
    min_pair_dist = np.min(dists)
    min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
    
    # There might be multiple pairs with the same min distance
    # We find all pairs with distance close to min
    eps = 1e-9
    pair_mask = (dists <= min_pair_dist + eps) & (dists < np.inf)
    bottleneck_pairs = []
    for i in range(N):
        for j in range(i+1, N):
            if pair_mask[i, j]:
                bottleneck_pairs.append((i, j))
    
    # Wall bottlenecks
    min_wall_dist = np.min(wall_dists)
    bottleneck_walls = []
    for i in range(N):
        if abs(wall_dists[i] - min_wall_dist) < eps:
            # Determine which wall is closest
            c = centers[i]
            # Check each face
            dist_x_pos = 0.5 - c[0]
            dist_x_neg = 0.5 + c[0]
            dist_y_pos = 0.5 - c[1]
            dist_y_neg = 0.5 + c[1]
            dist_z_pos = 0.5 - c[2]
            dist_z_neg = 0.5 + c[2]
            
            min_d = min(dist_x_pos, dist_x_neg, dist_y_pos, dist_y_neg, dist_z_pos, dist_z_neg)
            
            # Add all walls that are at the minimum distance
            if abs(dist_x_pos - min_d) < eps: bottleneck_walls.append((i, 0, 1))
            if abs(dist_x_neg - min_d) < eps: bottleneck_walls.append((i, 0, -1))
            if abs(dist_y_pos - min_d) < eps: bottleneck_walls.append((i, 1, 1))
            if abs(dist_y_neg - min_d) < eps: bottleneck_walls.append((i, 1, -1))
            if abs(dist_z_pos - min_d) < eps: bottleneck_walls.append((i, 2, 1))
            if abs(dist_z_neg - min_d) < eps: bottleneck_walls.append((i, 2, -1))
            
    radius = np.min(np.minimum(min_wall_dist, min_pair_dist / 2.0))
    return radius, bottleneck_pairs, bottleneck_walls

def _initial_grid():
    """
    Generate a 4x4x4 grid (64 points) and select 50.
    We try to select points that are spread out. 
    Actually, a simple random selection from the grid is fine, but let's try to 
    maximize initial radius by picking a subset that is well-distributed.
    For simplicity and speed, we just pick the first 50 or a random 50.
    A better initial config for 50 spheres in a cube might be a 5x5x4 grid (100 pts) or similar.
    Let's stick to 4x4x4 = 64. Removing 14 points.
    """
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # We want to remove 14 points. 
    # To maximize radius, we should probably remove points that are "redundant" or create high density.
    # In a 4x4x4 grid, all points are equidistant from neighbors.
    # Removing any 14 will result in the same local density for the remaining ones 
    # unless we break symmetry.
    # Let's just use a fixed random seed selection for reproducibility.
    rng = np.random.default_rng(42)
    indices = rng.choice(len(grid), size=N, replace=False)
    return grid[indices].copy()

def _force_directed_optimize(centers, deadline):
    """
    Iterative geometric relaxation.
    """
    current = centers.copy()
    current_radius = _compute_radius(current)
    best = current.copy()
    best_radius = current_radius
    
    # Cooling schedule
    # Start with a step size that allows significant movement, reduce over time.
    # The "temperature" T controls the step size.
    T = 0.05
    T_min = 1e-6
    alpha = 0.9995  # Cooling rate
    
    start_time = time.time()
    
    # Main loop
    while time.time() < deadline - 5:
        radius, pairs, walls = _compute_radius_and_bottlenecks(current)
        
        if radius > best_radius:
            best_radius = radius
            best = current.copy()
            
        # If no bottlenecks (should not happen), break
        if not pairs and not walls:
            break
            
        # Compute forces
        # We want to increase the minimum distance.
        # For each bottleneck pair (i, j), we push i and j apart.
        # For each bottleneck wall (i, axis, sign), we push i away from the wall.
        
        # We accumulate displacement vectors
        displacements = np.zeros_like(current)
        
        # Normalize force magnitude? 
        # The "strength" of the constraint is how close it is to the global minimum.
        # Here, we assume all listed bottlenecks are at the global minimum radius.
        # We apply a unit force to each, scaled by T.
        
        for i, j in pairs:
            vec = current[i] - current[j]
            dist = np.linalg.norm(vec)
            if dist < 1e-9:
                # Random nudge
                vec = np.random.normal(0, 1, 3)
                dist = np.linalg.norm(vec)
            unit_vec = vec / dist
            # Push i away from j, j away from i
            displacements[i] += unit_vec
            displacements[j] -= unit_vec
            
        for i, axis, sign in walls:
            # Push i away from the wall
            # Wall is at -0.5 * sign? No.
            # If sign=1 (positive wall, e.g., x=0.5), we want to decrease x (push towards -x).
            # If sign=-1 (negative wall, e.g., x=-0.5), we want to increase x (push towards +x).
            # So direction is -sign * e_axis
            dir_vec = np.zeros(3)
            dir_vec[axis] = -sign
            displacements[i] += dir_vec
            
        # Normalize displacements to avoid huge steps if many constraints
        # Actually, we want to move in the direction that increases the min distance.
        # If we just sum unit vectors, a point with 3 constraints might move 3x as much.
        # Let's normalize the total displacement for each point to have magnitude <= 1.
        norms = np.linalg.norm(displacements, axis=1)
        for i in range(N):
            if norms[i] > 1:
                displacements[i] /= norms[i]
                
        # Apply move
        # Step size is T
        proposed = current + T * displacements
        
        # Project back to cube
        proposed = np.clip(proposed, -0.5, 0.5)
        
        # Check if this move improved the radius
        # If not, we might still accept it with some probability (simulated annealing) 
        # or just reject it.
        # Given the "force" direction is towards increasing local min distances, 
        # it usually improves the global min unless it creates a *new* smaller bottleneck elsewhere.
        # Let's calculate the new radius.
        new_radius = _compute_radius(proposed)
        
        if new_radius >= current_radius:
            current = proposed
            current_radius = new_radius
        else:
            # Accept with probability exp((new_radius - current_radius) / T_annealing)
            # T_annealing should be related to the scale of radius changes.
            # Radius is ~0.1. Changes are small.
            # Let's use a fixed annealing temperature for acceptance.
            T_ann = 0.001
            if np.random.random() < math.exp((new_radius - current_radius) / T_ann):
                current = proposed
                current_radius = new_radius
        
        # Cool down
        T *= alpha
        if T < T_min:
            T = T_min
            # If we've cooled down, we might want to restart with a slightly perturbed best solution
            # to escape local minima.
            # Simple restart: add small noise to best
            if best_radius > 0:
                noise = np.random.normal(0, 0.01, size=(N, 3))
                perturbed = best + noise
                perturbed = np.clip(perturbed, -0.5, 0.5)
                p_radius = _compute_radius(perturbed)
                if p_radius > current_radius:
                    current = perturbed
                    current_radius = p_radius
                # Reset temperature
                T = 0.01
                # But don't loop forever if no progress
                # We rely on the deadline.

    return best

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # 1. Initial Configuration
    centers = _initial_grid()
    
    # 2. Force-Directed Optimization
    if time.time() < deadline - 10:
        centers = _force_directed_optimize(centers, deadline)
        
    # 3. Final Polish with a very fine step size
    # Do a few iterations with T=1e-6 to settle
    if time.time() < deadline - 5:
        # Create a short deadline for the final polish
        polish_deadline = time.time() + 5
        centers = _force_directed_optimize(centers, polish_deadline)
        
    # Validation
    final_radius = _compute_radius(centers)
    if final_radius < 0 or not np.all(np.isfinite(centers)):
        # Fallback
        centers = _initial_grid()
        
    return centers

# EVOLVE-BLOCK-END
