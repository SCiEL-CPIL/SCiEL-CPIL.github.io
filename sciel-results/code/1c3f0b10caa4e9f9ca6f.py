"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def _compute_radius(centers):
    """Compute the valid radius for a set of centers."""
    if centers is None or centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    
    # Distance to walls
    min_dist_wall = 0.5 - np.max(np.abs(centers), axis=1)
    r_wall = np.min(min_dist_wall)
    
    # Distance between spheres
    if N < 2:
        return r_wall
        
    # Use scipy or manual calculation? Manual is safer for dependencies, but scipy is allowed.
    # Let's use a vectorized approach with numpy to avoid scipy dependency issues if any, 
    # though scipy is listed as available. 
    # For N=100, 100*99/2 pairs is small.
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    # Set diagonal to inf
    np.fill_diagonal(dist, np.inf)
    min_dist_pair = np.min(dist)
    r_pair = min_dist_pair / 2.0
    
    return min(r_wall, r_pair)

def _force_directed_relaxation(centers, max_iter=5000, dt=0.001, eps=1e-10):
    """
    Relax the centers to maximize the minimum distance using a force-directed method.
    Force: Repulsion from close neighbors (LJ-like) + Attraction to center (to keep them in).
    Constraint: Keep centers within [-0.5+r, 0.5-r].
    """
    # We don't know r yet. We can estimate it or just use a generic repulsion.
    # A common trick: maximize sum of 1/d^2 or similar.
    # Or use a potential that penalizes d < 2r.
    
    # Let's use a simpler gradient ascent on the minimum distance is hard.
    # Let's use a soft-min surrogate: S = -log(sum(exp(-beta * d_ij))) approximates min(d_ij).
    # Or even simpler: maximize sum of d_ij^2? No, that pushes to corners.
    # We want to maximize the minimum distance.
    
    # Strategy: 
    # 1. Start with a reasonable grid or random points.
    # 2. Iteratively:
    #    a. Find the closest pair (i, j) and the sphere closest to a wall (k).
    #    b. Move i and j apart.
    #    c. Move k away from the wall.
    #    d. Ensure no overlap is created (or handle collisions).
    
    # This is basically the "push" algorithm.
    
    # Better: Use a global optimization heuristic?
    # Given 120s, we can afford a decent amount of computation.
    
    # Let's implement a "Repulsion" method where each point feels a force from all others.
    # F_ij = k * (r_target - d_ij) / d_ij * (c_i - c_j) if d_ij < r_target else 0?
    # We want d_ij >= 2r.
    
    # Since we don't know r, let's just maximize the minimum distance directly using a soft objective.
    # Objective: f(x) = -log(sum_{i<j} exp(-alpha * d_ij)) - beta * sum_i log(0.5 - |x_i|)
    # This is smooth and differentiable.
    
    # However, implementing gradient ascent manually for 300 dims (100*3) is feasible.
    
    # Let's try a simpler approach first:
    # 1. Generate a grid.
    # 2. Run a local refinement that pushes overlapping spheres apart.
    
    # Actually, the "Inspiration 2" mentioned Force-Directed Relaxation.
    # Let's implement a robust version of that.
    
    # Initial guess: Grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    # Randomize slightly to break symmetry
    centers += np.random.normal(0, 0.01, centers.shape)
    
    # Clip to valid range (assuming r ~ 0.1)
    centers = np.clip(centers, -0.4, 0.4)
    
    # Relaxation Loop
    # We want to maximize min_dist.
    # Let's use a potential energy approach:
    # E = sum_{i<j} max(0, 2*r_est - d_ij)^2 + sum_i max(0, 0.5 - r_est - |c_i|)^2
    # We can iteratively estimate r_est as the current min_dist/2.
    
    r_est = 0.1
    for iter in range(1000):
        # Calculate distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        # Current min distance
        min_dist = np.min(dist)
        r_current = min_dist / 2.0
        
        # Update r_est slowly to track the improvement
        r_est = 0.9 * r_est + 0.1 * r_current
        
        # Calculate forces
        # Force from pairs
        # If d < 2*r_est, push apart.
        # F_ij = k * (2*r_est - d_ij) / d_ij * (c_i - c_j)
        
        forces = np.zeros_like(centers)
        
        # Vectorized force calculation is tricky with the conditional.
        # We can use a mask.
        mask = dist < 2 * r_est
        # For pairs where mask is True, add force.
        
        # To do this efficiently:
        # We can just loop over pairs if N is small? 100*100 = 10000 pairs. 1000 iters = 10M ops.
        # Python loop over 10M ops is too slow (10s+).
        # We need vectorization.
        
        # Let's use a soft repulsion that is always active but strong when close.
        # U = 1/d^2. Force = -grad U = 2/d^3 * diff.
        # This pushes all points apart, but strongly for close ones.
        # We also need attraction to center to keep them in the box.
        # F_center = -k_c * centers
        
        # Let's compute 1/d^3
        # Avoid division by zero
        dist_safe = np.where(dist < 1e-6, 1e-6, dist)
        inv_d3 = 1.0 / (dist_safe ** 3)
        
        # Force from pair i on j is proportional to (c_i - c_j) / d^3
        # Sum over j for each i.
        # F_i = sum_j (c_i - c_j) / d_ij^3
        
        # This can be computed as:
        # F_i = (sum_j (c_i - c_j) / d_ij^3)
        # = c_i * sum_j (1/d_ij^3) - sum_j (c_j / d_ij^3)
        
        # Let's compute matrix M where M[i,j] = 1/d_ij^3
        # Then F_i = c_i * row_sum(M[i,:]) - sum_j M[i,j] * c_j
        
        # We set diagonal to 0 to avoid self-interaction
        np.fill_diagonal(inv_d3, 0.0)
        
        row_sums = np.sum(inv_d3, axis=1)
        # Matrix multiplication: (N, N) @ (N, 3) -> (N, 3)
        weighted_centers = inv_d3 @ centers
        
        repulsive_force = centers * row_sums[:, np.newaxis] - weighted_centers
        
        # Attractive force to center
        attractive_force = -10.0 * centers  # Tunable
        
        total_force = repulsive_force + attractive_force
        
        # Update positions
        # Adaptive step size
        step = 0.001
        centers += step * total_force
        
        # Constrain to box
        # We don't know exact r, but we know it must be < 0.5.
        # Let's clip to [-0.49, 0.49] to be safe, then refine.
        centers = np.clip(centers, -0.49, 0.49)
        
        # Check convergence
        if iter % 100 == 0:
            # print(f"Iter {iter}, r={r_current:.6f}")
            pass
            
    # Final Polish
    # Use a local optimization to maximize the minimum distance precisely.
    # We can use scipy.optimize.minimize with a custom objective.
    # Objective: -min_dist
    
    # However, min_dist is non-smooth.
    # Let's do a few iterations of "push closest pair" to polish.
    
    for _ in range(200):
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        min_idx = np.unravel_index(np.argmin(dist), dist.shape)
        i, j = min_idx
        d = dist[i, j]
        
        if d < 1e-6:
            break
            
        # Move i and j apart by half the overlap? No, just push them apart a bit.
        # We want to increase d.
        # Direction: c_i - c_j
        dir_vec = centers[i] - centers[j]
        dir_norm = np.linalg.norm(dir_vec)
        if dir_norm < 1e-9:
            dir_vec = np.array([1.0, 0.0, 0.0])
            dir_norm = 1.0
        else:
            dir_vec /= dir_norm
            
        # Push apart
        # How much? A small step.
        step = 0.001
        centers[i] += step * dir_vec
        centers[j] -= step * dir_vec
        
        # Also check walls
        # Find point closest to wall
        dist_wall = 0.5 - np.max(np.abs(centers), axis=1)
        min_wall_idx = np.argmin(dist_wall)
        c = centers[min_wall_idx]
        # Move away from closest wall
        # Find which coordinate is closest to bound
        abs_c = np.abs(c)
        max_coord = np.argmax(abs_c)
        sign = np.sign(c[max_coord])
        # Move inwards
        centers[min_wall_idx, max_coord] -= sign * step
        
        centers = np.clip(centers, -0.5, 0.5)
        
    return centers

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Initial Grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    best_centers = grid[:N].copy()
    best_radius = _compute_radius(best_centers)
    
    # 2. Force Directed Relaxation
    # Run multiple restarts if time permits
    restarts = 0
    while time.time() < deadline - 20.0:
        # Randomize initial grid
        centers = best_centers + np.random.normal(0, 0.05, best_centers.shape)
        centers = np.clip(centers, -0.45, 0.45)
        
        relaxed_centers = _force_directed_relaxation(centers)
        
        r = _compute_radius(relaxed_centers)
        if r > best_radius:
            best_radius = r
            best_centers = relaxed_centers
            # If we found a good one, maybe tighten the randomization for next restart?
            # Or just continue.
        
        restarts += 1
        
    # 3. Final Polish on best_centers
    # Do a more precise local search
    # We can use a finer step size for the "push" algorithm
    
    polish_centers = best_centers.copy()
    
    for _ in range(500):
        if time.time() > deadline - 5.0:
            break
            
        diff = polish_centers[:, np.newaxis, :] - polish_centers[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        min_idx = np.unravel_index(np.argmin(dist), dist.shape)
        i, j = min_idx
        d = dist[i, j]
        
        if d < 1e-8:
            break
            
        dir_vec = polish_centers[i] - polish_centers[j]
        dir_norm = np.linalg.norm(dir_vec)
        if dir_norm < 1e-9:
            continue
            
        dir_vec /= dir_norm
        
        # Small step
        step = 0.0001
        polish_centers[i] += step * dir_vec
        polish_centers[j] -= step * dir_vec
        
        # Wall constraint
        dist_wall = 0.5 - np.max(np.abs(polish_centers), axis=1)
        min_wall_idx = np.argmin(dist_wall)
        c = polish_centers[min_wall_idx]
        abs_c = np.abs(c)
        max_coord = np.argmax(abs_c)
        sign = np.sign(c[max_coord])
        polish_centers[min_wall_idx, max_coord] -= sign * step
        
        polish_centers = np.clip(polish_centers, -0.5, 0.5)
        
    r_polish = _compute_radius(polish_centers)
    if r_polish > best_radius:
        best_centers = polish_centers
        best_radius = r_polish
        
    return best_centers

# EVOLVE-BLOCK-END
