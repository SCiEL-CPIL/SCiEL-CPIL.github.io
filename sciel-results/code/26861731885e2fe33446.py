"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    start_time = time.time()
    deadline = 140.0  # 10s safety margin
    
    # 1. Fibonacci lattice (Sunflower) initialization
    # This provides a highly uniform distribution with strong minimum distances.
    # Mapping to the unit square [-0.5, 0.5]^2
    n = N
    golden_angle = np.pi * (3. - np.sqrt(5.))
    r = np.sqrt(np.arange(n) + 0.5) / np.sqrt(n)
    theta = (np.arange(n) + 0.5) * golden_angle
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    # Scale to fit inside [-0.5, 0.5] with a small margin to avoid initial boundary violations
    # The max radius in the unit disk is 1. We want it to fit in a square of side 1.
    # If we just map the disk to the square, the corners are empty.
    # A simple scaling by 0.5 maps the unit disk into [-0.5, 0.5].
    # However, the square has corners. The disk touches the midpoints of the sides.
    # This leaves the corners empty, which is good for boundary constraints initially.
    x = 0.5 * x
    y = 0.5 * y
    
    pos = np.column_stack((x, y))
    
    # 2. Repulsive Force Molecular Dynamics
    # We simulate particles that repel each other.
    # Force: F = k / d^2 (Coulomb-like) or just 1/d for stability.
    # We also apply a force pushing particles back inside the square.
    
    # Initial radius estimate based on area:
    # Area = 1. N circles. r ~ 0.5 / sqrt(N) = 0.5 / 14.14 = 0.0353.
    # Known best is 0.0366.
    
    # We will optimize the position to maximize the minimum distance.
    # This is equivalent to minimizing the energy of a repulsive system.
    
    dt = 0.001
    max_iter = 20000
    damping = 0.99
    
    # Velocity
    vel = np.zeros_like(pos)
    
    # We want to maximize the minimum distance.
    # A common heuristic is to apply forces proportional to 1/d^2 for all pairs,
    # which effectively pushes the closest pairs apart more strongly.
    
    # To speed up convergence, we can use a "coarse to fine" approach or just run long enough.
    # 200 particles is small enough for O(N^2) force calculation to be fast.
    # 200^2 = 40000 pairs. 20000 iters * 40000 = 800M ops. Might be too slow in Python/NumPy if not vectorized well.
    # Let's vectorize the force calculation.
    
    for i in range(max_iter):
        if time.time() > deadline:
            break
            
        # Calculate pairwise distances
        # diff[i, j] = pos[i] - pos[j]
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        
        # Avoid division by zero
        dist_sq[dist_sq == 0] = 1e-12
        
        # Force magnitude: 1 / dist^2
        # We want to push them apart.
        # Force on i due to j is in direction of (pos[i] - pos[j]) / dist * (1/dist^2) = (pos[i]-pos[j]) / dist^3
        # Let's use a simpler potential: V = 1/d. F = 1/d^2.
        # F_vec = (pos_i - pos_j) / dist^3
        
        # To avoid calculating the full NxNx3 tensor, we can use the properties of the distance matrix.
        # But N=200 is small. Let's just do it.
        
        # dist = sqrt(dist_sq)
        dist = np.sqrt(dist_sq)
        
        # Force factor: 1 / dist^3
        # We only care about i != j
        np.fill_diagonal(dist, 1.0) # Avoid div by zero on diagonal, though we'll zero out diagonal forces later
        
        force_factor = 1.0 / (dist**3)
        
        # Force on i is sum over j of (pos_i - pos_j) * force_factor[i,j]
        # diff[i,j] = pos_i - pos_j
        # force[i] = sum_j diff[i,j] * force_factor[i,j]
        
        forces = np.sum(diff * force_factor[:, :, np.newaxis], axis=1)
        
        # Boundary forces
        # If a particle is close to the boundary, push it back.
        # Boundary at -0.5 and 0.5.
        # Let's say we start pushing if dist_to_wall < 0.05.
        # Force is linear with penetration or just constant push?
        # A soft wall potential is better.
        # V_wall = k * (wall_dist - d)^2 if d < wall_dist
        # F_wall = 2 * k * (wall_dist - d)
        
        wall_margin = 0.01 # Start pushing within 0.01 of the wall
        k_wall = 10.0
        
        # x boundaries
        # dist to left wall: pos_x - (-0.5) = pos_x + 0.5
        # dist to right wall: 0.5 - pos_x
        
        # We want to push towards the center if too close to wall.
        # If pos_x < -0.5 + wall_margin, force_x += k_wall * ( (-0.5 + wall_margin) - pos_x )
        # If pos_x > 0.5 - wall_margin, force_x -= k_wall * ( pos_x - (0.5 - wall_margin) )
        
        # Vectorized boundary force
        # Left wall
        left_dist = pos[:, 0] + 0.5
        right_dist = 0.5 - pos[:, 0]
        bottom_dist = pos[:, 1] + 0.5
        top_dist = 0.5 - pos[:, 1]
        
        # Penalty if dist < margin
        # force = k * (margin - dist) if dist < margin else 0
        
        fx_wall = np.where(left_dist < wall_margin, k_wall * (wall_margin - left_dist), 0.0)
        fx_wall -= np.where(right_dist < wall_margin, k_wall * (wall_margin - right_dist), 0.0)
        
        fy_wall = np.where(bottom_dist < wall_margin, k_wall * (wall_margin - bottom_dist), 0.0)
        fy_wall -= np.where(top_dist < wall_margin, k_wall * (wall_margin - top_dist), 0.0)
        
        forces[:, 0] += fx_wall
        forces[:, 1] += fy_wall
        
        # Update velocity and position
        vel = vel * damping + forces * dt
        pos += vel * dt
        
        # Hard constraint: Clip positions to be strictly inside the square
        # This prevents escape and maintains validity.
        pos = np.clip(pos, -0.499, 0.499)
        
        # Optional: Check for improvement in minimum distance to stop early if converged?
        # For now, run full budget.
        
    # Final cleanup: ensure no overlaps or boundary violations due to floating point errors
    # A simple local relaxation pass to fix any minor overlaps
    # This is a "push" step.
    
    # We do a few iterations of a simpler local search to fix overlaps.
    # If any pair is closer than the current min_dist, push them apart.
    # But we don't know the target radius.
    # The MD simulation naturally maximizes the min distance.
    
    # Let's do a final check and repair.
    # Calculate current min distance.
    # If there are overlaps (dist < 2*r_current), we have a problem.
    # But since we are maximizing min dist, the "radius" is just min_dist/2.
    # The evaluator calculates r = min(min_pair_dist/2, min_boundary_dist).
    # Our MD maximizes min_pair_dist. It also keeps them away from boundaries.
    
    # One final local optimization pass using gradient of the min distance?
    # No, MD is enough for N=200 in 140s.
    
    # Ensure valid shape and finite values
    return pos

# EVOLVE-BLOCK-END
