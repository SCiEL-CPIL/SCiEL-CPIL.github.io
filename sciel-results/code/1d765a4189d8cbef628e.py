"""Rewritten program: Fibonacci disk initialization + repulsion-based relaxation with soft penalties."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    """
    Construct N=300 circle centers in the unit disk to maximize the minimum
    pairwise distance and distance to boundary.
    
    Strategy:
    1. Initialize using Fibonacci sphere distribution mapped to disk.
    2. Apply repulsion-based relaxation with soft penalty functions for
       overlaps and boundary violations.
    """
    rng = np.random.default_rng(123)
    n = N
    
    # --- Step 1: Initialize using Fibonacci sphere mapped to disk ---
    # Fibonacci sphere: points on sphere, then project to disk
    # Use golden angle spiral on sphere, then map to disk via stereographic or simple projection
    
    # Alternative: Use Fibonacci disk (Vogel) which is already 2D
    # But let's use a different initialization: hexagonal grid with perturbation
    # or use the Fibonacci sphere approach mapped to disk
    
    # Let's use a hybrid: Fibonacci disk with different parameters
    indices = np.arange(n)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    
    # Fibonacci disk with slight randomization
    r = np.sqrt((indices + 0.5) / n)
    theta = golden_angle * indices
    
    # Add small random perturbation to break symmetry
    r += rng.random(n) * 0.02
    theta += rng.random(n) * 0.02
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    centers = np.column_stack([x, y])
    
    # Scale to fit inside unit disk with margin
    radii = np.linalg.norm(centers, axis=1)
    max_r = np.max(radii)
    if max_r > 0.95:
        centers *= 0.95 / max_r
    
    # --- Step 2: Repulsion-based relaxation with soft penalties ---
    # We want to maximize min(min_pair_dist/2, min_boundary_dist)
    # Use soft penalty: penalize pairs that are too close and points too close to boundary
    
    n_iterations = 8000
    lr_init = 0.02
    lr_decay = 0.9995
    lr = lr_init
    
    # Softness parameter for penalty function
    # We'll use: penalty = sum of exp(-alpha * (dist - target)^2) for pairs
    # and exp(-beta * (1 - r)^2) for boundary
    # Actually, let's use a simpler approach: 
    # For each pair, if dist < 2*r_target, apply repulsion force
    # For boundary, if r > 1 - r_target, apply inward force
    
    # Target radius estimate
    r_target = 0.053
    
    best_centers = centers.copy()
    best_score = -np.inf
    
    for it in range(n_iterations):
        # Compute pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        dist = np.sqrt(dist_sq)
        
        # Mask diagonal
        mask_diag = np.eye(n, dtype=bool)
        dist[mask_diag] = np.inf
        
        # Compute boundary distances
        radii = np.linalg.norm(centers, axis=1)
        boundary_dists = 1.0 - radii
        
        # Current score
        min_pair_dist = np.min(dist)
        min_boundary_dist = np.min(boundary_dists)
        current_score = min(min_pair_dist / 2.0, min_boundary_dist)
        
        if current_score > best_score:
            best_score = current_score
            best_centers = centers.copy()
        
        # Compute forces (repulsion)
        # For pairs: if dist < 2*r_target, push apart
        # Force magnitude: k * (2*r_target - dist) / dist for dist < 2*r_target
        
        force = np.zeros_like(centers)
        
        # Pair forces
        # For each pair (i,j), if dist_ij < 2*r_target:
        #   force_i += k * (2*r_target - dist_ij) / dist_ij * (c_i - c_j) / dist_ij
        #   force_j += k * (2*r_target - dist_ij) / dist_ij * (c_j - c_i) / dist_ij
        
        # Vectorized computation
        # Only consider pairs with dist < 2*r_target
        threshold = 2.0 * r_target
        pair_mask = (dist < threshold) & (~mask_diag)
        
        # Weight for each pair: k * (2*r_target - dist) / dist^2
        # where dist > 0
        dist_safe = np.where(pair_mask, dist, 1.0)
        weights = np.where(pair_mask, (2.0 * r_target - dist_safe) / (dist_safe ** 2), 0.0)
        
        # Force on i from j: weights[i,j] * (c_i - c_j)
        # Sum over j
        # force_i = sum_j weights[i,j] * (c_i - c_j)
        #          = c_i * sum_j weights[i,j] - sum_j weights[i,j] * c_j
        
        sum_weights = np.sum(weights, axis=1)
        weighted_centers = weights @ centers
        pair_forces = centers * sum_weights[:, np.newaxis] - weighted_centers
        
        # Boundary forces
        # For each point, if boundary_dist < r_target:
        #   force_i += k_b * (r_target - boundary_dist) * (-c_i / r_i)
        # i.e., push inward
        
        boundary_threshold = r_target
        boundary_mask = boundary_dists < boundary_threshold
        
        # Force magnitude: k_b * (r_target - boundary_dist)
        # Direction: inward, i.e., -c_i / |c_i|
        radii_safe = np.where(radii > 1e-6, radii, 1.0)
        inward_direction = -centers / radii_safe[:, np.newaxis]
        boundary_forces = inward_direction * (boundary_threshold - boundary_dists)[:, np.newaxis]
        boundary_forces = boundary_forces * boundary_mask[:, np.newaxis]
        
        # Total force
        total_force = pair_forces + boundary_forces
        
        # Update with simple gradient ascent
        centers += lr * total_force
        
        # Hard constraint: keep inside unit disk
        radii = np.linalg.norm(centers, axis=1)
        outside = radii >= 0.999
        if np.any(outside):
            centers[outside] = centers[outside] / radii[outside, np.newaxis] * 0.999
        
        # Decay learning rate
        lr *= lr_decay
        
        if lr < 1e-8:
            break
    
    # Final cleanup
    radii = np.linalg.norm(best_centers, axis=1)
    outside = radii >= 1.0
    if np.any(outside):
        best_centers[outside] = best_centers[outside] / radii[outside, np.newaxis] * 0.999
        
    return best_centers
# EVOLVE-BLOCK-END
