"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time

def _score(centers):
    """Compute the radius score: min of half min-distance and min boundary distance."""
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Boundary constraint: centers must be inside [-0.5, 0.5]^2
    boundary_margin = 0.5 - np.max(np.abs(centers), axis=1)
    min_boundary = np.min(boundary_margin)
    if min_boundary <= 0:
        return 0.0
    # Pairwise distances
    dists = np.sqrt(np.sum((centers[:, np.newaxis, :] - centers[np.newaxis, :, :])**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    r = min(min_dist / 2.0, min_boundary)
    return r

def _initial_hex_grid(n):
    """
    Generate a hexagonal grid of n points centered at (0,0) within the unit square.
    """
    # Estimate grid size
    # Area per point ~ 1/n
    # Hex spacing s ~ sqrt(1/n / (sqrt(3)/2))
    s = np.sqrt(1.0 / (n * np.sqrt(3) / 2.0))
    
    # Generate a large enough hex grid
    # Rows and cols needed
    rows = int(np.ceil(np.sqrt(n * 1.2))) + 2
    cols = int(np.ceil(np.sqrt(n * 1.2))) + 2
    
    points = []
    dy = s * np.sqrt(3) / 2.0
    for i in range(-rows, rows + 1):
        y = i * dy
        x_offset = 0 if i % 2 == 0 else s / 2.0
        for j in range(-cols, cols + 1):
            x = j * s + x_offset
            points.append((x, y))
            
    points = np.array(points)
    
    # Keep only points inside the square (with a small margin to be safe? No, we want them inside)
    # Actually, we want to select n points from this set that are closest to the center?
    # Or just take the first n?
    # Better: sort by distance from origin and take the closest n?
    # This ensures a compact cluster in the center, which is good for packing.
    
    dists_to_origin = np.linalg.norm(points, axis=1)
    sorted_indices = np.argsort(dists_to_origin)
    points = points[sorted_indices[:n]]
    
    return points

def _force_displacement(centers, max_iters=5000, dt=0.01, k_repel=1.0, k_wall=10.0, deadline=None):
    """
    Optimize circle packing using force displacement.
    - Repulsive force between circles: F ~ 1/d^2 (or similar)
    - Repulsive force from walls: F ~ 1/(distance to wall)^2
    """
    centers = centers.copy()
    n = len(centers)
    
    # Precompute pairwise indices for efficiency? 
    # For N=200, N^2 is 40,000. This is small enough for vectorized operations.
    
    # We will update centers iteratively.
    # To avoid O(N^2) Python loops, we use vectorized numpy operations for force calculation.
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
            
        # Calculate forces
        # 1. Pairwise repulsion
        # diff[i, j] = centers[i] - centers[j]
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        # Avoid division by zero
        np.fill_diagonal(dist_sq, 1.0)
        
        # Force magnitude: k_repel / dist_sq
        # Direction: diff / dist
        dist = np.sqrt(dist_sq)
        # F_ij = k_repel * (diff / dist) / dist_sq = k_repel * diff / dist^3
        # To keep it simple and stable, use F = k_repel * diff / (dist_sq + epsilon)
        # Or simply F = k_repel * diff / dist_sq (inverse square law on distance, so force is 1/d^2 * direction)
        # Let's use F = k_repel * diff / (dist_sq + 1e-6)
        
        forces = np.zeros_like(centers)
        
        # Vectorized force calculation
        # F_i = sum_j F_ij
        # F_ij = k_repel * (centers[i] - centers[j]) / (dist_sq[i,j] + eps)
        
        eps = 1e-6
        inv_dist_sq = 1.0 / (dist_sq + eps)
        # Multiply diff by inv_dist_sq
        # diff is (N, N, 2)
        # inv_dist_sq is (N, N)
        # We need to multiply each vector by the scalar.
        # force_contrib = diff * inv_dist_sq[..., np.newaxis]
        force_contrib = diff * (inv_dist_sq * k_repel)[..., np.newaxis]
        
        # Sum over j (axis 1) to get force on i
        forces += np.sum(force_contrib, axis=1)
        
        # 2. Wall repulsion
        # Distance to walls: 
        # Left: x + 0.5, Right: 0.5 - x, Bottom: y + 0.5, Top: 0.5 - y
        # Force is inward.
        # F_x from left wall: if x < -0.5, push right. But we assume points are mostly inside.
        # Let's define distance to boundary d_b = min(x+0.5, 0.5-x, y+0.5, 0.5-y)
        # If d_b < threshold, apply force.
        # Actually, simpler: apply force based on distance to each wall.
        # F_x_left = k_wall * (x + 0.5) / (x + 0.5)^2 ? No.
        # Let's use a potential V = k_wall / d^2. Force = -dV/dx.
        # For left wall: d = x + 0.5. V = k / (x+0.5)^2. dV/dx = -2k/(x+0.5)^3. Force = +2k/(x+0.5)^3 (towards right).
        # This is valid for x > -0.5.
        
        x = centers[:, 0]
        y = centers[:, 1]
        
        # Distances to walls
        d_left = x + 0.5
        d_right = 0.5 - x
        d_bottom = y + 0.5
        d_top = 0.5 - y
        
        # Add epsilon to avoid singularities if a point is exactly on the wall
        d_left = np.maximum(d_left, 1e-6)
        d_right = np.maximum(d_right, 1e-6)
        d_bottom = np.maximum(d_bottom, 1e-6)
        d_top = np.maximum(d_top, 1e-6)
        
        # Forces
        # F_x = 2*k_wall * (1/d_left^3 - 1/d_right^3) ? 
        # Wait, potential is k/d^2. Gradient is -2k/d^3.
        # Force = -grad V.
        # For left wall (x increases away from wall): V = k/d_left^2. dV/dx = -2k/d_left^3. Force_x = +2k/d_left^3.
        # For right wall (x decreases away from wall): d = 0.5 - x. V = k/d^2. dV/dx = k * (-2) * d^-3 * (-1) = 2k/d^3. Force_x = -2k/d^3.
        # So F_x = 2*k_wall * (1/d_left^3 - 1/d_right^3)
        
        F_x = 2.0 * k_wall * (1.0 / (d_left**3) - 1.0 / (d_right**3))
        F_y = 2.0 * k_wall * (1.0 / (d_bottom**3) - 1.0 / (d_top**3))
        
        forces[:, 0] += F_x
        forces[:, 1] += F_y
        
        # Update positions
        # Use a simple Euler step with damping or just fixed dt
        # To prevent explosion, we can limit the step size or use adaptive dt.
        # Let's try a fixed small dt and see.
        centers += dt * forces
        
        # Clamp to square to prevent points from flying out (though wall forces should keep them in)
        # If a point goes out, the wall force becomes undefined or attractive in our simple model.
        # So we MUST clamp.
        centers[:, 0] = np.clip(centers[:, 0], -0.5, 0.5)
        centers[:, 1] = np.clip(centers[:, 1], -0.5, 0.5)
        
    return centers

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # 1. Initialize
    centers = _initial_hex_grid(N)
    
    # 2. Optimize
    # Tune parameters
    # k_repel needs to be strong enough to push apart.
    # k_wall needs to be strong enough to keep in bounds.
    # dt needs to be small enough for stability.
    
    # Let's run a few iterations with different parameters if time permits, 
    # or just one robust run.
    
    # A good strategy is to start with a large dt and reduce it (simulated annealing-like)
    # or just use a fixed small dt for many iterations.
    
    # Given N=200, N^2=40000 ops per iteration. 
    # 5000 iterations * 40000 = 200M ops. NumPy can do this in < 1s.
    # So we can afford many iterations.
    
    # Let's try a multi-stage approach:
    # Stage 1: Coarse optimization (large dt, few iters)
    # Stage 2: Fine optimization (small dt, many iters)
    
    # Stage 1
    centers = _force_displacement(centers, max_iters=1000, dt=0.05, k_repel=0.1, k_wall=1.0, deadline=deadline)
    
    # Stage 2
    centers = _force_displacement(centers, max_iters=5000, dt=0.005, k_repel=0.5, k_wall=5.0, deadline=deadline)
    
    # Stage 3 (if time)
    if time.time() < deadline - 10:
        centers = _force_displacement(centers, max_iters=5000, dt=0.001, k_repel=1.0, k_wall=10.0, deadline=deadline)
        
    # Final check and return
    # Ensure no points are coincident or outside
    # The clamping ensures inside.
    # Coincident points: force_displacement should separate them.
    
    return centers
# EVOLVE-BLOCK-END
