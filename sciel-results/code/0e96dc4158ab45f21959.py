"""Pack 300 equal circles in a unit disk using hexagonal lattice init and robust local search."""
import numpy as np
import time
import math

N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _hexagonal_init(n, radius=1.0):
    """
    Initialize n points in a hexagonal lattice pattern within a disk of given radius.
    This provides a denser, more structured starting point than Fibonacci spiral.
    """
    # Estimate optimal spacing for hexagonal packing in a disk of radius R
    # Area of disk = pi * R^2. Area per circle ~ 2*pi*r^2 (approx for dense packing)
    # Number of circles n ~ (pi * R^2) / (2 * pi * r^2 * sqrt(3)/2) ? 
    # Actually, density of hex packing is pi/sqrt(12) ~ 0.9069
    # n * pi * r^2 <= pi * R^2 * 0.9069
    # r <= R * sqrt(0.9069 / n)
    
    # We want to place points on a hex grid.
    # Spacing s.
    # Number of points in a hexagon of "radius" k (in lattice units) is 3k(k+1)+1
    # Total radius in cartesian is roughly k * s * sqrt(3) (distance to corner)
    
    # Let's just generate a large hex grid and filter points inside the disk.
    # We need about N points.
    # Estimate s: r_est = radius * sqrt(0.9 / n)
    # s_est = 2 * r_est
    
    r_est = radius * math.sqrt(0.9 / n)
    s_est = 2.0 * r_est
    
    # Generate points
    points = []
    # Max index needed to cover the disk
    max_k = int(math.ceil(radius / (s_est * 0.5)) + 2)
    
    # Hex grid coordinates
    # x = s * (q + r/2)
    # y = s * (sqrt(3)/2) * r
    
    sqrt3 = math.sqrt(3)
    for q in range(-max_k, max_k + 1):
        for r in range(-max_k, max_k + 1):
            x = s_est * (q + 0.5 * r)
            y = s_est * (0.5 * sqrt3 * r)
            if math.sqrt(x*x + y*y) <= radius:
                points.append((x, y))
                
    points = np.array(points)
    
    # If we have more points than needed, randomly select N
    if len(points) > n:
        indices = np.random.choice(len(points), n, replace=False)
        points = points[indices]
    elif len(points) < n:
        # If we didn't generate enough (shouldn't happen with max_k logic), 
        # add random points
        while len(points) < n:
            # Sample random point in disk
            r = radius * np.sqrt(np.random.random())
            theta = 2 * np.pi * np.random.random()
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            points = np.vstack([points, [[x, y]]])
            
    return points

def _compute_min_radius(points):
    """
    Compute the maximum possible radius for the given points.
    """
    if len(points) == 0:
        return 0.0
    
    dists_to_center = np.linalg.norm(points, axis=1)
    boundary_min = np.min(1.0 - dists_to_center)
    
    if boundary_min < 0:
        return 0.0
    
    # Vectorized pairwise distance
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist_matrix, np.inf)
    min_pair_dist = np.min(dist_matrix)
    
    pair_min = min_pair_dist / 2.0
    
    return min(boundary_min, pair_min)

def _local_search(points, max_iters=2000, time_limit=5.0):
    """
    Robust local search using gradient ascent on the minimum distance.
    Includes simulated annealing-like noise to escape local minima.
    """
    points = points.copy().astype(np.float64)
    n = len(points)
    start_time = time.time()
    
    # Initial radius
    current_r = _compute_min_radius(points)
    best_r = current_r
    best_points = points.copy()
    
    # Adaptive learning rate
    lr = 0.01
    temp = 0.05 # Initial temperature for noise
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Compute forces
        dists_to_center = np.linalg.norm(points, axis=1)
        boundary_dist = 1.0 - dists_to_center
        
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist_matrix, np.inf)
        
        min_pair_dist = np.min(dist_matrix)
        pair_radius = min_pair_dist / 2.0
        boundary_radius = np.min(boundary_dist)
        
        current_r = min(pair_radius, boundary_radius)
        
        # Update best
        if current_r > best_r:
            best_r = current_r
            best_points = points.copy()
        
        # Gradient/Force calculation
        forces = np.zeros_like(points)
        
        # 1. Boundary forces
        # Push points inward if they are too close to boundary
        # We want to maximize min(1 - |c_i|)
        # If boundary is the limiting factor, we should pull all points in?
        # Or just the ones close?
        # Actually, if boundary is limiting, it means some point is at dist 1-r.
        # To increase r, we need to move that point in.
        # But moving one point in might break pair constraints.
        # A good heuristic: apply a weak inward force to all points proportional to their distance from center,
        # and a strong inward force to points near the boundary.
        
        # Let's use a soft penalty for boundary violation in the "virtual" radius
        # But we are maximizing min distance.
        
        # Strategy:
        # If boundary_radius < pair_radius:
        #   The boundary is the bottleneck. We need to move points inward.
        #   Apply force towards center for all points, stronger for those close to boundary.
        # Else:
        #   The pair distance is the bottleneck. We need to separate close pairs.
        #   Apply repulsive forces between close pairs.
        
        if boundary_radius <= pair_radius:
            # Boundary limiting
            # Normalize forces to center
            norm_vecs = points / np.maximum(dists_to_center, 1e-6)[:, np.newaxis]
            # Weight: points closer to boundary get stronger pull
            # dist_to_boundary = 1.0 - dists_to_center
            # We want to increase min dist_to_boundary.
            # Force = - norm_vec * weight
            # Let's apply a uniform inward force scaled by lr, plus extra for close points
            forces -= norm_vecs * lr * 0.5
            close_mask = dists_to_center > (1.0 - (boundary_radius + 0.01))
            forces[close_mask] -= norm_vecs[close_mask] * lr * 2.0
        else:
            # Pair limiting
            # Find pairs with small distance
            # Use a threshold to identify "close" pairs
            # To be efficient, we can just use the gradient of the min distance approximated by repulsion
            # Force on i from j: (p_i - p_j) / d_ij^2
            # Only for pairs with d_ij < cutoff
            
            cutoff = 2.0 * (pair_radius + 0.05)
            mask = dist_matrix < cutoff
            np.fill_diagonal(mask, False)
            
            # Compute forces
            # For each pair (i, j) in mask, add force
            # This can be done vectorized
            # diff is already p_i - p_j
            # We need to sum over j for each i
            # force_i = sum_j (p_i - p_j) / d_ij^2 * weight
            
            # To avoid division by zero, we already filled diagonal with inf, so mask[i,i] is False.
            # But we need d_ij^2.
            dist_sq = np.sum(diff**2, axis=2)
            np.fill_diagonal(dist_sq, 1.0) # Avoid div by 0 for diagonal (though masked)
            
            # Weight: stronger for closer pairs
            # w = 1 / d_ij
            weights = np.where(mask, 1.0 / dist_sq, 0.0)
            
            # force_i = sum_j diff[i,j] * weights[i,j]
            forces += np.sum(diff * weights[:, :, np.newaxis], axis=1)
            
            # Scale forces
            forces *= lr
        
        # Add noise (Simulated Annealing style)
        noise = np.random.normal(0, temp, size=points.shape)
        points += forces + noise
        
        # Project back to disk (hard constraint)
        new_dists = np.linalg.norm(points, axis=1)
        outside_mask = new_dists > 1.0
        if np.any(outside_mask):
            points[outside_mask] *= (1.0 / new_dists[outside_mask, np.newaxis]) * 0.999
        
        # Decay temperature and adjust learning rate
        temp *= 0.995
        # If we are making progress, keep lr; if not, maybe reduce?
        # Simple adaptive: if current_r > best_r (improving), keep lr. Else reduce slightly?
        # Let's keep it simple: constant lr, decaying temp.
        
    return best_points, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 120.0 # Safety margin (limit is 150s)
    
    # 1. Initialize with Hexagonal Lattice
    points = _hexagonal_init(N)
    current_r = _compute_min_radius(points)
    
    best_points = points.copy()
    best_r = current_r
    
    # 2. Multi-Start Local Search
    # We run local search multiple times.
    # First run on the hex init.
    # Then, if time permits, we can try perturbing the best solution or restarting.
    
    # Run 1: Direct optimization of hex init
    if time.time() < deadline - 5.0:
        new_points, new_r = _local_search(points, max_iters=3000, time_limit=10.0)
        if new_r > best_r:
            best_points = new_points
            best_r = new_r
    
    # Run 2: Perturb best and optimize
    # Create a perturbed version of the best solution
    if time.time() < deadline - 10.0:
        perturbed_points = best_points + np.random.normal(0, 0.01, size=best_points.shape)
        # Project to disk
        dists = np.linalg.norm(perturbed_points, axis=1)
        outside = dists > 1.0
        if np.any(outside):
            perturbed_points[outside] *= (1.0 / dists[outside, np.newaxis]) * 0.999
        
        new_points, new_r = _local_search(perturbed_points, max_iters=3000, time_limit=10.0)
        if new_r > best_r:
            best_points = new_points
            best_r = new_r
            
    # Run 3: Another restart with different noise
    if time.time() < deadline - 15.0:
        # More aggressive perturbation
        perturbed_points = best_points + np.random.normal(0, 0.05, size=best_points.shape)
        dists = np.linalg.norm(perturbed_points, axis=1)
        outside = dists > 1.0
        if np.any(outside):
            perturbed_points[outside] *= (1.0 / dists[outside, np.newaxis]) * 0.999
        
        new_points, new_r = _local_search(perturbed_points, max_iters=3000, time_limit=10.0)
        if new_r > best_r:
            best_points = new_points
            best_r = new_r
            
    # Run 4: Fine-tune the best solution with very small steps
    if time.time() < deadline - 5.0:
        # Use a very short local search with small lr to polish
        # We can hack _local_search to take lr, or just run it again
        # Let's just run it again, the noise will be small if we could control it, 
        # but here we rely on the internal decay.
        # To ensure polishing, we can run a short search.
        new_points, new_r = _local_search(best_points, max_iters=1000, time_limit=5.0)
        if new_r > best_r:
            best_points = new_points
            best_r = new_r

    return best_points
# EVOLVE-BLOCK-END
