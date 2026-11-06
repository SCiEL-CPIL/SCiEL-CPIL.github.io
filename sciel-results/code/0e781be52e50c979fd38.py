"""Bootstrap portfolio: MODEL-AND-OPTIMIZE for packing 300 equal circles in a unit circle."""
import numpy as np
import time

N = 300

def construct():
    # Deadline guard: 150s total, return best valid result before 140s
    t_start = time.time()
    deadline = t_start + 140.0
    
    # 1. Initialize with a hexagonal lattice (better than random)
    # Estimate spacing to get ~N points
    # Area of unit circle = pi. Area per circle ~ (sqrt(3)/2) * s^2
    # N * (sqrt(3)/2) * s^2 ~ pi  => s ~ sqrt(2*pi / (N*sqrt(3)))
    s_init = np.sqrt(2 * np.pi / (N * np.sqrt(3)))
    
    # Generate hex lattice points
    points = []
    # Determine range
    max_i = int(1.0 / s_init) + 2
    for j in range(-max_i, max_i + 1):
        for i in range(-max_i, max_i + 1):
            x = (i + 0.5 * (j % 2)) * s_init
            y = j * s_init * np.sqrt(3.0) / 2.0
            if np.hypot(x, y) <= 1.0:
                points.append([x, y])
    
    points = np.array(points)
    
    # If we have too many, pick the N closest to the center (or just random subset if too few? 
    # Usually hex lattice gives more than N. Let's take the first N or those with smallest radius)
    if len(points) > N:
        radii = np.hypot(points[:, 0], points[:, 1])
        indices = np.argsort(radii)
        points = points[indices[:N]]
    elif len(points) < N:
        # Fallback: random uniform in disk
        # Re-init with random if lattice failed to generate enough
        theta = np.random.uniform(0, 2*np.pi, N)
        r = np.sqrt(np.random.uniform(0, 1, N))
        points = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
    
    # Ensure exactly N points
    if len(points) < N:
        # Pad with random points near boundary if needed (unlikely)
        needed = N - len(points)
        theta = np.random.uniform(0, 2*np.pi, needed)
        r = np.sqrt(np.random.uniform(0, 1, needed))
        new_pts = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
        points = np.vstack([points, new_pts])
    else:
        points = points[:N]

    # 2. Relaxation using simple gradient ascent on the score function
    # Score r = min( min_{i<j} |c_i - c_j|/2, min_i (1 - |c_i|) )
    # To maximize r, we can use a soft objective or simply iterative push-apart.
    # Let's use a standard "repulsion" scheme.
    
    # We will iterate:
    # 1. Compute pairwise distances
    # 2. For pairs closer than current target radius (or just always repel), push apart.
    # 3. For points too close to boundary, push in.
    
    # Target radius estimation:
    # Optimal r is approx 0.053. Start with a slightly smaller target to allow movement.
    # We want to maximize the minimum distance.
    
    # Use a simple iterative relaxation:
    # For each point, compute net force from repulsion from neighbors and attraction to center (weak) or just constraint.
    # Actually, a better approach for "equal circles" is to maximize the minimum distance.
    # Let's use a "best improvement" or "random restart + local search" style, but simple physics is faster to code correctly.
    
    # Let's use a simple iterative repulsion algorithm.
    # We don't know the exact r yet. We can estimate it as half the average nearest neighbor distance, or just start with a guess.
    # Better: We can optimize the positions to maximize the minimum distance.
    
    # Let's try a simple coordinate descent or gradient ascent on the "soft" minimum.
    # But simple repulsion is robust.
    
    # Initialize a target radius. 
    # Area argument: pi * 1^2 / N = pi/300. r_max ~ sqrt(pi/300 / (pi/3))? No.
    # Hex packing density: pi / (2*sqrt(3)) ~ 0.9069.
    # Area of N circles: N * pi * r^2 <= 0.9069 * pi * 1^2
    # r^2 <= 0.9069 / N
    # r <= sqrt(0.9069 / 300) ~ sqrt(0.003023) ~ 0.055
    # This matches the known 0.053.
    
    r_target = np.sqrt(0.90 / N) # Start a bit low
    
    # We will iterate, increasing r_target slightly if we can fit, or just let the repulsion find the max.
    # Actually, let's just run a repulsion force simulation for a fixed number of steps.
    
    num_steps = 5000 # Adjust based on time budget
    
    # Precompute for speed? N=300, N^2=90k ops per step. 5000 steps = 450M ops. 
    # In Python/NumPy, 5000 steps of N=300 might be slow if not vectorized well.
    # Let's vectorize the pairwise distance calculation.
    
    # To be safe with time, we'll check time every 100 steps.
    
    # Current positions
    pos = points.copy()
    
    # We want to maximize the minimum distance. 
    # A common trick is to maximize the sum of inverse distances (repulsive potential) or similar.
    # Or simply: for each pair, if dist < 2*r, push apart. But what is r?
    # Let's define a "potential" that penalizes closeness.
    # V = sum_{i<j} 1/d_{ij}^2 + sum_i (|c_i| - 1)^2 * lambda (if outside)
    # Minimizing V maximizes distances.
    
    # However, calculating all pairs every step is O(N^2).
    # For N=300, that's 45k pairs. Vectorized numpy can handle this.
    
    # Let's use a simpler heuristic: 
    # 1. Find the current minimum distance d_min.
    # 2. Set a target separation d_target = d_min * 1.01 (or similar).
    # 3. Push pairs closer than d_target apart.
    # 4. Push points outside boundary in.
    
    # This is a "hard constraint" relaxation.
    
    # Initial d_min
    def calc_min_dist(pos):
        # Vectorized pairwise distance
        # (N, 1, 2) - (1, N, 2) -> (N, N, 2)
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        # Mask diagonal
        np.fill_diagonal(dist, np.inf)
        return np.min(dist)
    
    # This is expensive (N^2 memory and compute). 300x300 = 90k floats. 
    # 5000 iterations * 90k ops = 450M. Might be tight in 140s in pure Python/NumPy overhead?
    # Let's reduce steps or use a smarter update.
    
    # Alternative: Only update a few points at a time? No, vectorized is better.
    # Let's try 1000 steps first and see.
    
    # We can also use a simpler "local" repulsion if we track neighbors, but that's complex.
    # Let's stick to full vectorized pairwise for simplicity and correctness, but limit steps.
    
    # To speed up, we can use a "soft" repulsion that doesn't require finding the exact min dist every time?
    # No, finding min dist is O(N^2).
    
    # Let's try a different objective: Maximize the sum of log(distances) or just 1/distance^2.
    # This is smooth and differentiable. We can use gradient ascent.
    # Grad of sum_{i<j} -2/d_{ij}^2 w.r.t pos_i is sum_{j!=i} 2*(pos_j - pos_i)/d_{ij}^4 * 2?
    # Let F = sum_{i<j} 1/d_{ij}^2. We want to minimize F.
    # dF/dpos_i = sum_{j != i} d/dpos_i (1/d_{ij}^2) = sum_{j != i} -2 * (pos_i - pos_j) / d_{ij}^4
    # So gradient descent step: pos_i -= lr * grad_i
    
    # Boundary constraint: if |pos_i| > 1 - r_est, push in.
    # What is r_est? We can use half the current min distance.
    
    # Let's implement Gradient Descent on F = sum 1/d^2.
    
    # Precompute indices for i < j? Or just use full matrix and ignore diagonal.
    # Using full matrix is easier to vectorize.
    
    # F = sum_{i,j, i!=j} 1/d_{ij}^2 / 2 (since i<j and j<i are both in sum)
    
    # Let's run for a fixed number of iterations, checking time.
    
    lr = 0.001 # Learning rate
    
    # We need to be careful with d=0. Add small epsilon.
    
    for step in range(2000):
        if time.time() > deadline - 10:
            break
            
        # Compute pairwise differences
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist_sq, 1.0) # Avoid division by zero, diagonal won't be used in grad
        
        # Inverse squared distance
        inv_dist_sq = 1.0 / dist_sq
        
        # Gradient of F = sum 1/d^2
        # dF/dpos_i = -2 * sum_{j != i} (pos_i - pos_j) / d_{ij}^4
        # Note: diff[i,j] = pos_i - pos_j
        # So we need sum_j diff[i,j] / dist_sq[i,j]^2
        
        # dist_sq^2 = 1 / inv_dist_sq^2
        # So term is diff * inv_dist_sq^2
        
        # grad_i = -2 * sum_j (diff[i,j] * inv_dist_sq[i,j]^2)
        # But we must exclude j=i. Since we set dist_sq[i,i]=1, inv_dist_sq[i,i]=1.
        # We should zero out the diagonal in the summation.
        
        mask = 1.0 - np.eye(N)
        inv_dist_sq_masked = inv_dist_sq * mask
        
        # grad = -2 * sum_j (diff[i,j] * (inv_dist_sq_masked[i,j])^2)
        # Let's compute factor = inv_dist_sq_masked^2
        factor = inv_dist_sq_masked ** 2
        
        # Sum over j (axis 1)
        grad = -2.0 * np.sum(diff * factor[:, :, np.newaxis], axis=1)
        
        # Boundary force
        # If a point is too close to boundary, push it towards center.
        # "Too close" means |pos| > 1 - r_current
        # Estimate r_current as 0.5 * min_dist
        # We can compute min_dist from dist_sq
        min_dist = np.sqrt(np.min(dist_sq))
        r_est = 0.5 * min_dist
        
        radii = np.hypot(pos[:, 0], pos[:, 1])
        # Force is proportional to (radii - (1 - r_est)) if positive
        # Let's apply a strong penalty if outside the safe zone
        boundary_violation = np.maximum(0, radii - (1.0 - r_est))
        
        # Gradient of boundary penalty: if we use (r - r_max)^2, grad is 2(r-r_max) * pos/r
        # Let's just add a simple push-in vector
        # push = -boundary_violation * pos / radii (normalized direction)
        # Actually, let's just project points back if they go too far?
        # Or add a force.
        
        # Force = -grad * lr + boundary_force
        # Let's define boundary force as pointing to center, magnitude proportional to violation
        # f_b = -k * boundary_violation * (pos / radii)
        
        # To avoid division by zero in pos/radii
        pos_norm = np.where(radii > 1e-6, pos / radii[:, np.newaxis], 0.0)
        boundary_force = -10.0 * boundary_violation[:, np.newaxis] * pos_norm
        
        # Update
        pos = pos + lr * grad + boundary_force
        
        # Hard constraint: Ensure points are inside unit circle
        # If any point is outside, scale it back? Or just clip?
        # Clipping might cause overlaps. Scaling is better.
        new_radii = np.hypot(pos[:, 0], pos[:, 1])
        mask_out = new_radii > 1.0
        if np.any(mask_out):
            pos[mask_out] = pos[mask_out] / new_radii[mask_out, np.newaxis]
            
    # Final adjustment: Calculate the actual score
    # Score is min( min_pair_dist/2, min_boundary_dist )
    
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)
    
    radii = np.hypot(pos[:, 0], pos[:, 1])
    min_boundary_dist = 1.0 - np.max(radii)
    
    r_score = min(min_pair_dist / 2.0, min_boundary_dist)
    
    # If the score is bad, maybe try a few more steps or different init?
    # For now, return this.
    
    return pos

