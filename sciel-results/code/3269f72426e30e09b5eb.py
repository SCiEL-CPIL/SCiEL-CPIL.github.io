import numpy as np
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin for safety
    
    # Helper to compute the soft-min radius
    def soft_min_radius(points, beta):
        # points: (N, 2)
        # Compute pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Boundary distances
        radii = np.linalg.norm(points, axis=1)
        bound_dists = 1.0 - radii
        
        # Combine all constraints into one array
        # We want min( min_pair_dist, min_bound_dist )
        # Soft min: -1/beta * log(sum(exp(-beta * val)))
        
        # Flatten pair distances (upper triangle only to avoid double counting, 
        # but for soft min sum, double counting just scales the log term, 
        # effectively changing beta. We can just use all i!=j or just i<j.
        # Using i<j is more efficient.
        iu = np.triu_indices(N, 1)
        pair_dists = dists[iu]
        
        all_dists = np.concatenate([pair_dists, bound_dists])
        
        # Numerical stability for log-sum-exp
        # max_val = np.max(all_dists)
        # exp_terms = np.exp(-beta * (all_dists - max_val))
        # log_sum = np.log(np.sum(exp_terms)) + (-beta * max_val)
        # soft_min = -1.0 / beta * log_sum
        
        # Simpler stable version:
        # Since we want max(soft_min), which is equivalent to min(-soft_min)
        # Let's just compute the gradient directly or use the explicit formula.
        
        # Gradient of soft_min w.r.t. point k:
        # d/dp_k ( -1/beta * log( sum_j exp(-beta * d_j) ) )
        # = 1/beta * sum_j ( -beta * exp(-beta * d_j) / sum_j exp(-beta * d_j) ) * d/dp_k d_j
        # = - sum_j ( w_j * d/dp_k d_j )  where w_j = exp(-beta * d_j) / sum exp(...)
        
        # This is getting complex to vectorize perfectly in one shot without errors.
        # Alternative: Use a simpler penalty method.
        
        # Let's stick to a simpler heuristic: 
        # Maximize r by pushing apart.
        # Force model:
        # F_ij = (2r - d_ij) * (p_i - p_j) / d_ij  if d_ij < 2r
        # F_bound = (|p_i| - (1-r)) * p_i / |p_i| if |p_i| > 1-r
        
        # But r is not fixed.
        
        # Let's use the standard "Repulsion" algorithm which effectively maximizes the minimum distance.
        # 1. Initialize r to a small value.
        # 2. Iterate:
        #    a. Calculate forces to increase r? No, usually r is fixed and we check feasibility, or we maximize the min distance.
        
        # Let's maximize the minimum distance D = min( min_pair_dist, min_bound_dist ).
        # The score is D/2.
        # We can use a "soft" repulsion that approximates maximizing the min distance.
        
        # Objective: E = sum_{i<j} exp(-beta * d_ij) + sum_i exp(-beta * (1-|c_i|))
        # Minimizing E pushes points apart and away from boundary? 
        # No, exp(-beta * d) is small when d is large. So minimizing E maximizes d.
        # This is a valid approach.
        
        return all_dists, pair_dists, bound_dists, radii

    # Initialization: Hexagonal lattice
    def init_hex():
        # Generate a hex grid that fits in the circle
        # Approximate number of points in hex grid with spacing s:
        # Area ~ pi * (1/s * 2/s * sqrt(3)/2)^2 ? 
        # Let's just generate a grid and filter.
        s = 0.15  # Initial spacing
        points = []
        # Generate points in a square grid covering the circle
        # x from -1 to 1, y from -1 to 1
        # Hex offset
        for i in range(-20, 21):
            for j in range(-20, 21):
                x = i * s
                y = j * s * np.sqrt(3)
                if i % 2 != 0:
                    y += s * np.sqrt(3) / 2
                # Check if inside unit circle (with some margin)
                if x*x + y*y < 1.0:
                    points.append([x, y])
        
        points = np.array(points)
        # If we have more than N, select N closest to center? Or random?
        # Selecting closest to center is a good heuristic for packing.
        if len(points) > N:
            dists = np.linalg.norm(points, axis=1)
            idx = np.argsort(dists)[:N]
            points = points[idx]
        elif len(points) < N:
            # If not enough, add random points
            while len(points) < N:
                pt = np.random.uniform(-0.9, 0.9, 2)
                points = np.vstack([points, pt])
        
        return points

    p = init_hex()
    
    # Optimization Loop
    # We will minimize E = sum exp(-beta * d)
    # Gradient descent on p.
    
    beta = 10.0
    lr = 0.01
    
    best_score = 0.0
    best_p = p.copy()
    
    while time.time() < deadline:
        # Compute distances
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, 1e-9) # Avoid div by 0
        
        iu = np.triu_indices(N, 1)
        pair_dists = dists[iu]
        
        radii = np.linalg.norm(p, axis=1)
        bound_dists = 1.0 - radii
        
        # We want to maximize the minimum of these.
        # Using soft min: 
        # Let's use a penalty that is large when distances are small.
        # E = sum (1/d_ij)^2 + sum (1/(1-|c_i|))^2 ? 
        # This blows up at 0. Good for repulsion.
        
        # Gradient of sum (1/d_ij)^2 w.r.t p_i:
        # d/dp_i ( (p_i - p_j)^-2 * 2 ) ? 
        # Let d^2 = ||p_i - p_j||^2.
        # f = 1/d^2.
        # df/dp_i = -2/d^4 * d(d^2)/dp_i = -2/d^4 * 2(p_i - p_j) = -4(p_i - p_j)/d^4.
        # We want to MINIMIZE E, so we move in direction -grad E.
        # -grad E = 4(p_i - p_j)/d^4. This pushes i away from j. Correct.
        
        # For boundary: f = 1/(1-|c_i|)^2.
        # Let r_i = |c_i|. f = (1-r_i)^-2.
        # df/dr_i = 2(1-r_i)^-3.
        # dr_i/dp_i = p_i / r_i.
        # df/dp_i = 2(1-r_i)^-3 * p_i / r_i.
        # -grad E = -2(1-r_i)^-3 * p_i / r_i.
        # This pushes p_i towards origin (inward). Correct.
        
        # Combine gradients
        grad = np.zeros_like(p)
        
        # Pairwise forces
        # We need to sum contributions from all j != i.
        # Vectorized:
        # diff is (N, N, 2)
        # d2 = sum(diff^2, axis=2)
        # We want to compute grad_i = sum_{j != i} 4 * (p_i - p_j) / d_ij^4
        # Note: (p_i - p_j) is diff[i,j]
        # d_ij^4 is d2[i,j]^2
        
        d2 = np.sum(diff**2, axis=2)
        # Avoid division by zero
        d2[d2 < 1e-6] = 1e-6
        
        # Force factor: 4 / d2^2
        force_factor = 4.0 / (d2**2)
        
        # Gradient contribution from pairs:
        # grad_i += sum_j force_factor[i,j] * diff[i,j]
        # diff[i,j] = p_i - p_j
        # So grad += sum_j force_factor[i,j] * (p_i - p_j)
        
        # Use matrix multiplication?
        # grad = (force_factor * diff).sum(axis=1)
        # force_factor is (N,N), diff is (N,N,2)
        # Multiply element-wise then sum over j (axis 1)
        
        grad_pairs = (force_factor[:, :, np.newaxis] * diff).sum(axis=1)
        
        # Boundary forces
        # grad_i += -2(1-r_i)^-3 * p_i / r_i
        # radii = np.linalg.norm(p, axis=1)
        # r_safe = np.where(radii < 1e-6, 1e-6, radii)
        # factor = -2.0 * (1.0 - radii)**-3 / r_safe
        # grad_bound = factor[:, np.newaxis] * p
        
        # Wait, the boundary term should only be active if we are close to boundary?
        # The 1/(1-r)^2 term is always active but small when r is small.
        # This is fine.
        
        radii_safe = np.where(radii < 1e-6, 1e-6, radii)
        # Only apply if 1-radii is not too small to avoid overflow?
        # If 1-radii is large, force is small.
        # If 1-radii is small, force is huge.
        # Let's clip the denominator to avoid numerical issues.
        denom = np.maximum(1.0 - radii, 0.01)
        factor = -2.0 / (denom**3 * radii_safe)
        grad_bound = factor[:, np.newaxis] * p
        
        grad = grad_pairs + grad_bound
        
        # Update
        p -= lr * grad
        
        # Re-center? 
        # The forces are symmetric, so center of mass should stay near 0.
        # But numerical errors might drift it.
        p -= np.mean(p, axis=0)
        
        # Clamp to unit circle (hard constraint to prevent divergence)
        # If a point goes outside, project it back?
        # The gradient pushes it in, so it shouldn't go far outside.
        # But if lr is too big, it might overshoot.
        # Let's project points outside the circle back to the boundary.
        current_radii = np.linalg.norm(p, axis=1)
        outside_mask = current_radii > 1.0
        if np.any(outside_mask):
            p[outside_mask] = p[outside_mask] / current_radii[outside_mask, np.newaxis]
            
        # Calculate current score
        # min_pair_dist
        min_pair = np.min(pair_dists)
        # min_bound_dist
        min_bound = np.min(bound_dists)
        
        current_r = 0.5 * min(min_pair, min_bound)
        
        if current_r > best_score:
            best_score = current_r
            best_p = p.copy()
            
        # Adaptive parameters
        # Increase beta? No, this method doesn't use beta.
        # Decrease learning rate
        lr *= 0.999
        if lr < 1e-5:
            lr = 1e-5
            
        # Check time
        if time.time() > deadline - 2.0:
            break
            
    # Final check
    # Ensure validity
    final_dists = np.linalg.norm(best_p[:, np.newaxis, :] - best_p[np.newaxis, :, :], axis=2)
    np.fill_diagonal(final_dists, np.inf)
    min_pair_final = np.min(final_dists)
    min_bound_final = np.min(1.0 - np.linalg.norm(best_p, axis=1))
    
    r_final = 0.5 * min(min_pair_final, min_bound_final)
    
    # If the result is invalid (e.g. overlapping), fall back to hex lattice
    if r_final <= 0 or not np.isfinite(r_final):
         return init_hex()[:N]
         
    return best_p

# EVOLVE-BLOCK-END
