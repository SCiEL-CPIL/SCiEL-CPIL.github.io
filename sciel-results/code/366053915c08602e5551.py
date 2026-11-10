import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # 1. Initialization: Hexagonal lattice
    # We want a dense packing.
    # The seed uses a loop to find a spacing that yields >= N points.
    # We can do this more directly or just use a fixed dense grid and pick the best N.
    
    def get_initial_points():
        # Generate a hexagonal grid of points
        # Spacing s. We want roughly N points in area pi.
        # Area of hexagon cell is s^2 * sqrt(3)/2.
        # N * s^2 * sqrt(3)/2 ~ pi  => s ~ sqrt(2*pi / (N*sqrt(3)))
        s = np.sqrt(2.0 * np.pi / (N * np.sqrt(3.0)))
        
        k = int(1.0 / s) + 2
        points = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) <= 1.0:
                    points.append((x, y))
        
        points = np.array(points)
        
        # If we have more than N points, pick the N "best" ones.
        # "Best" usually means closest to center or forming a stable cluster.
        # Picking closest to center is a good heuristic for packing.
        if len(points) > N:
            dists = np.hypot(points[:, 0], points[:, 1])
            idx = np.argsort(dists, kind='stable')[:N]
            points = points[idx]
        elif len(points) < N:
            # Fallback: random or expand grid
            # This shouldn't happen with the formula above, but just in case
            points = np.random.uniform(-1, 1, (N, 2))
            norms = np.hypot(points[:, 0], points[:, 1])
            points = points / norms[:, None] * 0.9
            
        return points

    def evaluate(points):
        # Compute the minimum radius
        # r = min( min_{i<j} dist/2, min_i (1 - norm) )
        if len(points) < 2:
            return 1.0 - np.hypot(*points[0])
        
        # Pairwise distances
        # Using broadcasting
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Boundary distance
        norms = np.linalg.norm(points, axis=1)
        min_boundary_dist = np.min(1.0 - norms)
        
        return min(min_pair_dist / 2.0, min_boundary_dist)

    def optimize(points, max_iters=10000):
        # Soft-min gradient ascent
        # Objective: Maximize soft_min( distances )
        # soft_min(x) = -1/lambda * log(sum(exp(-lambda * x)))
        
        points = points.copy()
        best_points = points.copy()
        best_r = evaluate(points)
        
        # Annealing parameters
        # lambda controls the "sharpness" of the min. High lambda -> focuses on very small distances.
        # We start with moderate lambda and increase it? Or start high and decrease?
        # Actually, for maximization of min, we want to focus on the bottlenecks.
        # Let's start with lambda=20 and keep it or anneal it slightly.
        # Step size alpha.
        
        lambda_val = 20.0
        alpha = 0.05
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
                
            # Compute distances
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, 0.0) # Will be ignored by weights
            
            norms = np.linalg.norm(points, axis=1)
            bound_dists = 1.0 - norms
            
            # Combine all constraints into one vector for uniform processing?
            # Or handle separately.
            # Let's handle separately for clarity.
            
            # Weights for pairs
            # exp(-lambda * dist)
            # To avoid underflow/overflow, shift by max?
            # Since we want to maximize the MIN, we care about SMALL distances.
            # So exp(-lambda * small_dist) is LARGE.
            # We want to move points to INCREASE the small distances.
            
            # Gradient of -log(sum(exp(-lambda * d))) w.r.t d is -lambda * exp(-lambda * d) / sum(exp(-lambda * d))
            # Wait.
            # f = -1/lambda * log(sum(exp(-lambda * d)))
            # df/dd = -1/lambda * (1/sum) * sum(exp(-lambda * d) * (-lambda))
            # df/dd = sum(exp(-lambda * d)) / sum(exp(-lambda * d)) = 1.
            # This is for a single variable.
            
            # For vector d, the gradient w.r.t the parameter that increases d is the weight.
            # We want to move points to increase the distances.
            # So we add alpha * weight * direction_to_increase_distance.
            
            # Pairwise
            # d_ij = ||c_i - c_j||
            # dd_ij / dc_i = (c_i - c_j) / d_ij
            # We want to increase d_ij, so we move c_i in direction (c_i - c_j).
            
            # Boundary
            # b_i = 1 - ||c_i||
            # db_i / dc_i = - c_i / ||c_i||
            # We want to increase b_i, so we move c_i in direction -c_i (towards center).
            
            # Calculate weights
            # Mask diagonal
            mask = ~np.eye(N, dtype=bool)
            
            # Pair weights
            # exp(-lambda * dists)
            # We only care about pairs that are "close" to the minimum.
            # Using log-sum-exp for stability is good, but for N=50, simple exp is fine if we scale.
            # Let's normalize by the sum of exps.
            
            # To prevent numerical issues, we can just use the raw exps if lambda isn't too large.
            # Or we can subtract the min dist before exp?
            # No, we want the relative weights based on absolute distance.
            
            # Let's compute the gradient vector for each point.
            grads = np.zeros_like(points)
            
            # Pairwise contribution
            # For each i, sum over j != i of w_ij * (c_i - c_j) / d_ij
            # w_ij = exp(-lambda * d_ij) / Z_pair
            
            # Vectorized computation of forces
            # diff is (N, N, 2)
            # dists is (N, N)
            
            # Weights matrix W (N, N)
            # W_ij = exp(-lambda * dists_ij) for i!=j
            # We can compute this directly.
            
            # To avoid overflow/underflow, we can shift.
            # But since we want to emphasize SMALL distances, and exp(-lambda * x) decays for large x,
            # the values for large distances will be ~0, which is fine.
            # The values for small distances will be ~1.
            
            W = np.exp(-lambda_val * dists)
            W[~mask] = 0.0 # Zero out diagonal
            
            Z_pair = np.sum(W)
            if Z_pair > 0:
                W = W / Z_pair
                
                # Force on i from j: W_ij * (c_i - c_j) / d_ij
                # Note: dists is symmetric.
                # unit_vec = diff / dists[:,:,None]
                # But dists can be 0 if points coincide.
                # Add epsilon to dists for division.
                safe_dists = np.where(dists > 1e-8, dists, 1.0)
                unit_diff = diff / safe_dists[:, :, None]
                
                # Force_i = sum_j W_ij * unit_diff_ij
                # unit_diff_ij = (c_i - c_j) / d_ij
                # So we sum W_ij * (c_i - c_j) / d_ij
                # This is effectively a weighted sum of unit vectors pointing away from neighbors.
                
                # Multiply W by unit_diff and sum over j (axis 1)
                # W is (N, N). unit_diff is (N, N, 2).
                # We want sum over j for each i.
                # So we do: np.sum(W[:, :, None] * unit_diff, axis=1)
                
                pair_grads = np.sum(W[:, :, None] * unit_diff, axis=1)
                grads += pair_grads
            
            # Boundary contribution
            # B_i = exp(-lambda * (1 - norm_i))
            # We want to move towards center if close to boundary.
            # Direction is -c_i / norm_i.
            
            B = np.exp(-lambda_val * bound_dists)
            Z_bound = np.sum(B)
            if Z_bound > 0:
                B = B / Z_bound
                safe_norms = np.where(norms > 1e-8, norms, 1.0)
                # Direction to center is -points / norms
                bound_grads = -points / safe_norms[:, None]
                # Weighted sum
                grads += B[:, None] * bound_grads
            
            # Update
            points += alpha * grads
            
            # Project to unit circle
            norms = np.linalg.norm(points, axis=1)
            too_far = norms > 1.0
            if np.any(too_far):
                points[too_far] /= norms[too_far, None]
            
            # Evaluate
            r = evaluate(points)
            if r > best_r:
                best_r = r
                best_points = points.copy()
            
            # Annealing
            # Decay alpha to allow convergence
            alpha *= 0.995
            # Increase lambda to focus more on the minimum?
            # If we increase lambda, the weights become more concentrated on the absolute minimum.
            # This helps with fine-tuning.
            lambda_val *= 1.01
            
        return best_points, best_r

    # Initial points
    pts = get_initial_points()
    
    # Optimization
    best_pts, best_r = optimize(pts)
    
    # If time remains, try a few random restarts or perturbations
    # But usually, one good run with annealing is sufficient.
    # Let's check time.
    if time.time() < deadline - 10.0:
        # Try a second run with different initial perturbation
        # Perturb the best points found
        perturb_pts = best_pts + np.random.normal(0, 0.01, best_pts.shape)
        # Clip
        norms = np.linalg.norm(perturb_pts, axis=1)
        perturb_pts[norms > 0.95] /= (norms[norms > 0.95, None] / 0.95)
        
        pts2, r2 = optimize(perturb_pts)
        if r2 > best_r:
            best_pts = pts2
            best_r = r2

    return best_pts

# EVOLVE-BLOCK-END
