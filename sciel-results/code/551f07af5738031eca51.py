import numpy as np
import time

N = 200

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_radius = 0.0
    best_points = None
    
    # Helper to compute the raw score (radius)
    def compute_radius(pts):
        # pts shape (N, 2)
        # 1. Min distance to boundary
        # Square is [-0.5, 0.5]
        # dist to left: x + 0.5, right: 0.5 - x, etc.
        # min dist to boundary = min(0.5 - max(|x|, |y|) for all points? No.
        # For each point i, dist to boundary is min(0.5 - |x_i|, 0.5 - |y_i|)
        # Wait, the distance from a point (x,y) to the boundary of the square [-0.5, 0.5] is:
        # d_left = x - (-0.5) = x + 0.5
        # d_right = 0.5 - x
        # d_bottom = y + 0.5
        # d_top = 0.5 - y
        # min_dist_boundary_i = min(d_left, d_right, d_bottom, d_top)
        # Global min_dist_boundary = min(min_dist_boundary_i)
        
        # 2. Min distance between pairs
        # dist(i,j) = |p_i - p_j|
        # Global min_pair_dist = min(dist(i,j) for i<j)
        
        # Radius r = 0.5 * min(min_dist_boundary, min_pair_dist)
        
        # Efficient calculation
        # Boundary distances
        # x in [-0.5, 0.5], y in [-0.5, 0.5]
        # d_b = min(0.5 - np.abs(pts[:, 0]), 0.5 - np.abs(pts[:, 1]))
        # Actually, 0.5 - |x| is the distance to the nearest vertical wall?
        # If x=0.4, dist to right wall (0.5) is 0.1, to left (-0.5) is 0.9. Min is 0.1.
        # 0.5 - |0.4| = 0.1. Correct.
        # So min_dist_boundary = min(0.5 - np.abs(pts[:, 0]), 0.5 - np.abs(pts[:, 1]))
        
        d_bound_x = 0.5 - np.abs(pts[:, 0])
        d_bound_y = 0.5 - np.abs(pts[:, 1])
        min_bound = np.min(np.minimum(d_bound_x, d_bound_y))
        
        # Pairwise distances
        # Use scipy or numpy broadcasting. N=200 is small enough for NxN matrix.
        # But for speed in loop, we can use a trick or just numpy.
        # dist_matrix = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
        # np.fill_diagonal(dist_matrix, np.inf)
        # min_pair = np.min(dist_matrix)
        
        # Faster:
        # We only need the minimum.
        # We can compute it in chunks or just use the full matrix since N=200 is 40k floats, which is fast.
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        min_pair_sq = np.min(dist_sq)
        min_pair = np.sqrt(min_pair_sq)
        
        return 0.5 * min(min_bound, min_pair)

    def soft_min_radius(pts, beta):
        """
        Compute a smooth approximation of the radius using log-sum-exp.
        We want to maximize r.
        r = 0.5 * min(d_bound, d_pair)
        log-sum-exp is for sum. 
        min(x) = -log-sum-exp(-x).
        So soft_min(x) = -1/beta * log(sum(exp(-beta*x)))
        
        We want to maximize soft_min_radius.
        This is equivalent to minimizing -soft_min_radius.
        
        However, gradient ascent on the raw min is non-smooth.
        Using soft min makes it smooth.
        
        Let's define the objective function to MAXIMIZE.
        F(pts) = 0.5 * soft_min(d_bound, d_pairs, beta)
        
        Gradient of soft_min:
        Let S = sum(exp(-beta * x_i))
        soft_min = -1/beta * log(S)
        d(soft_min)/dx_j = -1/beta * (1/S) * (-beta * exp(-beta * x_j))
                         = exp(-beta * x_j) / S
        
        This is a weighted average of gradients, where weights are proportional to exp(-beta * x_j).
        Smaller x_j (bottlenecks) get higher weight.
        
        d_bound components:
        d_bk = 0.5 - |c_k| (component-wise? No, min of 4 distances per point, then min over points)
        Actually, the boundary constraint is min over all points and all 4 walls.
        Let D_b be the vector of all boundary distances (4*N values).
        Let D_p be the vector of all pair distances (N*(N-1)/2 values).
        
        Total vector D = concat(D_b, D_p).
        soft_min(D) = -1/beta * log(sum(exp(-beta * D)))
        
        Gradient wrt point i:
        Sum of contributions from all D_k that depend on point i.
        
        This is computationally expensive to compute analytically for all pairs if done naively with high beta.
        But N=200 is small.
        
        Alternative: Use a simpler heuristic gradient.
        For each point, calculate the "pressure" from nearest neighbors and walls.
        Push away from nearest neighbors and walls.
        
        Let's try the analytic soft-min gradient.
        """
        # Calculate all distances
        # Boundary distances
        # For each point i:
        # d_left = x_i + 0.5
        # d_right = 0.5 - x_i
        # d_bottom = y_i + 0.5
        # d_top = 0.5 - y_i
        # We take the min of these 4 for each point, then min over points?
        # No, the global min is the min of ALL these 4*N distances.
        
        # Let's construct the vector of all constraints.
        # This might be memory heavy if we store all N^2 pairs explicitly for gradient.
        # N=200, pairs=19900. 4*N=800. Total ~20700 constraints.
        # This is fine.
        
        # 1. Boundary constraints
        x = pts[:, 0]
        y = pts[:, 1]
        d_left = x + 0.5
        d_right = 0.5 - x
        d_bottom = y + 0.5
        d_top = 0.5 - y
        
        # 2. Pair constraints
        # We need distances and gradients.
        # To save memory and time, we can iterate or use vectorized ops carefully.
        
        # Let's use a slightly different approach for the gradient:
        # The gradient of the soft-min radius is a weighted sum of unit vectors pointing away from the "active" constraints.
        # Weight w_k = exp(-beta * d_k) / sum(exp(-beta * d_j))
        
        # We will accumulate forces.
        forces = np.zeros_like(pts)
        
        # Accumulate boundary forces
        # For each point i, the constraint is min(d_left, d_right, d_bottom, d_top).
        # Actually, the global min is the min over ALL these.
        # So we treat each of the 4*N boundary distances as a separate constraint.
        
        # Let's compute all boundary distances and their gradients.
        # d_left_i = x_i + 0.5. Grad wrt x_i is 1.
        # d_right_i = 0.5 - x_i. Grad wrt x_i is -1.
        # d_bottom_i = y_i + 0.5. Grad wrt y_i is 1.
        # d_top_i = 0.5 - y_i. Grad wrt y_i is -1.
        
        # We will compute exp(-beta * d) for all these.
        
        # To avoid overflow/underflow, we can normalize by the max.
        # But let's just compute the values.
        
        # Collect all distances
        # d_vals list
        # grads list (tuple of (index, dx, dy))
        
        # This explicit list construction in Python is slow.
        # Let's use numpy vectorization where possible.
        
        # Boundary part:
        # d_b_all = np.concatenate([d_left, d_right, d_bottom, d_top])
        # This is 4*N values.
        
        # Pair part:
        # We need to compute pairwise distances.
        # diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        # dist_sq = np.sum(diff**2, axis=2)
        # dist = np.sqrt(dist_sq)
        # This creates a 200x200 matrix. 40000 floats.
        
        # Let's do it.
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        # Set diagonal to inf
        np.fill_diagonal(dist_sq, np.inf)
        dist = np.sqrt(dist_sq)
        
        # We only need upper triangle for unique pairs, but for the sum in log-sum-exp,
        # if we include both (i,j) and (j,i), the sum doubles, which just shifts the log by log(2).
        # The gradient weights will be consistent.
        # However, the gradient of dist(i,j) wrt p_i is (p_i - p_j)/dist(i,j).
        # And wrt p_j is (p_j - p_i)/dist(i,j) = -(p_i - p_j)/dist(i,j).
        
        # Let's compute the weights for all constraints.
        # All distances:
        # d_bound (4*N) and d_pair (N*N, with diagonal inf)
        
        # To compute the denominator Z = sum(exp(-beta * d_k)), we need all d_k.
        # d_bound_vals = np.concatenate([d_left, d_right, d_bottom, d_top])
        # d_pair_vals = dist.ravel() # Includes diagonal inf, exp(-inf)=0, so it's fine.
        
        # Z = sum(exp(-beta * d_bound_vals)) + sum(exp(-beta * d_pair_vals))
        
        # This is efficient in numpy.
        
        d_bound_vals = np.concatenate([d_left, d_right, d_bottom, d_top])
        d_pair_vals = dist.ravel()
        
        # Calculate exp terms
        # To prevent underflow, find min d.
        all_d = np.concatenate([d_bound_vals, d_pair_vals])
        min_d = np.min(all_d)
        
        # Shift by min_d
        exp_terms_bound = np.exp(-beta * (d_bound_vals - min_d))
        exp_terms_pair = np.exp(-beta * (d_pair_vals - min_d))
        
        Z_bound = np.sum(exp_terms_bound)
        Z_pair = np.sum(exp_terms_pair)
        Z = Z_bound + Z_pair
        
        # Probabilities (weights)
        # w_k = exp(-beta * d_k) / Z
        # Note: We shifted, so exp(-beta * (d_k - min_d)) / Z_shifted
        # Z_shifted = Z * exp(beta * min_d)
        # So w_k = exp_terms_k / Z_shifted
        # We can just use the normalized terms.
        
        # Force from boundaries:
        # For d_left (x+0.5), grad is (1, 0). Force is w * (1, 0).
        # For d_right (0.5-x), grad is (-1, 0). Force is w * (-1, 0).
        # etc.
        
        # w_left = exp_terms_bound[0:N] / Z
        # w_right = exp_terms_bound[N:2N] / Z
        # w_bottom = exp_terms_bound[2N:3N] / Z
        # w_top = exp_terms_bound[3N:4N] / Z
        
        w_left = exp_terms_bound[0:N] / Z
        w_right = exp_terms_bound[N:2*N] / Z
        w_bottom = exp_terms_bound[2*N:3*N] / Z
        w_top = exp_terms_bound[3*N:4*N] / Z
        
        # Accumulate forces
        forces[:, 0] += w_left - w_right
        forces[:, 1] += w_bottom - w_top
        
        # Force from pairs:
        # We have a matrix of exp terms: E_pair (N x N)
        # E_pair[i,j] = exp(-beta * (dist[i,j] - min_d))
        # Weight W[i,j] = E_pair[i,j] / Z
        # Gradient of dist[i,j] wrt p_i is (p_i - p_j) / dist[i,j]
        # Force on i from j is W[i,j] * (p_i - p_j) / dist[i,j]
        # Note: dist[i,i] is inf, so E_pair[i,i] is 0.
        
        # Vectorized calculation:
        # Unit vectors U[i,j] = (p_i - p_j) / dist[i,j]
        # dist is 0 on diagonal, so we need to handle division by zero.
        # But E_pair[i,i] is 0, so we can set dist[i,i] to 1 to avoid NaN, result will be 0.
        
        dist_safe = dist.copy()
        np.fill_diagonal(dist_safe, 1.0)
        
        # U = (pts[:, np.newaxis, :] - pts[np.newaxis, :, :]) / dist_safe[:, :, np.newaxis]
        # U shape (N, N, 2)
        
        # Force on i = sum_j W[i,j] * U[i,j]
        # W_pair = E_pair / Z
        # Force = np.sum(W_pair[:, :, np.newaxis] * U, axis=1)
        
        U = diff / dist_safe[:, :, np.newaxis]
        W_pair = exp_terms_pair.reshape(N, N) / Z
        pair_forces = np.sum(W_pair[:, :, np.newaxis] * U, axis=1)
        
        forces += pair_forces
        
        # The gradient of the soft-min radius is 0.5 * forces?
        # soft_min = -1/beta * log(Z_shifted)
        # d(soft_min)/dp = -1/beta * (1/Z_shifted) * d(Z_shifted)/dp
        # Z_shifted = sum exp(-beta(d_k - min_d)) = sum exp(-beta d_k) exp(beta min_d)
        # d(Z_shifted)/dp_i = sum_k exp(-beta d_k) exp(beta min_d) * (-beta) * d(d_k)/dp_i
        # d(soft_min)/dp_i = -1/beta * (1/Z_shifted) * (-beta) * sum_k exp(-beta d_k) exp(beta min_d) * d(d_k)/dp_i
        # = (1/Z_shifted) * sum_k exp(-beta d_k) exp(beta min_d) * d(d_k)/dp_i
        # = sum_k (exp(-beta d_k) / Z) * d(d_k)/dp_i   (since Z = Z_shifted / exp(beta min_d))
        # = sum_k w_k * d(d_k)/dp_i
        #
        # So the gradient of the soft-min *distance* is `forces`.
        # The radius is 0.5 * distance.
        # So gradient of radius is 0.5 * forces.
        
        return 0.5 * forces

    # Optimization Loop
    beta = 50.0  # Temperature for soft-min. Higher is sharper.
    lr = 0.01
    
    # Restart strategy
    # 1. Random uniform
    # 2. Jittered grid
    # 3. Hexagonal lattice (if fits)
    
    n_restarts = 50 # Adjust based on time
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
            
        # Initialize
        if restart == 0:
            # Jittered grid
            m = int(np.ceil(np.sqrt(N)))
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y) for y in ticks for x in ticks])
            pts = grid[:N].copy()
            # Add small jitter
            pts += np.random.normal(0, 0.01, pts.shape)
        else:
            # Random
            pts = np.random.uniform(-0.5, 0.5, (N, 2))
            
        # Clip to inside
        pts = np.clip(pts, -0.5, 0.5)
        
        # Optimize
        for step in range(2000):
            if time.time() > deadline:
                break
                
            # Compute gradient
            grad = soft_min_radius(pts, beta)
            
            # Update
            pts += lr * grad
            
            # Project to square
            pts = np.clip(pts, -0.5, 0.5)
            
            # Adaptive learning rate?
            # Simple decay
            if step % 100 == 0:
                lr *= 0.95
                
            # Check for improvement
            # We can check every few steps to avoid computing exact radius too often
            if step % 50 == 0:
                current_r = compute_radius(pts)
                if current_r > best_radius:
                    best_radius = current_r
                    best_points = pts.copy()
                    
        # Final check
        final_r = compute_radius(pts)
        if final_r > best_radius:
            best_radius = final_r
            best_points = pts.copy()

    # If we didn't find any valid points (should not happen), return grid
    if best_points is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
        
    return best_points
# EVOLVE-BLOCK-END
