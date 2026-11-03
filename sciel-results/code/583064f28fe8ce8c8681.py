import numpy as np
import time
import math
from scipy.optimize import minimize

N = 200

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def compute_radius(centers):
        """Compute the maximum common radius for given centers."""
        if centers.shape != (N, 2):
            return 0.0
        if not np.all(np.isfinite(centers)):
            return 0.0
        # Wall distance
        wall_dists = 0.5 - np.abs(centers)
        min_wall = np.min(wall_dists)
        if min_wall < -1e-9:
            return 0.0
        
        # Pairwise distance
        # Use cdist for efficiency, but for gradient we need to be careful.
        # For the final score calculation, cdist is fine.
        from scipy.spatial.distance import cdist
        dist_matrix = cdist(centers, centers)
        np.fill_diagonal(dist_matrix, np.inf)
        min_pair = np.min(dist_matrix)
        
        r = min(min_wall, min_pair / 2.0)
        return max(0.0, r)

    def hexagonal_init():
        """Generate a hexagonal lattice initial guess."""
        # Target radius ~0.0366 -> d ~0.073
        # We want a grid that covers the square.
        d = 0.073
        y_spacing = d * math.sqrt(3) / 2
        
        # Estimate rows and cols to cover [-0.5, 0.5]
        # Width 1.0, Height 1.0
        cols = int(1.0 / d) + 2
        rows = int(1.0 / y_spacing) + 2
        
        pts = []
        # Center the grid
        # Total width approx (cols-1)*d
        # Total height approx (rows-1)*y_spacing
        
        # Let's just generate a generous grid and clip/filter
        for i in range(rows):
            y = -0.5 + (i + 0.5) * y_spacing
            if y > 0.5: break
            if i % 2 == 0:
                x_start = -0.5 + d/2
                for j in range(cols):
                    x = x_start + j * d
                    if x > 0.5: break
                    pts.append([x, y])
            else:
                x_start = -0.5 + d
                for j in range(cols):
                    x = x_start + j * d
                    if x > 0.5: break
                    pts.append([x, y])
        
        pts = np.array(pts)
        
        # If we have more than N, select N.
        # If we have less, add random or adjust.
        # Ideally, we want exactly N points for the optimizer to work on a fixed vector size.
        # If len(pts) != N, we need to handle it.
        # For L-BFGS-B, the dimension must be fixed.
        # Strategy: If too many, randomly subsample. If too few, add random points.
        # To be robust, let's force exactly N.
        
        if len(pts) > N:
            # Random subsample
            idx = np.random.choice(len(pts), N, replace=False)
            pts = pts[idx]
        elif len(pts) < N:
            diff = N - len(pts)
            # Add random points in the square
            random_pts = np.random.uniform(-0.45, 0.45, (diff, 2))
            pts = np.vstack([pts, random_pts])
            
        return pts

    def soft_min_objective(flat_centers, beta):
        """
        Objective function to minimize: -soft_min(distances)
        We want to maximize the minimum distance.
        soft_min(d) = -1/beta * log(sum(exp(-beta * d)))
        So we minimize -soft_min(d) = 1/beta * log(sum(exp(-beta * d)))
        
        Constraints:
        1. Centers inside [-0.5, 0.5]^2 (handled by bounds)
        2. We also need to account for the walls.
           The radius is min(min_pair_dist/2, min_wall_dist).
           So we want to maximize min(min_pair_dist/2, min_wall_dist).
           
           Let D_pair = distances between centers.
           Let D_wall = distances to walls.
           
           We want to maximize r such that all pair dists >= 2r and all wall dists >= r.
           This is equivalent to maximizing the soft-min of:
           [d_ij/2 for all i<j] and [d_iw for all i, w]
           
           Let's construct the vector of all "constraints" values:
           V = [d_ij/2 ... d_1w ... d_Nw]
           We want to maximize min(V).
           So we minimize -soft_min(V).
        """
        centers = flat_centers.reshape(N, 2)
        
        # Pairwise distances / 2
        # To avoid O(N^2) memory explosion in gradient, we can compute it, 
        # but N=200 is 40k pairs. 40k exps is fine.
        # dist_matrix = cdist(centers, centers)
        # We only need upper triangle.
        dists = np.zeros((N, N))
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(centers[i] - centers[j])
                dists[i, j] = d
                dists[j, i] = d
        
        # Wall distances
        wall_dists = 0.5 - np.abs(centers) # shape (N, 2)
        # Flatten wall dists
        wall_flat = wall_dists.flatten()
        
        # Pair flat (upper triangle)
        pair_flat = dists[np.triu_indices(N, k=1)] / 2.0
        
        # Combine
        # We want to maximize the minimum of these.
        # Values: pair_flat (N*(N-1)/2) and wall_flat (2*N)
        all_vals = np.concatenate([pair_flat, wall_flat])
        
        # Soft min: -1/beta * log(sum(exp(-beta * val)))
        # To avoid overflow/underflow, shift by max
        # But we are minimizing -soft_min, so we want to maximize the min.
        # Let's just compute soft_min directly and return negative.
        
        # Numerical stability:
        # logsumexp(-beta * vals) = max(-beta*vals) + log(sum(exp(-beta*vals - max(-beta*vals))))
        # Let m = max(-beta * vals) = -beta * min(vals)
        # Then soft_min = -1/beta * (m + log(sum(exp(-beta*vals - m))))
        # = min(vals) - 1/beta * log(sum(exp(-beta*vals - m)))
        
        # We want to minimize -soft_min = -min(vals) + 1/beta * log(sum(exp(-beta*vals - m)))
        
        # Actually, standard trick for "maximize min":
        # Minimize f(x) = 1/beta * log(sum(exp(-beta * v_i)))
        # This approximates -min(v_i) as beta -> inf.
        
        # Let's use the stable version:
        # f = 1/beta * logsumexp(-beta * all_vals)
        
        # logsumexp(z) = max(z) + log(sum(exp(z - max(z))))
        z = -beta * all_vals
        z_max = np.max(z)
        # If z_max is very large, exp(z - z_max) is fine.
        # sum_exp = np.sum(np.exp(z - z_max))
        # log_sum_exp = z_max + np.log(sum_exp)
        # f = (1.0 / beta) * log_sum_exp
        
        # However, we also need the gradient for L-BFGS-B.
        # It is much faster to provide the gradient analytically.
        # But writing the full analytic gradient for 40k pairs is tedious.
        # L-BFGS-B uses finite differences if jac is None.
        # N=200 -> 400 variables.
        # Finite difference gradient requires 401 evaluations of the objective.
        # Each evaluation is O(N^2) ~ 40k ops.
        # 400 * 40k = 16M ops per gradient step.
        # L-BFGS-B might take 100-200 iterations.
        # 16M * 200 = 3.2B ops. This might be too slow in Python loops.
        
        # We MUST provide the gradient or use a faster objective.
        # Let's implement the gradient analytically.
        
        # Objective: f = (1/beta) * log( sum_i exp(-beta * v_i) )
        # df/dv_i = (1/beta) * (1/sum) * (-beta * exp(-beta * v_i)) = -exp(-beta * v_i) / sum
        # Let p_i = exp(-beta * v_i) / sum(exp(-beta * v_j))
        # Then df/dv_i = -p_i
        
        # v_i are either d_ij/2 or d_iw.
        
        # 1. Compute p_i for all constraints
        # z = -beta * all_vals
        # p = exp(z) / sum(exp(z))
        # Use stable log-sum-exp
        # log_p_i = z_i - log_sum_exp
        # p_i = exp(log_p_i)
        
        # 2. Chain rule
        # For pair constraint v_k = d_ij/2 = 0.5 * ||c_i - c_j||
        # dv_k/dc_i = 0.5 * (c_i - c_j) / ||c_i - c_j||
        # dv_k/dc_j = 0.5 * (c_j - c_i) / ||c_i - c_j||
        
        # For wall constraint v_k = 0.5 - c_i_x (if x>0) or c_i_x + 0.5 (if x<0)
        # dv_k/dc_i_x = -sign(c_i_x)
        
        # Let's compute gradients.
        
        # Initialize gradient array (N, 2)
        grad = np.zeros((N, 2))
        
        # Compute p values
        # We need to map back from all_vals to indices.
        # all_vals = [pair_flat, wall_flat]
        
        # Compute z and p
        z = -beta * all_vals
        z_max = np.max(z)
        exp_z = np.exp(z - z_max)
        sum_exp = np.sum(exp_z)
        p = exp_z / sum_exp
        
        # Objective value
        # f = (1/beta) * (z_max + log(sum_exp))
        obj_val = (1.0 / beta) * (z_max + np.log(sum_exp))
        
        # Gradient accumulation
        # Pair part: first N*(N-1)/2 elements
        n_pairs = N * (N - 1) // 2
        p_pairs = p[:n_pairs]
        p_walls = p[n_pairs:]
        
        # Map pairs back to (i, j)
        # We need to iterate or use vectorization.
        # Vectorization is hard because of the norms.
        # Let's do a loop for pairs, but it's 20k iterations.
        # 20k iterations in Python is fast (~10ms).
        
        # Precompute distances to avoid re-computing norms
        # We already have dists matrix.
        
        for k in range(n_pairs):
            # Find i, j for k-th pair in upper triangle
            # k = i*(N-1) - i*(i-1)/2 + (j-i-1) ? No.
            # Standard unrolling:
            # k = 0 -> (0,1)
            # k = 1 -> (0,2)
            # ...
            # k = N-2 -> (0, N-1)
            # k = N-1 -> (1, 2)
            
            # Inverse of upper triangle index
            # i = floor((N - 0.5) - sqrt((N-0.5)^2 - 2k)) - 1 ?
            # Let's just use a precomputed mapping or simple loop.
            # Since N=200, n_pairs ~ 20k.
            # A simple loop to find i,j is O(N) each -> 4M ops. Too slow.
            # We need O(1) mapping or precomputation.
            
            # Precompute i, j arrays
            # This is done outside the objective function ideally, but we are inside.
            # Let's pass i_idx, j_idx as closure variables or compute once.
            
            # For now, let's assume we have i_idx, j_idx available.
            # I will refactor to compute i_idx, j_idx once in construct().
            
            i = i_idx[k]
            j = j_idx[k]
            
            d = dists[i, j]
            if d < 1e-12:
                continue
            
            # Gradient of v_k = 0.5 * d
            # dv/dc_i = 0.5 * (c_i - c_j) / d
            # dv/dc_j = 0.5 * (c_j - c_i) / d
            
            factor = -p_pairs[k] * 0.5 / d
            
            diff = centers[i] - centers[j]
            
            grad[i] += factor * diff
            grad[j] -= factor * diff
            
        # Wall part
        # p_walls shape (2N)
        # First N are x-walls, next N are y-walls?
        # In wall_flat = wall_dists.flatten(), it's [x0, y0, x1, y1, ...]
        # So p_walls[2*i] is x-wall for i, p_walls[2*i+1] is y-wall for i.
        
        for i in range(N):
            # X wall
            px = p_walls[2*i]
            # v_x = 0.5 - |x_i|
            # If x_i > 0, v_x = 0.5 - x_i, dv/dx_i = -1
            # If x_i < 0, v_x = 0.5 + x_i, dv/dx_i = 1
            # If x_i == 0, subgradient, let's use 0 or random? 0 is safe.
            if centers[i, 0] > 0:
                grad[i, 0] += -px * (-1.0) # df/dv * dv/dx = -p * (-1) = p ?
                # Wait.
                # f = sum p_k v_k? No.
                # f = (1/beta) log sum exp(-beta v_k)
                # df/dv_k = -p_k
                # df/dx_i = sum_k df/dv_k * dv_k/dx_i
                # df/dx_i = -p_k * dv_k/dx_i
                
                # So grad[i,0] += -px * (-1) = px
            elif centers[i, 0] < 0:
                grad[i, 0] += -px * (1.0) # = -px
            else:
                grad[i, 0] += 0.0
                
            # Y wall
            py = p_walls[2*i+1]
            if centers[i, 1] > 0:
                grad[i, 1] += py
            elif centers[i, 1] < 0:
                grad[i, 1] += -py
            else:
                grad[i, 1] += 0.0
                
        return obj_val, grad.flatten()

    # Precompute pair indices
    i_idx = []
    j_idx = []
    for i in range(N):
        for j in range(i+1, N):
            i_idx.append(i)
            j_idx.append(j)
    i_idx = np.array(i_idx)
    j_idx = np.array(j_idx)

    def optimize_packing(centers_init, time_budget):
        """
        Run L-BFGS-B optimization with increasing beta.
        """
        current_centers = centers_init.copy()
        flat_centers = current_centers.flatten()
        
        # Beta schedule
        # Start with small beta (smooth), end with large beta (sharp)
        # beta_start = 50, beta_end = 500
        # Number of stages: 5
        stages = 5
        betas = np.linspace(20, 200, stages)
        
        best_r = compute_radius(current_centers)
        best_centers = current_centers.copy()
        
        for beta in betas:
            # Check time
            if time.time() > start_time + time_budget:
                break
                
            # Objective wrapper
            def obj_and_grad(x):
                return soft_min_objective(x, beta)
                
            # Bounds
            bounds = [(-0.5, 0.5)] * (2 * N)
            
            # L-BFGS-B
            # maxfun: limit function evaluations to save time
            # If 100 evals per stage, 5 stages = 500 evals.
            # Each eval ~ 10ms (due to python loop in grad). 500 * 10ms = 5s.
            # This is fast enough.
            
            try:
                res = minimize(
                    obj_and_grad, 
                    flat_centers, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 50, 'maxfun': 100, 'disp': False}
                )
                
                if res.success or res.fun < obj_and_grad(flat_centers)[0]:
                    flat_centers = res.x
                    current_centers = flat_centers.reshape(N, 2)
                    
                    # Update best
                    r = compute_radius(current_centers)
                    if r > best_r:
                        best_r = r
                        best_centers = current_centers.copy()
                        
            except Exception as e:
                # If optimization fails, continue with current centers
                pass

        # Final polish: Discrete pair-push on exact bottlenecks
        # This is similar to the original local search but only a few iterations
        # to fix any numerical errors from the smooth approximation.
        
        centers = best_centers.copy()
        for _ in range(50):
            if time.time() > start_time + time_budget:
                break
                
            r = compute_radius(centers)
            
            # Find bottleneck
            # Compute all distances
            dist_matrix = np.zeros((N, N))
            for i in range(N):
                for j in range(i+1, N):
                    d = np.linalg.norm(centers[i] - centers[j])
                    dist_matrix[i, j] = d
                    dist_matrix[j, i] = d
            
            # Wall dists
            wall_dists = 0.5 - np.abs(centers)
            min_wall = np.min(wall_dists)
            min_wall_idx = np.argmin(wall_dists)
            
            min_pair = np.min(dist_matrix)
            min_pair_idx = np.argmin(dist_matrix)
            i_min, j_min = np.unravel_index(min_pair_idx, (N, N))
            
            current_r = min(min_wall, min_pair / 2.0)
            
            if current_r <= r + 1e-10:
                break
                
            # Move to improve bottleneck
            if min_wall < min_pair / 2.0:
                # Move wall point
                idx = min_wall_idx // 2
                axis = min_wall_idx % 2
                # Move away from wall
                if axis == 0:
                    # x
                    if centers[idx, 0] > 0:
                        centers[idx, 0] -= 0.001
                    else:
                        centers[idx, 0] += 0.001
                else:
                    # y
                    if centers[idx, 1] > 0:
                        centers[idx, 1] -= 0.001
                    else:
                        centers[idx, 1] += 0.001
            else:
                # Move pair
                i, j = i_min, j_min
                if i == j: continue
                d = np.linalg.norm(centers[i] - centers[j])
                if d < 1e-10: continue
                dir_vec = (centers[i] - centers[j]) / d
                step = 0.001
                centers[i] += dir_vec * step
                centers[j] -= dir_vec * step
                
            # Clip to bounds
            centers = np.clip(centers, -0.5, 0.5)
            
            new_r = compute_radius(centers)
            if new_r > best_r:
                best_r = new_r
                best_centers = centers.copy()
                
        return best_centers, best_r

    # Main logic
    best_centers = None
    best_radius = 0.0
    
    # 1. Hexagonal init
    init_hex = hexagonal_init()
    centers, r = optimize_packing(init_hex, 120.0)
    if r > best_radius:
        best_radius = r
        best_centers = centers.copy()
        
    # 2. If time remains, try a slightly perturbed hex init
    if time.time() < deadline - 10:
        init_hex2 = hexagonal_init()
        # Perturb
        init_hex2 += np.random.randn(N, 2) * 0.01
        init_hex2 = np.clip(init_hex2, -0.49, 0.49)
        
        centers, r = optimize_packing(init_hex2, 20.0)
        if r > best_radius:
            best_radius = r
            best_centers = centers.copy()
            
    # Fallback
    if best_centers is None:
        best_centers = init_hex
        
    return best_centers

# EVOLVE-BLOCK-END
