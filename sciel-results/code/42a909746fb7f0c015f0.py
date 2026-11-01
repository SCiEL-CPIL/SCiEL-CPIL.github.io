import numpy as np
import time
from scipy.optimize import minimize

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for 150s limit
    
    # 1. Initialization: Hexagonal lattice
    # A hexagonal packing is denser than square.
    # We generate a grid of hex points and select the best 50 that fit in the square.
    # Alternatively, just generate a dense hex grid and take the first 50, then optimize.
    # Let's create a hex grid that covers the square [-0.5, 0.5]^2.
    
    # Approximate radius for hex grid to get ~50 points
    # Area = 1.0. Area per hex point ~ sqrt(3)/2 * r_hex^2.
    # 50 * sqrt(3)/2 * r_hex^2 = 1 => r_hex^2 = 2/(50*sqrt(3)) ~ 0.023 => r_hex ~ 0.15
    # Distance between centers in hex lattice:
    # dx = r_hex, dy = r_hex * sqrt(3)
    
    # Let's just generate a fine grid and pick 50 points that form a hex-like structure
    # Or simpler: generate a hex lattice with a specific spacing and center it.
    
    # Let's try a specific hex arrangement.
    # Rows of alternating length.
    # For N=50, a 7x8 or similar block might work.
    # Let's generate a hex grid with spacing s.
    # We want to maximize the minimum distance.
    
    # Simple initialization:
    # Place points in a hexagonal lattice.
    # s is the distance between adjacent points.
    # We want ~50 points in area 1.
    # s = sqrt(1 / (50 * sqrt(3)/2)) = sqrt(2/(50*1.732)) = sqrt(0.0231) = 0.152
    
    s = 0.152
    points = []
    # Generate hex grid
    # y steps: s * sqrt(3) / 2
    dy = s * np.sqrt(3) / 2
    dx = s
    
    # Iterate over rows
    max_y = 0.5
    min_y = -0.5
    row = 0
    while True:
        y = min_y + row * dy
        if y > max_y:
            break
        # Shift x for odd rows
        x_offset = 0 if row % 2 == 0 else dx / 2
        x = min_y + x_offset
        while x < max_y:
            points.append((x, y))
            x += dx
        row += 1
        if len(points) > 100: # Safety break
            break
            
    points = np.array(points)
    
    # We need exactly N points. 
    # If we have more, we should pick the ones that are most "central" or just take the first N?
    # Taking the first N from a grid starting at corner might bias the packing.
    # Better: Generate a larger grid, then randomly sample N points?
    # Or better: Generate a grid that is slightly smaller so it fits, or just use the first N but center the grid.
    
    # Let's re-center the grid to the origin.
    if len(points) > 0:
        center = np.mean(points, axis=0)
        points = points - center
        
        # Filter points that are strictly inside or close to the square
        # Actually, for initialization, we just need N distinct points.
        # If we have more than N, we can randomly select N to avoid bias.
        # If we have fewer, we need to add more.
        
        # Let's try a different init: 
        # Just use a square grid for simplicity and let the optimizer fix it, 
        # but hex is better.
        
        # Let's ensure we have at least N points.
        # If not, generate a denser grid.
        while len(points) < N:
             s *= 0.95
             dy = s * np.sqrt(3) / 2
             dx = s
             points = []
             row = 0
             while True:
                 y = -0.5 + row * dy
                 if y > 0.5:
                     break
                 x_offset = 0 if row % 2 == 0 else dx / 2
                 x = -0.5 + x_offset
                 while x < 0.5:
                     points.append((x, y))
                     x += dx
                 row += 1
                 if len(points) > 200:
                     break
             points = np.array(points)
             center = np.mean(points, axis=0)
             points = points - center
             
        # Randomly select N points to avoid symmetry bias
        idx = np.random.choice(len(points), N, replace=False)
        x0 = points[idx]
        
    else:
        # Fallback to square grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        x0 = grid[:N]

    # Flatten for optimizer
    x0_flat = x0.flatten()
    
    # Bounds
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    def objective_and_grad(x_flat, beta):
        x = x_flat.reshape(N, 2)
        
        # Compute pairwise distances
        # d_ij = ||x_i - x_j||
        # d_i_wall = min(0.5 - x_ix, 0.5 + x_ix, 0.5 - x_iy, 0.5 + x_iy)
        
        # Efficient distance calculation
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Wall distances
        wall_dists = np.min(np.column_stack([
            0.5 - x[:, 0],
            0.5 + x[:, 0],
            0.5 - x[:, 1],
            0.5 + x[:, 1]
        ]), axis=1)
        
        # Combine all distances
        # We want to maximize the minimum of all these distances.
        # Let D be the vector of all relevant distances.
        # Pairwise: N*(N-1)/2 values.
        # Walls: N values.
        
        # To avoid duplicating pairwise distances in the vector, we can take the upper triangle.
        iu = np.triu_indices(N, 1)
        pair_dists = dists[iu]
        
        D = np.concatenate([pair_dists, wall_dists])
        
        # Soft min: -1/beta * log(sum(exp(-beta * D)))
        # For numerical stability, subtract max(D)
        D_max = np.max(D)
        exp_terms = np.exp(-beta * (D - D_max))
        sum_exp = np.sum(exp_terms)
        
        if sum_exp == 0:
            # Fallback, shouldn't happen
            return -D_max, np.zeros_like(x_flat)
            
        obj = -(-1.0 / beta * np.log(sum_exp) - D_max)
        # Wait, soft_min = -1/beta * log(sum(exp(-beta D)))
        # Let's re-derive.
        # f(D) = min(D). Approx by -1/beta log(sum exp(-beta D)).
        # We want to MINIMIZE -soft_min(D) to MAXIMIZE soft_min(D).
        # So objective = 1/beta * log(sum exp(-beta D)).
        
        # Let's compute gradient of obj = 1/beta * log(sum exp(-beta D)).
        # Let S = sum exp(-beta D).
        # d obj / d D_k = 1/beta * 1/S * d S / d D_k
        # d S / d D_k = -beta exp(-beta D_k)
        # d obj / d D_k = -1/S * exp(-beta D_k) = - exp(-beta D_k) / S
        
        # Now we need d obj / d x.
        # d obj / d x = sum_k (d obj / d D_k) * (d D_k / d x)
        
        # This is getting complicated to vectorize perfectly without errors.
        # Let's use a simpler, robust objective:
        # Minimize sum of (r_target - d_k)^2 for d_k < r_target? No, that's for fixed r.
        
        # Let's stick to the soft-min but implement carefully.
        # Actually, there is a known issue with log-sum-exp for large beta causing under/overflow.
        # The subtraction of D_max handles it.
        
        # Let's re-evaluate the objective function.
        # We want to maximize r = min(D).
        # Equivalent to minimizing f(x) = -min(D).
        # Smooth approx: f(x) = 1/beta * log(sum(exp(-beta D_k))).
        
        # Gradient:
        # p_k = exp(-beta D_k) / S
        # d f / d D_k = -p_k
        # d f / d x = - sum_k p_k * (d D_k / d x)
        
        # Let's compute p_k first.
        p = exp_terms / sum_exp
        
        # Now, d D_k / d x.
        # D consists of pair_dists and wall_dists.
        
        # Gradient for wall distances:
        # d (0.5 - x_i) / d x_i = -1
        # d (0.5 + x_i) / d x_i = 1
        # etc.
        # Since we took min, the gradient is the gradient of the active constraint.
        # But soft-min mixes them.
        # d wall_dist_i / d x_i = 
        # If wall_dist_i is from left (0.5 - x), grad is -1.
        # If right (0.5 + x), grad is 1.
        # We can compute which one is the min and use that gradient, or use a smooth min of the 4 walls.
        # Let's use smooth min for walls too.
        
        # This is getting complex. Let's use a simpler strategy:
        # Use scipy.optimize.minimize with method 'Nelder-Mead' or 'Powell' which don't need gradients?
        # Nelder-Mead is slow for 100 dimensions.
        # L-BFGS-B is best.
        
        # Alternative: 
        # Objective: sum of squared overlaps assuming a target radius r?
        # No, we want to find the max r.
        
        # Let's go back to the standard approach for this task:
        # Maximize the minimum distance.
        # Use a penalty method?
        
        # Actually, there is a very effective trick:
        # Minimize the sum of (1/d_k^2) or similar? No.
        
        # Let's try the following objective:
        # f(x) = sum_{i<j} max(0, 2r - d_ij)^2 + sum_i max(0, r - w_i)^2
        # But r is a variable.
        
        # Let's use the auxiliary variable t.
        # Minimize -t
        # Subject to d_ij >= 2t, w_i >= t.
        # This is a convex problem? No, d_ij is not convex in x.
        
        # Okay, let's use the soft-min gradient correctly.
        
        # Gradient of soft-min of D:
        # g_D = -p (vector of probabilities)
        # We need to accumulate gradients into x.
        
        grad_x = np.zeros_like(x)
        
        # 1. Wall contributions
        # For each point i, wall_dist is min of 4 terms.
        # Let's compute smooth min of the 4 wall distances for each point.
        # w_i = smooth_min(w_i_left, w_i_right, w_i_down, w_i_up)
        # This is too nested.
        
        # Simpler: Just use the raw min for walls in the distance vector, 
        # and for the gradient, use the gradient of the active constraint.
        # Since we are using soft-min over ALL distances (pairs + walls), 
        # the weight p_k applies to the gradient of D_k.
        
        # For pair distance d_ij:
        # d d_ij / d x_i = (x_i - x_j) / d_ij
        # d d_ij / d x_j = (x_j - x_i) / d_ij
        
        # For wall distance w_i:
        # It is min of 4 linear functions.
        # The gradient is the gradient of the active one (or a mix if we used smooth min for walls).
        # Let's assume we use the hard min for walls in the D vector.
        # Then the gradient is 0 or +/- 1.
        # But the soft-min p_k distributes weight.
        # If p_k is significant for a wall constraint, we add its gradient.
        
        # Let's iterate over pairs and walls to build the gradient.
        # This is O(N^2) which is fine for N=50.
        
        # Pairwise gradients
        # p_pair is the first N*(N-1)/2 elements of p.
        # We need to map p_pair back to i,j.
        
        # It is easier to compute the gradient by looping.
        
        # Reset grad
        grad_x = np.zeros((N, 2))
        
        # Pairwise
        # p_pair corresponds to the upper triangle of dists.
        # Let's reconstruct the mapping.
        # idx 0: (0,1), idx 1: (0,2), ..., idx 4: (0,4), idx 5: (1,2)...
        
        # Actually, let's just compute the gradient of the objective with respect to x
        # by using the chain rule and vectorized operations where possible.
        
        # Let's try a different, simpler objective that is differentiable:
        # f(x) = sum_{i<j} (2r - d_ij)^2 for d_ij < 2r
        # This requires knowing r.
        
        # Let's stick to the soft-min but implement the gradient loop.
        
        # We have p (length N*(N-1)/2 + N).
        p_pair = p[:len(pair_dists)]
        p_wall = p[len(pair_dists):]
        
        # Gradient from pairs
        # For each pair (i,j) with weight p_k:
        # grad_x[i] += p_k * (x_i - x_j) / d_ij
        # grad_x[j] += p_k * (x_j - x_i) / d_ij
        
        # We can do this vectorized:
        # We have dists matrix.
        # We have p_pair.
        # We need to scatter the gradients.
        
        # Create a matrix of weights W where W[i,j] = p_k if (i,j) is in upper triangle, 0 otherwise.
        W = np.zeros((N, N))
        W[iu] = p_pair
        # Symmetrize? No, the formula is:
        # d f / d x_i = - sum_j p_{ij} * (x_i - x_j) / d_{ij}
        # Note that p_{ij} is only defined for i<j in our p_pair.
        # But the term (x_i - x_j)/d_{ij} is antisymmetric.
        
        # Let's define a matrix G where G[i,j] = p_{ij} / d_{ij} for i != j.
        # We need to be careful with the mapping.
        
        # Let's just loop. N=50, 1225 pairs. Loop is fast enough.
        pair_idx = 0
        for i in range(N):
            for j in range(i+1, N):
                w = p_pair[pair_idx]
                if w > 1e-12:
                    d = dists[i, j]
                    if d > 1e-12:
                        diff = x[i] - x[j]
                        grad_x[i] -= w * diff / d
                        grad_x[j] += w * diff / d
                pair_idx += 1
        
        # Gradient from walls
        # For each i, w_i = min(0.5-x_i0, 0.5+x_i0, 0.5-x_i1, 0.5+x_i1)
        # grad_x[i] -= p_wall[i] * grad(w_i)
        
        for i in range(N):
            w = p_wall[i]
            if w > 1e-12:
                # Determine active wall
                # We used the min value in D. We need the gradient of that specific min.
                # If multiple are equal, it's non-differentiable, but soft-min handles it probabilistically?
                # No, we used hard min for D construction.
                # So we should use the gradient of the active constraint.
                
                x0, x1 = x[i]
                walls = [
                    (0.5 - x0, -1, 0), # left
                    (0.5 + x0, 1, 0),  # right
                    (0.5 - x1, 0, -1), # bottom
                    (0.5 + x1, 0, 1)   # top
                ]
                min_val = np.inf
                min_grad = None
                for val, gx, gy in walls:
                    if val < min_val:
                        min_val = val
                        min_grad = (gx, gy)
                
                if min_grad:
                    grad_x[i, 0] -= w * min_grad[0]
                    grad_x[i, 1] -= w * min_grad[1]
                    
        return obj, grad_x.flatten()
    
    # Optimization loop with increasing beta
    x_best = x0_flat
    r_best = 0.0
    
    def compute_radius(x_flat):
        x = x_flat.reshape(N, 2)
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        wall_dists = np.min(np.column_stack([
            0.5 - x[:, 0],
            0.5 + x[:, 0],
            0.5 - x[:, 1],
            0.5 + x[:, 1]
        ]), axis=1)
        min_wall = np.min(wall_dists)
        
        return min(min_pair, min_wall)
    
    r_best = compute_radius(x_best)
    
    betas = [10, 50, 200, 1000]
    
    for beta in betas:
        if time.time() > deadline:
            break
            
        # Define objective for this beta
        def obj_func(x_flat):
            val, _ = objective_and_grad(x_flat, beta)
            return val
            
        def grad_func(x_flat):
            _, grad = objective_and_grad(x_flat, beta)
            return grad
            
        res = minimize(
            obj_func, 
            x_best, 
            method='L-BFGS-B', 
            jac=grad_func, 
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-15}
        )
        
        if res.success:
            x_best = res.x
            r_current = compute_radius(x_best)
            if r_current > r_best:
                r_best = r_current
            else:
                # If it got worse, revert?
                # L-BFGS-B should not make it worse if converged, but sometimes it does.
                pass
        else:
            # Try to use the best point found during optimization
            # res.x is the last point.
            r_current = compute_radius(res.x)
            if r_current > r_best:
                x_best = res.x
                r_best = r_current

    # Final polish with very high beta
    if time.time() < deadline - 5:
        beta = 5000
        def obj_func(x_flat):
            val, _ = objective_and_grad(x_flat, beta)
            return val
        def grad_func(x_flat):
            _, grad = objective_and_grad(x_flat, beta)
            return grad
        res = minimize(
            obj_func, 
            x_best, 
            method='L-BFGS-B', 
            jac=grad_func, 
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-16, 'gtol': 1e-16}
        )
        if res.success:
             r_current = compute_radius(res.x)
             if r_current > r_best:
                 x_best = res.x
                 r_best = r_current

    return x_best.reshape(N, 2)
