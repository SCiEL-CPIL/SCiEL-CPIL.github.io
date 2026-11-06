import numpy as np
import time
from scipy.optimize import minimize

N = 200

def construct():
    start_time = time.time()
    deadline = 115.0 # 5s margin

    # 1. Initialization: Hexagonal Lattice
    # Estimate radius for 200 circles in unit square
    # Area per circle approx pi*r^2. Total area 200*pi*r^2 <= 1. r <= 1/sqrt(200*pi) ~ 0.04
    # Hex packing density ~ 0.9.
    # Let's just generate a grid and trim.
    
    def generate_hex_lattice(n_points):
        # Try a few densities to get exactly n_points
        best_pts = None
        best_err = np.inf
        for scale in np.linspace(0.03, 0.05, 50):
            # Hex grid
            dx = 2 * scale
            dy = scale * np.sqrt(3)
            
            # Generate points in a large enough area
            x_range = np.arange(-0.5 - 2*scale, 0.5 + 2*scale, dx)
            y_range = np.arange(-0.5 - 2*scale, 0.5 + 2*scale, dy)
            
            pts = []
            for i, x in enumerate(x_range):
                y_offset = 0 if i % 2 == 0 else dy / 2
                for y in y_range:
                    pts.append([x, y + y_offset])
            pts = np.array(pts)
            
            # Filter points inside [-0.5, 0.5]
            mask = (pts[:, 0] >= -0.5) & (pts[:, 0] <= 0.5) & (pts[:, 1] >= -0.5) & (pts[:, 1] <= 0.5)
            pts = pts[mask]
            
            if len(pts) >= n_points:
                # Randomly select n_points? Or take from center out?
                # Taking from center out is better for packing
                centers = np.array([0, 0])
                dists = np.linalg.norm(pts - centers, axis=1)
                indices = np.argsort(dists)
                selected = pts[indices[:n_points]]
                
                # Calculate current min distance
                from scipy.spatial.distance import pdist
                min_d = pdist(selected)
                min_d = np.min(min_d)
                # Also check boundaries
                min_b = np.min(np.minimum(0.5 - np.abs(selected), 0.5 - np.abs(selected[:, 1]))) # Wait, min of x and y distances
                # Actually min dist to boundary is min(0.5-|x|, 0.5-|y|)
                min_b = np.min(np.minimum(0.5 - np.abs(selected[:, 0]), 0.5 - np.abs(selected[:, 1])))
                
                current_score = min(min_d, min_b)
                
                # We want to maximize score. 
                # If we have more points, we can just take the first N.
                # Let's just use the first N closest to center.
                if current_score > best_err: # We are maximizing, so this logic is inverted if best_err is min. 
                    # Let's track best_score
                    pass
                    
        # Simpler: Just generate a dense grid, take N closest to center.
        # Then optimize.
        
        # Let's just generate a standard hex grid with estimated r.
        # r ~ 0.036
        r_init = 0.035
        dx = 2 * r_init
        dy = r_init * np.sqrt(3)
        
        x_range = np.arange(-0.5, 0.5 + dx, dx)
        y_range = np.arange(-0.5, 0.5 + dy, dy)
        
        pts = []
        for i, x in enumerate(x_range):
            y_offset = 0 if i % 2 == 0 else dy / 2
            for y in y_range:
                if -0.5 <= x <= 0.5 and -0.5 <= y + y_offset <= 0.5:
                    pts.append([x, y + y_offset])
        pts = np.array(pts)
        
        # If we have more than N, pick N closest to center
        if len(pts) > N:
            dists_to_center = np.linalg.norm(pts, axis=1)
            indices = np.argsort(dists_to_center)
            pts = pts[indices[:N]]
        elif len(pts) < N:
            # Pad with random points? Or adjust grid.
            # For now, assume N is met or close. If not, add random.
            while len(pts) < N:
                p = np.random.uniform(-0.4, 0.4, 2)
                pts = np.vstack([pts, [p]])
                
        return pts

    C = generate_hex_lattice(N)
    best_C = C.copy()
    best_score = -np.inf
    
    # 2. Optimization: Soft-Min Annealing with L-BFGS-B
    
    # Precompute indices for pairs? 
    # N=200, pairs=19900. 
    # Vectorized distance calculation is fast enough.
    
    def compute_soft_min_obj_grad(C_flat, beta):
        C = C_flat.reshape(N, 2)
        
        # Pairwise distances
        # Use broadcasting: (N, 1, 2) - (1, N, 2) -> (N, N, 2)
        # This creates a 200x200x2 array. 80k floats. Fine.
        diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        # Mask diagonal
        dists_sq[np.diag_indices(N)] = np.inf
        dists = np.sqrt(dists_sq)
        
        # Boundary distances
        # d_x = 0.5 - |x|, d_y = 0.5 - |y|
        # d_b = min(d_x, d_y)
        # Soft min of d_x and d_y?
        # Let's just use hard min for boundary in the "pool" of distances?
        # No, let's use soft min for boundaries too to keep gradient smooth.
        # d_b_soft = -1/beta * log( exp(-beta*(0.5-|x|)) + exp(-beta*(0.5-|y|)) )
        # Wait, the distance to the square boundary is min(0.5-|x|, 0.5-|y|).
        # Let's just add the 4 wall distances to the pool?
        # d_left = 0.5 + x, d_right = 0.5 - x, d_down = 0.5 + y, d_up = 0.5 - y.
        # Pool: all pairwise dists, all 4 wall dists for each point.
        
        # Wall distances
        d_left = 0.5 + C[:, 0]
        d_right = 0.5 - C[:, 0]
        d_down = 0.5 + C[:, 1]
        d_up = 0.5 - C[:, 1]
        
        # Flatten all distances
        pair_dists = dists[np.triu_indices(N, 1)]
        wall_dists = np.concatenate([d_left, d_right, d_down, d_up])
        
        all_dists = np.concatenate([pair_dists, wall_dists])
        
        # Soft min
        # S = -1/beta * logsumexp(-beta * all_dists)
        # To avoid underflow/overflow, use max subtraction
        max_d = np.max(all_dists)
        # If max_d is very large, -beta*all_dists might be very negative.
        # logsumexp(-beta * (all_dists - max_d)) + (-beta * max_d)
        
        # Actually, we want to maximize min dist.
        # Let's minimize -soft_min.
        # f = -S = 1/beta * logsumexp(-beta * all_dists)
        
        # Gradient of f w.r.t C
        # f = (1/beta) * log( sum exp(-beta d_k) )
        # df/dC = (1/beta) * ( sum exp(-beta d_k) * (-beta) * dd_k/dC ) / ( sum exp(-beta d_k) )
        # df/dC = - sum ( w_k * dd_k/dC ) where w_k = exp(-beta d_k) / sum(exp(-beta d_k))
        
        # We need weights w_k.
        # Compute log_weights = -beta * all_dists
        # Subtract max for stability
        log_w = -beta * all_dists
        max_log_w = np.max(log_w)
        log_w_stable = log_w - max_log_w
        w_stable = np.exp(log_w_stable)
        w = w_stable / np.sum(w_stable)
        
        # Now compute gradient
        # grad = - ( sum_{pairs} w_pair * grad_d_pair + sum_{walls} w_wall * grad_d_wall )
        
        grad = np.zeros_like(C)
        
        # Pairwise part
        # We have w for the upper triangle.
        # w_pair corresponds to dists[np.triu_indices(N, 1)]
        # We need to map these back to i, j.
        i_idx, j_idx = np.triu_indices(N, 1)
        
        # grad_d_ij = (c_i - c_j) / d_ij
        # diff_ij = C[i] - C[j]
        # dist_ij = dists[i, j]
        
        # Accumulate gradients
        # For each pair (i, j):
        # grad_i += w * (c_i - c_j)/d
        # grad_j -= w * (c_i - c_j)/d
        
        # Vectorized:
        # diff = C[i_idx] - C[j_idx]
        # dists_pair = dists[i_idx, j_idx]
        # grad_vec = w_pair[:, np.newaxis] * diff / dists_pair[:, np.newaxis]
        
        diff_pairs = C[i_idx] - C[j_idx]
        dists_pairs = dists[i_idx, j_idx]
        
        # Avoid division by zero (shouldn't happen with init, but just in case)
        mask = dists_pairs > 1e-12
        grad_vec = np.zeros_like(diff_pairs)
        grad_vec[mask] = (w[i_idx + j_idx * 0] * 0) # Placeholder, need to map w correctly
        
        # w is flattened. The order of pair_dists is from triu_indices.
        # So w[:len(pair_dists)] corresponds to pairs.
        w_pairs = w[:len(pair_dists)]
        
        # grad_vec = w_pairs[:, None] * diff_pairs / dists_pairs[:, None]
        grad_vec = (w_pairs[:, None] / dists_pairs[:, None]) * diff_pairs
        
        # Accumulate to grad
        # np.add.at(grad, i_idx, grad_vec)
        # np.add.at(grad, j_idx, -grad_vec)
        np.add.at(grad, i_idx, grad_vec)
        np.add.at(grad, j_idx, -grad_vec)
        
        # Wall part
        # w_walls = w[len(pair_dists):]
        w_walls = w[len(pair_dists):]
        
        # d_left: 0.5 + x. d/dx = 1.
        # d_right: 0.5 - x. d/dx = -1.
        # d_down: 0.5 + y. d/dy = 1.
        # d_up: 0.5 - y. d/dy = -1.
        
        # w_left = w_walls[0:N]
        # w_right = w_walls[N:2N]
        # w_down = w_walls[2N:3N]
        # w_up = w_walls[3N:4N]
        
        w_left = w_walls[0:N]
        w_right = w_walls[N:2*N]
        w_down = w_walls[2*N:3*N]
        w_up = w_walls[3*N:4*N]
        
        # grad_x from walls = - (w_left * 1 + w_right * (-1)) = -w_left + w_right
        # grad_y from walls = - (w_down * 1 + w_up * (-1)) = -w_down + w_up
        
        grad[:, 0] += -w_left + w_right
        grad[:, 1] += -w_down + w_up
        
        # Objective value
        # f = 1/beta * (log_sum_exp_stable + max_log_w)
        log_sum_exp = np.log(np.sum(w_stable)) + max_log_w
        obj_val = (1.0 / beta) * log_sum_exp
        
        return obj_val, grad.flatten()

    # Annealing schedule
    # Start with low beta (smooth), end with high beta (sharp)
    betas = [10, 50, 100, 200, 500, 1000, 2000]
    
    for beta in betas:
        if time.time() > deadline:
            break
            
        # Run L-BFGS-B
        # Initial guess is current best
        x0 = best_C.flatten()
        
        try:
            res = minimize(
                compute_soft_min_obj_grad,
                x0,
                args=(beta,),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if res.success or res.fun < (1.0/beta)*np.log(1): # Just check if it improved
                C_new = res.x.reshape(N, 2)
                
                # Evaluate true score (hard min)
                # Calculate min pair dist
                from scipy.spatial.distance import pdist
                min_pair = np.min(pdist(C_new))
                min_bound = np.min(np.minimum(0.5 - np.abs(C_new[:, 0]), 0.5 - np.abs(C_new[:, 1])))
                score = min(min_pair, min_bound)
                
                if score > best_score:
                    best_score = score
                    best_C = C_new
                    
        except Exception as e:
            pass # Keep best_C if optimization fails

    return best_C
