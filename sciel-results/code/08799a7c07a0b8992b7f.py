import numpy as np
import scipy.optimize as opt
import time

N = 50

def construct():
    start_time = time.time()
    deadline = 110.0  # Leave 10s margin
    
    best_points = None
    best_r = -np.inf
    
    # Helper to compute radius
    def compute_radius(pts):
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        # Exclude self distances
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Wall distances
        # dist to left: x + 0.5
        # dist to right: 0.5 - x
        # dist to bottom: y + 0.5
        # dist to top: 0.5 - y
        d_left = pts[:, 0] + 0.5
        d_right = 0.5 - pts[:, 0]
        d_bottom = pts[:, 1] + 0.5
        d_top = 0.5 - pts[:, 1]
        
        min_wall_dist = np.min(np.min([d_left, d_right, d_bottom, d_top], axis=0))
        
        r = min(min_pair_dist, min_wall_dist)
        return r

    # Objective function: minimize -softmin(distances)
    # softmin(D) = -1/beta * log(sum(exp(-beta * D)))
    # f = -softmin = 1/beta * log(sum(exp(-beta * D)))
    
    def objective_and_grad(pts_flat, beta):
        pts = pts_flat.reshape(N, 2)
        
        # Compute all distances D and their gradients wrt pts
        # We will accumulate the weighted sum of gradients
        
        # 1. Pairwise distances
        # diff[i,j] = pts[i] - pts[j]
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        # Mask diagonal
        mask = ~np.eye(N, dtype=bool)
        dists[~mask] = 0 # Set diagonal to 0, we will ignore them in sum
        
        # We only care about i < j to avoid double counting in sum, 
        # but for gradient accumulation, symmetry helps.
        # Let's just sum over all i != j. 
        # grad dists[i,j] wrt pts[i] = diff[i,j] / dists[i,j]
        # grad dists[i,j] wrt pts[j] = -diff[i,j] / dists[i,j]
        
        # Compute weights w_k = exp(-beta * D_k) / sum(exp(-beta * D_k))
        # To avoid overflow/underflow, we can shift by min dist
        # But D is positive.
        
        # We need to sum exp(-beta * D_k) for all k in the set of valid distances.
        # Let's collect all distances in a flat array and their corresponding gradient contributions.
        
        # This might be memory intensive if done naively for large N, but N=50 is small.
        # Number of pairs ~ 1225.
        
        # Initialize gradient accumulator
        grad_pts = np.zeros_like(pts)
        
        # --- Pairwise ---
        # dists is N x N. We want upper triangle.
        tri_u = np.triu_indices(N, k=1)
        pair_dists = dists[tri_u]
        
        # Weights for pairs
        # exp_vals = np.exp(-beta * pair_dists)
        # sum_exp = np.sum(exp_vals)
        # w = exp_vals / sum_exp
        
        # To be numerically stable:
        min_pair = np.min(pair_dists)
        # Shift by min_pair to make exponents <= 0
        exp_vals = np.exp(-beta * (pair_dists - min_pair))
        sum_exp = np.sum(exp_vals)
        if sum_exp == 0:
             sum_exp = 1e-300
        w_pairs = exp_vals / sum_exp
        
        # Gradient contribution from pairs
        # For each pair (i,j) with weight w:
        # grad_pts[i] += w * (pts[i] - pts[j]) / dist
        # grad_pts[j] += w * (pts[j] - pts[i]) / dist
        # diff[i,j] = pts[i] - pts[j]
        # grad_pair = diff / dist
        # We need to handle dist=0 case, but we start with distinct points.
        
        # Vectorized update
        # tri_u gives indices (i_list, j_list)
        i_idx, j_idx = tri_u
        pair_diff = diff[i_idx, j_idx] # shape (M_pairs, 2)
        pair_dist = pair_dists # shape (M_pairs,)
        
        # Unit vectors
        # Avoid division by zero
        pair_dist_safe = np.where(pair_dist > 1e-10, pair_dist, 1.0)
        unit_vecs = pair_diff / pair_dist_safe[:, np.newaxis]
        
        # Weighted unit vectors
        weighted_unit_vecs = w_pairs[:, np.newaxis] * unit_vecs
        
        # Scatter to grad_pts
        # grad_pts[i] += weighted_unit_vecs[k]
        # grad_pts[j] -= weighted_unit_vecs[k]
        np.add.at(grad_pts, i_idx, weighted_unit_vecs)
        np.add.at(grad_pts, j_idx, -weighted_unit_vecs)
        
        # --- Wall Distances ---
        # d_left = x + 0.5, grad = (1, 0)
        # d_right = 0.5 - x, grad = (-1, 0)
        # d_bottom = y + 0.5, grad = (0, 1)
        # d_top = 0.5 - y, grad = (0, -1)
        
        d_left = pts[:, 0] + 0.5
        d_right = 0.5 - pts[:, 0]
        d_bottom = pts[:, 1] + 0.5
        d_top = 0.5 - pts[:, 1]
        
        wall_dists = np.vstack([d_left, d_right, d_bottom, d_top]).flatten()
        
        # Combine with pair dists for global softmin?
        # The problem statement defines r = min(min_pair, min_wall).
        # So the softmin should be over the UNION of all these distances.
        
        # Let's restart the weight calculation to include both.
        all_dists = np.concatenate([pair_dists, wall_dists])
        
        # Recompute weights over all_dists
        min_all = np.min(all_dists)
        exp_vals_all = np.exp(-beta * (all_dists - min_all))
        sum_exp_all = np.sum(exp_vals_all)
        if sum_exp_all == 0:
            sum_exp_all = 1e-300
        w_all = exp_vals_all / sum_exp_all
        
        w_pairs_new = w_all[:len(pair_dists)]
        w_walls_new = w_all[len(pair_dists):]
        
        # Re-accumulate gradient with new weights
        grad_pts = np.zeros_like(pts)
        
        # Pairs
        weighted_unit_vecs_new = w_pairs_new[:, np.newaxis] * unit_vecs
        np.add.at(grad_pts, i_idx, weighted_unit_vecs_new)
        np.add.at(grad_pts, j_idx, -weighted_unit_vecs_new)
        
        # Walls
        # Indices for walls in w_walls_new:
        # 0..N-1: left (grad 1,0)
        # N..2N-1: right (grad -1,0)
        # 2N..3N-1: bottom (grad 0,1)
        # 3N..4N-1: top (grad 0,-1)
        
        # grad_pts[i] += w_left[i] * (1, 0)
        grad_pts[:, 0] += w_walls_new[0:N]
        grad_pts[:, 0] -= w_walls_new[N:2*N]
        grad_pts[:, 1] += w_walls_new[2*N:3*N]
        grad_pts[:, 1] -= w_walls_new[3*N:4*N]
        
        # Objective value
        # f = 1/beta * log(sum_exp_all) + min_all (due to shift)
        # Wait, softmin(D) = -1/beta log sum exp(-beta D).
        # We used exp(-beta (D - min_all)) = exp(-beta D) exp(beta min_all).
        # sum exp(-beta D) = exp(-beta min_all) sum exp(-beta (D - min_all)).
        # log sum exp(-beta D) = -beta min_all + log sum_exp_all.
        # softmin = -1/beta (-beta min_all + log sum_exp_all) = min_all - 1/beta log sum_exp_all.
        # We want to minimize -softmin = -min_all + 1/beta log sum_exp_all.
        
        obj_val = -min_all + (1.0 / beta) * np.log(sum_exp_all)
        
        # Gradient of f = -softmin
        # f = - (min_all - 1/beta log sum_exp_all) ? No.
        # Let's stick to the derivation:
        # f(c) = 1/beta log sum exp(-beta D_k).
        # With shift: f(c) = min_all - 1/beta log(sum exp(-beta (D_k - min_all)))
        # Wait, the shift is inside the log.
        # Let S_shifted = sum exp(-beta (D_k - min_all)).
        # S_orig = exp(-beta min_all) S_shifted.
        # f = 1/beta log S_orig = 1/beta (-beta min_all + log S_shifted) = -min_all + 1/beta log S_shifted.
        # We want to MINIMIZE f.
        # But we derived grad f = - sum w_k grad D_k.
        # Let's check sign.
        # If D increases, f should decrease?
        # If D increases, exp(-beta D) decreases, log sum decreases, f decreases.
        # So gradient should point in direction of increasing D?
        # No, we are minimizing f. We want to move in direction -grad f.
        # If grad f is negative, we move in positive direction?
        # Let's re-evaluate.
        # f = 1/beta log S.
        # df = 1/beta (1/S) dS = 1/beta (1/S) sum (-beta exp(-beta D_k)) dD_k = - sum (exp(-beta D_k)/S) dD_k = - sum w_k dD_k.
        # dD_k = grad D_k . dc.
        # So grad f = - sum w_k grad D_k.
        # We want to minimize f. So we update c = c - lr * grad f.
        # c = c + lr * sum w_k grad D_k.
        # This moves c in the direction that increases the weighted sum of distances.
        # This is correct. We want to increase the minimum distance.
        
        # So the gradient to return to minimize is grad_f = - sum w_k grad D_k.
        # My accumulation above computed sum w_k grad D_k.
        # So I should return -grad_pts.
        
        return obj_val, -grad_pts.flatten()

    def objective(pts_flat, beta):
        val, _ = objective_and_grad(pts_flat, beta)
        return val

    def grad(pts_flat, beta):
        _, g = objective_and_grad(pts_flat, beta)
        return g

    # Initialization strategies
    def init_hexagonal():
        # Generate hexagonal lattice points
        # Scale factor s such that distance between centers is approx 2*r
        # We want to fit 50 points in [-0.5, 0.5]^2
        # Approx area per point = 1/50 = 0.02
        # Hex area = s^2 * sqrt(3)/2 ? No, distance d = s. Area = d^2 * sqrt(3)/2.
        # s^2 * 0.866 = 0.02 => s^2 = 0.023 => s = 0.15
        s = 0.15
        pts = []
        # Generate a grid of hex points
        # y step = s * sqrt(3)/2
        dy = s * np.sqrt(3) / 2
        # x step = s
        # Rows alternate offset
        row = 0
        while len(pts) < N + 10:
            y = -0.5 + dy/2 + row * dy
            if y > 0.5:
                break
            x_offset = 0 if row % 2 == 0 else s/2
            col = 0
            while True:
                x = -0.5 + s/2 + x_offset + col * s
                if x > 0.5:
                    break
                pts.append([x, y])
                col += 1
            row += 1
        pts = np.array(pts[:N])
        # Add small random jitter to break symmetry
        pts += np.random.uniform(-0.01, 0.01, pts.shape)
        # Clip to bounds
        pts[:, 0] = np.clip(pts[:, 0], -0.5, 0.5)
        pts[:, 1] = np.clip(pts[:, 1], -0.5, 0.5)
        return pts

    def init_random():
        pts = np.random.uniform(-0.5, 0.5, (N, 2))
        return pts

    bounds = [(-0.5, 0.5)] * (2 * N)
    
    num_restarts = 5
    # We can adjust num_restarts based on time, but let's start with a fixed number and check time.
    
    for r_idx in range(num_restarts):
        if time.time() > deadline:
            break
            
        if r_idx == 0:
            pts0 = init_hexagonal()
        else:
            pts0 = init_random()
            
        # Start with moderate beta
        beta = 20.0
        
        res = opt.minimize(
            objective, 
            pts0.flatten(), 
            method='L-BFGS-B', 
            jac=lambda x: grad(x, beta),
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        pts_opt = res.x.reshape(N, 2)
        r_opt = compute_radius(pts_opt)
        
        if r_opt > best_r:
            best_r = r_opt
            best_points = pts_opt.copy()
            
        # Polish with higher beta
        if time.time() < deadline - 10:
            beta_high = 100.0
            res2 = opt.minimize(
                objective, 
                pts_opt.flatten(), 
                method='L-BFGS-B', 
                jac=lambda x: grad(x, beta_high),
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            pts_opt2 = res2.x.reshape(N, 2)
            r_opt2 = compute_radius(pts_opt2)
            
            if r_opt2 > best_r:
                best_r = r_opt2
                best_points = pts_opt2.copy()

    # Final check
    if best_points is None:
        # Fallback to grid if something went wrong
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_points = grid[:N]
        
    return best_points
