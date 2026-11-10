# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

N = 200

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin

    # 1. Initialization: Hexagonal lattice
    # Calculate ideal spacing for N points in unit square
    # Area per point = 1/N
    # Hex area per point = d^2 * sqrt(3) / 2
    # d = sqrt(2 / (N * sqrt(3)))
    d = np.sqrt(2.0 / (N * np.sqrt(3)))
    
    # Generate points in a hexagonal grid
    # We need to cover [-0.5, 0.5] x [-0.5, 0.5]
    max_idx = int(np.ceil(0.5 / d)) + 2
    xs = []
    ys = []
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            x = i * d
            y = j * d * np.sqrt(3) / 2 + (i % 2) * d * np.sqrt(3) / 4
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                xs.append(x)
                ys.append(y)
    
    points = np.array([xs, ys]).T
    
    if len(points) > N:
        # Randomly select N points to break symmetry and avoid bias
        # Or better: select a contiguous block from the center?
        # Random selection is often better for escaping local minima in the init phase
        # But for a lattice, a contiguous block preserves local structure.
        # Let's try to center the block.
        # The points are generated in a square patch.
        # Let's just take a random subset to ensure we don't always have the same "cut".
        # Actually, for L-BFGS-B, a symmetric start is fine.
        # Let's use a deterministic "center-out" selection to keep it reproducible but varied.
        # Distance from center
        dists = np.sum(points**2, axis=1)
        # Sort by distance and take first N? No, that makes a circle.
        # Let's just take the first N as before, it's a valid lattice patch.
        points = points[:N]
    elif len(points) < N:
        # Fallback: Square grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        points = grid[:N]

    # 2. Optimization: L-BFGS-B with Log-Sum-Exp Objective
    # Objective: Minimize T * log(sum(exp(-dist/T)))
    # This approximates min(dist) as T -> 0.
    # We use a small T to focus on the minimum distance.
    T = 0.05
    
    def objective(flat_points):
        pts = flat_points.reshape(N, 2)
        
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=-1)
        np.fill_diagonal(dist_sq, np.inf)
        dist = np.sqrt(dist_sq)
        
        # Boundary distances
        d_b = np.min([
            0.5 - pts[:, 0],
            0.5 + pts[:, 0],
            0.5 - pts[:, 1],
            0.5 + pts[:, 1]
        ], axis=0)
        
        # Combine all distances into one array
        # We want to minimize the soft-min of all these distances.
        # dist_flat contains all pairwise distances (upper triangle to avoid double counting? 
        # Log-sum-exp is symmetric, so double counting just scales the log, 
        # but for gradient, it's fine. However, to be precise, let's take unique pairs.)
        # For N=200, 19900 pairs. 
        # dist is 200x200. Upper triangle has 19900 elements.
        iu = np.triu_indices(N, 1)
        pair_dists = dist[iu]
        
        all_dists = np.concatenate([pair_dists, d_b])
        
        # Log-Sum-Exp trick for stability:
        # log(sum(exp(-d/T))) = -min_d/T + log(sum(exp(-(d-min_d)/T)))
        # We want to minimize this.
        # Let f(d) = log(sum(exp(-d/T)))
        # grad f = - (1/T) * (sum(exp(-d/T) * exp(-d/T) / sum(exp(-d/T))) ) ... 
        # Actually, let's just use the numerical gradient or let L-BFGS-B approximate it?
        # L-BFGS-B uses numerical gradients if not provided. 
        # Providing analytical gradient is faster and more robust.
        
        # Gradient of log(sum(exp(-d/T))) w.r.t d_k is:
        # -1/T * exp(-d_k/T) / sum(exp(-d_j/T))
        # This is a softmax-like weight.
        
        # To avoid underflow/overflow in exp(-d/T), we shift by min_d.
        min_d = np.min(all_dists)
        # If min_d is very large, exp(-d/T) might be small.
        # exp(-(d - min_d)/T) is safe.
        shifted_dists = all_dists - min_d
        exp_terms = np.exp(-shifted_dists / T)
        sum_exp = np.sum(exp_terms)
        
        # Value: T * log(sum_exp) + min_d  (Wait, log(sum(exp(-d/T))) = log(sum(exp(-(d-min_d)/T) * exp(min_d/T))) = log(sum(exp(-(d-min_d)/T))) + min_d/T)
        # So T * log(sum(exp(-d/T))) = T * log(sum_exp) + min_d
        # We want to minimize this.
        val = T * np.log(sum_exp) + min_d
        
        # Gradient
        # d(val)/d(d_k) = d(val)/d(min_d) * d(min_d)/d(d_k) + T * (1/sum_exp) * d(sum_exp)/d(d_k)
        # d(min_d)/d(d_k) is 1 if k is argmin, 0 otherwise (subgradient).
        # This is non-differentiable at the min.
        # The smooth approximation gradient is:
        # d(val)/d(d_k) = - (1/sum_exp) * exp(-(d_k - min_d)/T)  (from the log-sum-exp part)
        # Wait, let's re-derive.
        # L = T * log(sum_j exp(-d_j/T))
        # dL/dd_k = T * (1/sum_j exp(-d_j/T)) * d(sum_j exp(-d_j/T))/dd_k
        # d(sum_j exp(-d_j/T))/dd_k = exp(-d_k/T) * (-1/T)
        # dL/dd_k = T * (1/sum_exp) * (-1/T) * exp(-d_k/T)
        # dL/dd_k = - exp(-d_k/T) / sum_exp
        # This is the negative of the softmax probability.
        
        # We need to map this gradient back to the points.
        # This is complex because d_k depends on multiple points.
        # Given the complexity, let's let L-BFGS-B use numerical gradients? 
        # For N=200 (400 vars), numerical gradient is 401 evaluations per step.
        # L-BFGS-B might be slow.
        
        # Alternative: Use the `1/d^p` objective with large p?
        # Or simply use the `max(0, r_target - d)` approach with binary search on r_target?
        
        # Let's stick to the `1/d^2` but with a twist: 
        # The previous score was 0.19. The best known is ~1.0.
        # The issue with `1/d^2` is that it's a "soft" constraint.
        
        # Let's try a different structural change: **Simulated Annealing** or **Local Search**?
        # No, L-BFGS-B is usually better for smooth landscapes.
        
        # Let's try to provide the analytical gradient for the `1/d^2` objective, which is easy.
        # E = sum(1/d_ij^2) + sum(1/d_bi^2)
        # dE/dd_ij = -2/d_ij^3
        # dd_ij/dx_i = (x_i - x_j)/d_ij
        # dE/dx_i = sum_j (-2/d_ij^3) * (x_i - x_j)/d_ij = -2 * sum_j (x_i - x_j) / d_ij^4
        
        # This is much faster than numerical gradient.
        
        return val # Placeholder, will implement 1/d^4 with analytic grad below

    # Let's implement 1/d^4 with analytic gradient for speed and focus on minimum.
    # Higher power focuses more on the minimum distance.
    
    def objective_and_grad(flat_points):
        pts = flat_points.reshape(N, 2)
        
        # Pairwise
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=-1)
        np.fill_diagonal(dist_sq, np.inf)
        
        # Clip to avoid div by zero
        min_dist_sq = 1e-12
        dist_sq_clipped = np.maximum(dist_sq, min_dist_sq)
        
        # Energy: sum(1/d^4) = sum(1/dist_sq^2)
        # We sum over all i!=j.
        energy = np.sum(1.0 / (dist_sq_clipped**2))
        
        # Gradient of energy w.r.t pts[i]
        # E = sum_{j!=i} (1/d_ij^4) + ...
        # dE/dx_i = sum_{j!=i} -4 * d_ij^-6 * d(d_ij)/dx_i
        # d(d_ij)/dx_i = (x_i - x_j)/d_ij
        # dE/dx_i = sum_{j!=i} -4 * d_ij^-7 * (x_i - x_j)
        
        # Vectorized:
        # We need sum_j (x_i - x_j) / d_ij^7
        # diff is (x_i - x_j)
        # inv_d7 = 1 / dist_sq_clipped^(3.5)
        # grad_pair[i] = -4 * sum_j (diff[i,j] * inv_d7[i,j])
        
        inv_d7 = 1.0 / (dist_sq_clipped ** 3.5)
        # Mask diagonal
        inv_d7[np.diag_indices_from(inv_d7)] = 0.0
        
        grad_pts = -4.0 * np.sum(diff * inv_d7[:, :, np.newaxis], axis=1)
        
        # Boundary
        d_b = np.min([
            0.5 - pts[:, 0],
            0.5 + pts[:, 0],
            0.5 - pts[:, 1],
            0.5 + pts[:, 1]
        ], axis=0)
        
        d_b_clipped = np.maximum(d_b, min_dist_sq)
        energy += np.sum(1.0 / (d_b_clipped**2))
        
        # Gradient for boundary
        # E_b = sum 1/d_bi^2
        # dE_b/dx_i = -2/d_bi^3 * d(d_bi)/dx_i
        # d_bi is min of 4 terms. The gradient is non-zero only for the active term.
        # We can approximate by summing gradients of all 4 terms? No, that's wrong.
        # But if we use a smooth min approximation for boundary too?
        # Or just use the active term.
        
        # For simplicity and robustness, let's use a smooth boundary penalty:
        # E_b = sum (1/(0.5-x)^2 + 1/(0.5+x)^2 + ...) ? No, that penalizes being near ANY wall, not just the closest one.
        # The "min" is the correct constraint.
        
        # Let's calculate gradient for the active boundary term.
        # active_idx: 0 for 0.5-x, 1 for 0.5+x, 2 for 0.5-y, 3 for 0.5+y
        # We can find the argmin for each point.
        
        # This is getting complex. Let's just use numerical gradient for boundary?
        # Or assume the points are not exactly on the boundary?
        
        # Actually, L-BFGS-B handles bounds. We can just add a penalty for violating bounds?
        # The bounds are (-0.5, 0.5). L-BFGS-B keeps them inside.
        # So we don't need boundary energy in the objective if we use bounds?
        # NO, we do, because we want to maximize the radius, which is limited by the boundary.
        # If we don't penalize boundary, points will go to the boundary, and r is limited by distance to boundary.
        # So we MUST include boundary distance in the objective.
        
        # Let's use a smooth approximation for the boundary distance as well?
        # Or just use the active term gradient.
        
        # Let's compute the active term index.
        # d_b is (N,). We need to know which of the 4 terms was the min.
        # We can compute the 4 arrays and argmin.
        
        db_x_neg = 0.5 - pts[:, 0]
        db_x_pos = 0.5 + pts[:, 0]
        db_y_neg = 0.5 - pts[:, 1]
        db_y_pos = 0.5 + pts[:, 1]
        
        db_all = np.stack([db_x_neg, db_x_pos, db_y_neg, db_y_pos], axis=1)
        active_idx = np.argmin(db_all, axis=1)
        
        # Gradient contribution from boundary
        # dE/dx_i for active term k:
        # if k=0 (0.5-x): d(0.5-x)/dx = -1. dE/dx = -2/d^3 * (-1) = 2/d^3
        # if k=1 (0.5+x): d(0.5+x)/dx = 1. dE/dx = -2/d^3 * (1) = -2/d^3
        # if k=2 (0.5-y): d(0.5-y)/dy = -1. dE/dy = 2/d^3
        # if k=3 (0.5+y): d(0.5+y)/dy = 1. dE/dy = -2/d^3
        
        db_active = db_all[np.arange(N), active_idx]
        db_active_clipped = np.maximum(db_active, min_dist_sq)
        grad_b = -2.0 / (db_active_clipped ** 3)
        
        # Apply to correct component
        # active_idx: 0->x, 1->x, 2->y, 3->y
        # sign: 0->-1 (but dE/dx = 2/d^3, so sign is + for x? No.
        # Let's re-verify.
        # E = 1/d^2. dE/dd = -2/d^3.
        # d = 0.5 - x. dd/dx = -1. dE/dx = (-2/d^3)*(-1) = 2/d^3.
        # So if active is 0, grad_x += 2/d^3.
        # If active is 1, d = 0.5 + x. dd/dx = 1. dE/dx = -2/d^3.
        # So grad_x -= 2/d^3.
        
        # We have grad_b = -2/d^3.
        # If active is 0, we want +2/d^3. So we multiply by -1?
        # Let's define a sign array.
        # Signs for d/d(coord):
        # 0: -1 (x) -> dE/dx = -grad_b * -1 = grad_b? No.
        # Let's just construct it directly.
        
        grad_b_vec = np.zeros((N, 2))
        # For each point i:
        # if active_idx[i] == 0: grad_b_vec[i, 0] = 2 / d^3
        # if active_idx[i] == 1: grad_b_vec[i, 0] = -2 / d^3
        # if active_idx[i] == 2: grad_b_vec[i, 1] = 2 / d^3
        # if active_idx[i] == 3: grad_b_vec[i, 1] = -2 / d^3
        
        # We have val = -2/d^3.
        # So:
        # case 0: grad_x = -val
        # case 1: grad_x = val
        # case 2: grad_y = -val
        # case 3: grad_y = val
        
        # Let's use masks.
        mask_x_neg = (active_idx == 0)
        mask_x_pos = (active_idx == 1)
        mask_y_neg = (active_idx == 2)
        mask_y_pos = (active_idx == 3)
        
        grad_b_vec[mask_x_neg, 0] = -grad_b[mask_x_neg]
        grad_b_vec[mask_x_pos, 0] = grad_b[mask_x_pos]
        grad_b_vec[mask_y_neg, 1] = -grad_b[mask_y_neg]
        grad_b_vec[mask_y_pos, 1] = grad_b[mask_y_pos]
        
        grad_pts += grad_b_vec
        
        return energy, grad_pts.flatten()

    def objective_only(flat_points):
        e, _ = objective_and_grad(flat_points)
        return e

    x0 = points.flatten()
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    best_r = -1.0
    best_points = points.copy()
    
    def calc_radius(pts):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_mat = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dist_mat, np.inf)
        min_pair_dist = np.min(dist_mat)
        
        d_b = np.min([
            0.5 - pts[:, 0],
            0.5 + pts[:, 0],
            0.5 - pts[:, 1],
            0.5 + pts[:, 1]
        ], axis=0)
        min_boundary_dist = np.min(d_b)
        
        return min(min_pair_dist / 2.0, min_boundary_dist)

    current_r = calc_radius(points)
    if current_r > best_r:
        best_r = current_r
        best_points = points.copy()

    try:
        def callback(flat_x):
            nonlocal best_r, best_points
            if time.time() > deadline:
                return True
            pts = flat_x.reshape(N, 2)
            pts_clipped = np.clip(pts, -0.5, 0.5)
            r = calc_radius(pts_clipped)
            if r > best_r:
                best_r = r
                best_points = pts_clipped.copy()
            return False

        result = minimize(objective_only, x0, jac=objective_and_grad, method='L-BFGS-B', bounds=bounds, 
                          options={'maxiter': 5000, 'callback': callback, 'maxls': 50})
        
        if result.success:
            final_points = result.x.reshape(N, 2)
            final_points = np.clip(final_points, -0.5, 0.5)
            final_r = calc_radius(final_points)
            if final_r > best_r:
                best_r = final_r
                best_points = final_points.copy()
                
    except Exception as e:
        pass

    return best_points

# EVOLVE-BLOCK-END
