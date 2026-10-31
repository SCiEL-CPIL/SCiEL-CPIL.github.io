import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 21

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit
    
    best_sum_r = 0.0
    best_result = None
    
    # We search over aspect ratios.
    # Perimeter 4 => 2(w+h)=4 => h = 2-w.
    # Due to symmetry, we can assume w <= h => w <= 1.
    # We will sample a range of w values.
    
    def solve_for_w(w_val, init_xs, init_ys, init_rs, max_iter=2000):
        h_val = 2.0 - w_val
        
        # v: [x1, y1, r1, ..., xN, y1, yN, rN]
        # Actually, let's keep v = [x1, y1, r1, ..., xN, yN, rN]
        
        def objective(v):
            xs = v[0:3*N:3]
            ys = v[1:3*N:3]
            rs = v[2:3*N:3]
            
            # Ensure radii are positive (bounds handle this, but penalty helps)
            # Wall constraints
            # x_i >= r_i, x_i <= w - r_i
            # y_i >= r_i, y_i <= h - r_i
            
            # Calculate violations
            # Left/Right walls
            viol_lr = np.maximum(0, rs - xs) + np.maximum(0, xs + rs - w_val)
            # Top/Bottom walls
            viol_tb = np.maximum(0, rs - ys) + np.maximum(0, ys + rs - h_val)
            
            wall_penalty = np.sum(viol_lr**2) + np.sum(viol_tb**2)
            
            # Circle-circle constraints
            # (xi-xj)^2 + (yi-yj)^2 >= (ri+rj)^2
            # Violation: (ri+rj)^2 - dist^2 > 0
            
            # Vectorized distance calculation
            # dist_sq[i,j] = (xi-xj)^2 + (yi-yj)^2
            dx = xs[:, np.newaxis] - xs[np.newaxis, :]
            dy = ys[:, np.newaxis] - ys[np.newaxis, :]
            dist_sq = dx**2 + dy**2
            
            min_dist = rs[:, np.newaxis] + rs[np.newaxis, :]
            min_dist_sq = min_dist**2
            
            # Only consider i < j to avoid double counting and self-interaction
            # Create a mask for i < j
            mask = np.triu(np.ones((N, N), dtype=bool), k=1)
            
            # Overlap violation
            # diff = min_dist_sq - dist_sq
            # If diff > 0, there is overlap.
            # We want to penalize the overlap.
            # A common penalty is (max(0, diff))^2
            diff = min_dist_sq - dist_sq
            overlap_viol = np.where(mask, np.maximum(0, diff), 0)
            
            circle_penalty = np.sum(overlap_viol**2)
            
            # Objective: Maximize sum(rs) => Minimize -sum(rs)
            # Add penalties
            # We need to tune the penalty weight. 
            # Since we want valid solutions, we should make penalty large.
            # However, if penalty is too large, it might dominate the radius growth.
            # Let's use a weighted penalty.
            weight = 100.0
            
            return -np.sum(rs) + weight * (wall_penalty + circle_penalty)
        
        def gradient(v):
            # Numerical gradient is slow, but L-BFGS-B can use finite differences.
            # However, providing an analytic gradient is much faster.
            # Let's derive it.
            
            xs = v[0:3*N:3]
            ys = v[1:3*N:3]
            rs = v[2:3*N:3]
            
            grad = np.zeros_like(v)
            
            # Wall gradients
            # L_left = max(0, r - x)^2
            # dL/dx = 2(r-x) if r > x else 0
            # dL/dr = 2(r-x) if r > x else 0
            
            # L_right = max(0, x + r - w)^2
            # dL/dx = 2(x+r-w) if x+r > w else 0
            # dL/dr = 2(x+r-w) if x+r > w else 0
            
            # L_bottom = max(0, r - y)^2
            # L_top = max(0, y + r - h)^2
            
            # Circle gradients
            # L_circ = sum_{i<j} max(0, (ri+rj)^2 - dist^2)^2
            # Let D_ij = (ri+rj)^2 - ((xi-xj)^2 + (yi-yj)^2)
            # If D_ij > 0, penalty term is D_ij^2.
            # dP/dxi = 2 D_ij * dD_ij/dxi = 2 D_ij * (-2(xi-xj)) = -4 D_ij (xi-xj)
            # dP/dri = 2 D_ij * dD_ij/dri = 2 D_ij * 2(ri+rj) = 4 D_ij (ri+rj)
            
            # We can compute this efficiently.
            dx = xs[:, np.newaxis] - xs[np.newaxis, :]
            dy = ys[:, np.newaxis] - ys[np.newaxis, :]
            dist_sq = dx**2 + dy**2
            min_dist = rs[:, np.newaxis] + rs[np.newaxis, :]
            min_dist_sq = min_dist**2
            
            diff = min_dist_sq - dist_sq
            # Only i < j
            mask = np.triu(np.ones((N, N), dtype=bool), k=1)
            
            # Gradient contribution from circle-circle
            # For each i, sum over j > i and j < i
            # Let's compute the full matrix contribution and then sum.
            # dP/dxi = sum_j (i!=j) -4 * max(0, diff_ij) * (xi - xj)
            # Note: diff_ij = diff_ji.
            # max(0, diff) is symmetric.
            # So grad_xi = -4 * sum_j max(0, diff_ij) * (xi - xj)
            
            viol_mat = np.where(diff > 0, diff, 0)
            
            # Sum over j for each i
            # grad_x = -4 * (viol_mat @ (xi - xj_matrix?))
            # (xi - xj) is dx.
            # grad_xi = -4 * sum_j viol_mat[i,j] * dx[i,j]
            
            grad_x_circ = -4.0 * (viol_mat * dx).sum(axis=1)
            grad_y_circ = -4.0 * (viol_mat * dy).sum(axis=1)
            
            # Gradient for radii
            # dP/dri = sum_j (i!=j) 4 * max(0, diff_ij) * (ri + rj)
            # grad_ri = 4 * sum_j viol_mat[i,j] * (ri + rj)
            # viol_mat * (ri + rj)
            # min_dist[i,j] = ri + rj
            grad_r_circ = 4.0 * (viol_mat * min_dist).sum(axis=1)
            
            # Wall gradients
            # Left
            viol_l = np.where(rs > xs, rs - xs, 0)
            grad_x_wall = -2.0 * viol_l  # d/dr(x) of (r-x)^2 is -2(r-x) wrt x? No.
            # L = (r-x)^2. dL/dx = -2(r-x). If r>x, r-x is positive. So grad is negative.
            # Wait, if x < r, we want to push x right (increase x).
            # Gradient of objective wrt x should be positive to increase x?
            # Minimize L. If x < r, L = (r-x)^2. dL/dx = -2(r-x).
            # Since r-x > 0, dL/dx is negative.
            # Gradient descent: x_new = x - lr * grad.
            # If grad is negative, x increases. Correct.
            
            # Right
            viol_r = np.where(xs + rs > w_val, xs + rs - w_val, 0)
            grad_x_wall += 2.0 * viol_r # dL/dx = 2(x+r-w). Positive. x decreases. Correct.
            
            # Bottom
            viol_b = np.where(rs > ys, rs - ys, 0)
            grad_y_wall = -2.0 * viol_b
            
            # Top
            viol_t = np.where(ys + rs > h_val, ys + rs - h_val, 0)
            grad_y_wall += 2.0 * viol_t
            
            # Radius wall gradients
            # Left: L = (r-x)^2. dL/dr = 2(r-x).
            # Right: L = (x+r-w)^2. dL/dr = 2(x+r-w).
            grad_r_wall = 2.0 * (viol_l + viol_r + viol_b + viol_t)
            
            # Combine
            weight = 100.0
            
            # Objective part: -sum(r). Gradient wrt r is -1.
            grad[0:3*N:3] += weight * (grad_x_circ + grad_x_wall)
            grad[1:3*N:3] += weight * (grad_y_circ + grad_y_wall)
            grad[2:3*N:3] += weight * (grad_r_circ + grad_r_wall) - 1.0
            
            return grad
        
        # Bounds
        bounds = []
        for i in range(N):
            bounds.append((0.0, w_val)) # x
            bounds.append((0.0, h_val)) # y
            bounds.append((0.0, min(w_val, h_val)/2.0)) # r
        
        # Initial guess
        v0 = np.concatenate([init_xs, init_ys, init_rs])
        
        # Run optimization
        try:
            res = minimize(objective, v0, jac=gradient, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': max_iter, 'ftol': 1e-8, 'gtol': 1e-5, 'maxls': 50})
            
            # Check validity of result
            xs_res = res.x[0:3*N:3]
            ys_res = res.x[1:3*N:3]
            rs_res = res.x[2:3*N:3]
            
            # Verify no overlaps
            valid = True
            for i in range(N):
                if xs_res[i] < rs_res[i] - 1e-5 or xs_res[i] > w_val - rs_res[i] + 1e-5:
                    valid = False
                    break
                if ys_res[i] < rs_res[i] - 1e-5 or ys_res[i] > h_val - rs_res[i] + 1e-5:
                    valid = False
                    break
            
            if valid:
                for i in range(N):
                    for j in range(i+1, N):
                        dist_sq = (xs_res[i]-xs_res[j])**2 + (ys_res[i]-ys_res[j])**2
                        min_dist = rs_res[i] + rs_res[j]
                        if dist_sq < min_dist**2 - 1e-5:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid and np.sum(rs_res) > 0:
                return np.sum(rs_res), res.x
            else:
                # If invalid, we might still use it if the penalty was low, but strictly we want valid.
                # For now, return None if invalid to be safe.
                return 0.0, None
        except Exception as e:
            return 0.0, None

    # Sample aspect ratios
    # w in [0.5, 1.0]
    w_values = np.linspace(0.5, 1.0, 20)
    
    # For each w, try a few initializations
    # 1. Grid
    # 2. Hexagonal
    
    for w in w_values:
        if time.time() > deadline:
            break
            
        h = 2.0 - w
        
        # Initialization 1: Grid
        # Estimate grid size
        # Try 4x6, 5x5, etc.
        splits = [(4, 6), (5, 5), (6, 4), (3, 7), (7, 3)]
        
        best_val_for_w = 0.0
        best_x_for_w = None
        
        for rows, cols in splits:
            if time.time() > deadline:
                break
            
            # Create grid points
            xs = []
            ys = []
            for i in range(rows):
                for j in range(cols):
                    x = (j + 0.5) * w / cols
                    y = (i + 0.5) * h / rows
                    xs.append(x)
                    ys.append(y)
            
            if len(xs) < N:
                continue
                
            xs = np.array(xs[:N])
            ys = np.array(ys[:N])
            
            # Initial radii
            # Estimate max possible r
            min_dist = np.inf
            for i in range(N):
                for j in range(i+1, N):
                    d = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                    if d < min_dist:
                        min_dist = d
            for i in range(N):
                d_w = min(xs[i], w-xs[i], ys[i], h-ys[i])
                if d_w < min_dist:
                    min_dist = d_w
            
            r_init = min_dist / 2.0 * 0.8
            rs = np.full(N, r_init)
            
            val, res_x = solve_for_w(w, xs, ys, rs)
            
            if val > best_val_for_w:
                best_val_for_w = val
                best_x_for_w = res_x
        
        # Initialization 2: Hexagonal
        # This is more complex to set up for arbitrary w,h, so we skip for now to save time
        # unless the grid didn't work well.
        
        if best_x_for_w is not None and best_val_for_w > best_sum_r:
            best_sum_r = best_val_for_w
            best_result = best_x_for_w
            
    if best_result is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    # best_result is [x1, y1, r1, ...]
    # Reshape to (N, 3)
    xs = best_result[0:3*N:3]
    ys = best_result[1:3*N:3]
    rs = best_result[2:3*N:3]
    
    return np.column_stack([xs, ys, rs])

# EVOLVE-BLOCK-END
