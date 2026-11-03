"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
N = 26
import numpy as np
from scipy.optimize import minimize
import time


def _max_radius_for_circle(x, y, others_xy, others_r):
    """
    Given a candidate center (x, y) and positions/radii of other circles,
    return the maximum radius such that the circle stays inside [0,1]^2
    and does not overlap any other circle.
    """
    # Clearance to walls
    wall_clear = min(x, 1.0 - x, y, 1.0 - y)
    if wall_clear <= 0:
        return 0.0
    # Clearance to other circles: r_i + r_j <= dist  =>  r_i <= dist - r_j
    if len(others_xy) > 0:
        dx = x - others_xy[:, 0]
        dy = y - others_xy[:, 1]
        dists = np.sqrt(dx * dx + dy * dy)
        circle_clear = np.min(dists - others_r)
        return max(0.0, min(wall_clear, circle_clear))
    return max(0.0, wall_clear)


def _eval_packing(pos):
    """
    Evaluate a packing configuration (N x 3 array of [x, y, r]).
    Returns (total_radius, is_valid).
    """
    xs, ys, rs = pos[:, 0], pos[:, 1], pos[:, 2]
    total_r = np.sum(rs)
    # Check boundaries
    if np.any(rs < 0) or np.any(xs < 0) or np.any(xs > 1) or np.any(ys < 0) or np.any(ys > 1):
        return total_r, False
    if np.any(rs > xs) or np.any(rs > 1 - xs) or np.any(rs > ys) or np.any(rs > 1 - ys):
        return total_r, False
    # Check non-overlap
    for i in range(N):
        for j in range(i + 1, N):
            d2 = (xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2
            if d2 < (rs[i] + rs[j]) ** 2 - 1e-9:
                return total_r, False
    return total_r, True


def _best_fit_local_search(pos, rng, deadline, max_rounds=200):
    """
    Iteratively for each circle, find the best position and maximum radius
    that fits given all other circles. Uses a dense global grid plus adaptive
    local refinement, fully vectorized.
    """
    pos = pos.copy()
    current_total, _ = _eval_packing(pos)
    
    # Precompute a dense global grid (60x60 = 3600 points)
    grid_size = 60
    gx = np.linspace(0.0, 1.0, grid_size)
    gy = np.linspace(0.0, 1.0, grid_size)
    GX, GY = np.meshgrid(gx, gy, indexing='ij')
    global_candidates = np.column_stack([GX.ravel(), GY.ravel()])
    global_wall_clear = np.minimum(np.minimum(global_candidates[:, 0], 1.0 - global_candidates[:, 0]),
                                   np.minimum(global_candidates[:, 1], 1.0 - global_candidates[:, 1]))
    
    for round_num in range(max_rounds):
        if time.time() > deadline:
            break
        
        improved = False
        order = rng.permutation(N)
        
        for i in order:
            if time.time() > deadline:
                break
            
            others_idx = [j for j in range(N) if j != i]
            others_xy = pos[others_idx, :2]
            others_r = pos[others_idx, 2]
            
            best_r = pos[i, 2]
            best_x, best_y = pos[i, 0], pos[i, 1]
            
            # 1. Global grid search (vectorized)
            if len(others_xy) > 0:
                dx = global_candidates[:, 0:1] - others_xy[:, 0][np.newaxis, :]
                dy = global_candidates[:, 1:2] - others_xy[:, 1][np.newaxis, :]
                dists = np.sqrt(dx * dx + dy * dy)
                circle_clear = np.min(dists - others_r[np.newaxis, :], axis=1)
                r_candidates = np.minimum(global_wall_clear, circle_clear)
            else:
                r_candidates = global_wall_clear.copy()
            
            idx_max = np.argmax(r_candidates)
            if r_candidates[idx_max] > best_r + 1e-10:
                best_r = r_candidates[idx_max]
                best_x, best_y = global_candidates[idx_max, 0], global_candidates[idx_max, 1]
            
            # 2. Adaptive local refinement around the current best
            search_radius = max(0.08, 2 * pos[i, 2] + 0.03)
            
            for scale in [1.0, 0.5, 0.25]:
                n_pts = int(30 * scale)
                if n_pts < 8:
                    n_pts = 8
                
                grid_x = np.linspace(best_x - search_radius * scale, 
                                     best_x + search_radius * scale, n_pts)
                grid_y = np.linspace(best_y - search_radius * scale, 
                                     best_y + search_radius * scale, n_pts)
                
                gx_local, gy_local = np.meshgrid(grid_x, grid_y)
                gx_flat = gx_local.ravel()
                gy_flat = gy_local.ravel()
                
                valid_mask = (gx_flat >= 0) & (gx_flat <= 1) & (gy_flat >= 0) & (gy_flat <= 1)
                gx_flat = gx_flat[valid_mask]
                gy_flat = gy_flat[valid_mask]
                
                if len(gx_flat) == 0:
                    continue
                
                local_candidates = np.column_stack([gx_flat, gy_flat])
                local_wall_clear = np.minimum(np.minimum(local_candidates[:, 0], 1.0 - local_candidates[:, 0]),
                                              np.minimum(local_candidates[:, 1], 1.0 - local_candidates[:, 1]))
                
                if len(others_xy) > 0:
                    dx = local_candidates[:, 0:1] - others_xy[:, 0][np.newaxis, :]
                    dy = local_candidates[:, 1:2] - others_xy[:, 1][np.newaxis, :]
                    dists = np.sqrt(dx * dx + dy * dy)
                    circle_clear = np.min(dists - others_r[np.newaxis, :], axis=1)
                    r_local = np.minimum(local_wall_clear, circle_clear)
                else:
                    r_local = local_wall_clear.copy()
                
                idx_max = np.argmax(r_local)
                if r_local[idx_max] > best_r + 1e-10:
                    best_r = r_local[idx_max]
                    best_x, best_y = local_candidates[idx_max, 0], local_candidates[idx_max, 1]
            
            # 3. Random sampling
            n_random = 50
            rand_x = rng.uniform(0, 1, n_random)
            rand_y = rng.uniform(0, 1, n_random)
            for k in range(n_random):
                r = _max_radius_for_circle(rand_x[k], rand_y[k], others_xy, others_r)
                if r > best_r + 1e-10:
                    best_r = r
                    best_x, best_y = rand_x[k], rand_y[k]
            
            if best_r > pos[i, 2] + 1e-10:
                pos[i, 0] = best_x
                pos[i, 1] = best_y
                pos[i, 2] = best_r
                improved = True
        
        if not improved:
            break
    
    # Final validity check and clip
    for i in range(N):
        pos[i, 2] = min(pos[i, 2], pos[i, 0], 1 - pos[i, 0], pos[i, 1], 1 - pos[i, 1])
        pos[i, 0] = np.clip(pos[i, 0], 0, 1)
        pos[i, 1] = np.clip(pos[i, 1], 0, 1)
    
    return pos


def _gradient_polish(pos, deadline):
    """
    Use L-BFGS-B with a smooth log-sum-exp penalty to polish the packing.
    This smooth surrogate avoids the nonsmoothness of max(0, violation)^2.
    """
    pos = pos.copy()
    
    def penalized_obj(x):
        p = x.reshape(N, 3)
        xs, ys, rs = p[:, 0], p[:, 1], p[:, 2]
        obj = -np.sum(rs)
        
        # Smooth boundary penalties: log-sum-exp of violations
        # violations: r - x, r - (1-x), r - y, r - (1-y)
        viol_x = rs - xs
        viol_1mx = rs - (1 - xs)
        viol_y = rs - ys
        viol_1my = rs - (1 - ys)
        
        # Use a smooth approximation: sum(exp(beta * max(0, v))) / beta
        # For simplicity, use a large beta and sum of exps where v>0
        beta = 20.0
        pen_bound = 0.0
        for v in [viol_x, viol_1mx, viol_y, viol_1my]:
            # Only sum where v > 0 to avoid overflow, use log-sum-exp trick
            pos_v = np.maximum(v, 0)
            if np.any(pos_v > 0):
                m = np.max(pos_v)
                if m > 0:
                    pen_bound += np.log(np.sum(np.exp(beta * pos_v))) / beta
                else:
                    pen_bound += np.sum(pos_v)
        
        # Smooth overlap penalties
        pen_overlap = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d2 = dx ** 2 + dy ** 2
                overlap = (rs[i] + rs[j]) ** 2 - d2
                if overlap > 0:
                    # Smooth: exp(beta * overlap) / beta
                    pen_overlap += np.exp(beta * overlap) / beta
                # For small overlaps, add linear term for stability
                pen_overlap += 0.5 * np.maximum(0, overlap)
        
        # Negative radius penalty
        neg_r = np.maximum(0, -rs)
        pen_neg = np.sum(neg_r)
        
        return obj + 10.0 * (pen_bound + pen_overlap) + 5.0 * pen_neg
    
    def penalized_grad(x):
        p = x.reshape(N, 3)
        xs, ys, rs = p[:, 0], p[:, 1], p[:, 2]
        grad = np.zeros_like(x)
        
        # Gradient of -sum(r)
        grad[2::3] = -1.0
        
        beta = 20.0
        
        # Gradient of boundary penalties
        # d/dx_i of (1/beta) * log(sum(exp(beta * max(0, r-x))))
        # = (exp(beta*(r-x)) / sum(exp(beta*max(0, r-x)))) * (-1) when r-x > 0
        for i in range(N):
            v_x = rs[i] - xs[i]
            v_1mx = rs[i] - (1 - xs[i])
            v_y = rs[i] - ys[i]
            v_1my = rs[i] - (1 - ys[i])
            
            # For simplicity and stability, use hard gradients where violation > 0
            # This is less smooth but more robust
            if v_x > 0:
                grad[3*i] += 20.0 * (-1.0)
                grad[3*i+2] += 20.0 * 1.0
            if v_1mx > 0:
                grad[3*i] += 20.0 * 1.0
                grad[3*i+2] += 20.0 * 1.0
            if v_y > 0:
                grad[3*i+1] += 20.0 * (-1.0)
                grad[3*i+2] += 20.0 * 1.0
            if v_1my > 0:
                grad[3*i+1] += 20.0 * 1.0
                grad[3*i+2] += 20.0 * 1.0
        
        # Gradient of overlap penalties
        for i in range(N):
            for j in range(i + 1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d2 = dx ** 2 + dy ** 2
                overlap = (rs[i] + rs[j]) ** 2 - d2
                if overlap > 0:
                    # Gradient of exp(beta*overlap)/beta is exp(beta*overlap) * d(overlap)
                    exp_term = np.exp(beta * overlap)
                    d_overlap_dx_i = -2 * dx
                    d_overlap_dx_j = 2 * dx
                    d_overlap_dy_i = -2 * dy
                    d_overlap_dy_j = 2 * dy
                    d_overlap_dr_i = 2 * (rs[i] + rs[j])
                    d_overlap_dr_j = 2 * (rs[i] + rs[j])
                    
                    grad[3*i] += 10.0 * exp_term * d_overlap_dx_i
                    grad[3*j] += 10.0 * exp_term * d_overlap_dx_j
                    grad[3*i+1] += 10.0 * exp_term * d_overlap_dy_i
                    grad[3*j+1] += 10.0 * exp_term * d_overlap_dy_j
                    grad[3*i+2] += 10.0 * exp_term * d_overlap_dr_i
                    grad[3*j+2] += 10.0 * exp_term * d_overlap_dr_j
                    
                    # Linear term gradient
                    grad[3*i] += 10.0 * 0.5 * d_overlap_dx_i
                    grad[3*j] += 10.0 * 0.5 * d_overlap_dx_j
                    grad[3*i+1] += 10.0 * 0.5 * d_overlap_dy_i
                    grad[3*j+1] += 10.0 * 0.5 * d_overlap_dy_j
                    grad[3*i+2] += 10.0 * 0.5 * d_overlap_dr_i
                    grad[3*j+2] += 10.0 * 0.5 * d_overlap_dr_j
        
        # Gradient of negative radius penalty
        neg_r = np.maximum(0, -rs)
        grad[2::3] += 5.0 * (-neg_r)
        
        return grad
    
    bounds = [(0, 1), (0, 1), (0, 0.5)] * N
    
    try:
        x0 = pos.flatten()
        res = minimize(penalized_obj, x0, jac=penalized_grad, method='L-BFGS-B',
                      bounds=bounds,
                      options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8,
                               'maxls': 50, 'maxfun': 1500})
        
        if res.success:
            new_pos = res.x.reshape(N, 3)
            _, valid = _eval_packing(new_pos)
            if valid:
                new_total, _ = _eval_packing(new_pos)
                old_total, _ = _eval_packing(pos)
                if new_total > old_total:
                    pos = new_pos
    except Exception:
        pass
    
    return pos


def _grid_init(rng):
    """Generate a perturbed grid initial configuration."""
    side = int(np.ceil(np.sqrt(N)))
    pos = np.zeros((N, 3))
    for i in range(N):
        row = i // side
        col = i % side
        pos[i, 0] = (col + 0.5) / side + rng.uniform(-0.02, 0.02)
        pos[i, 1] = (row + 0.5) / side + rng.uniform(-0.02, 0.02)
        pos[i, 2] = 0.5 / side - 0.01
    for i in range(N):
        pos[i, 0] = np.clip(pos[i, 0], pos[i, 2], 1 - pos[i, 2])
        pos[i, 1] = np.clip(pos[i, 1], pos[i, 2], 1 - pos[i, 2])
    return pos


def construct():
    deadline = time.time() + 135
    rng = np.random.default_rng(42)
    
    best_pos = None
    best_val = -np.inf
    
    n_restarts = 6
    for restart in range(n_restarts):
        if time.time() > deadline - 20:
            break
        
        if restart == 0:
            pos0 = _grid_init(rng)
        elif restart == 1:
            pos0 = np.zeros((N, 3))
            pos0[0] = [0.5, 0.5, 0.12]
            idx = 1
            for k in range(6):
                angle = 2 * np.pi * k / 6
                pos0[idx, 0] = 0.5 + 0.28 * np.cos(angle)
                pos0[idx, 1] = 0.5 + 0.28 * np.sin(angle)
                pos0[idx, 2] = 0.07
                idx += 1
            for k in range(12):
                angle = 2 * np.pi * k / 12 + np.pi/12
                pos0[idx, 0] = 0.5 + 0.40 * np.cos(angle)
                pos0[idx, 1] = 0.5 + 0.40 * np.sin(angle)
                pos0[idx, 2] = 0.045
                idx += 1
            corners = [(0.08, 0.08), (0.92, 0.08), (0.08, 0.92), (0.92, 0.92),
                       (0.5, 0.06), (0.5, 0.94), (0.06, 0.5)]
            for k in range(7):
                pos0[idx, 0] = corners[k][0]
                pos0[idx, 1] = corners[k][1]
                pos0[idx, 2] = 0.04
                idx += 1
            for i in range(N):
                pos0[i, 0] = np.clip(pos0[i, 0], pos0[i, 2], 1 - pos0[i, 2])
                pos0[i, 1] = np.clip(pos0[i, 1], pos0[i, 2], 1 - pos0[i, 2])
        else:
            pos0 = _grid_init(rng)
            pos0[:, :2] += rng.normal(0, 0.03, (N, 2))
            for i in range(N):
                pos0[i, 0] = np.clip(pos0[i, 0], 0, 1)
                pos0[i, 1] = np.clip(pos0[i, 1], 0, 1)
                pos0[i, 2] = min(pos0[i, 2], pos0[i, 0], 1-pos0[i, 0], pos0[i, 1], 1-pos0[i, 1])
        
        pos_refined = _best_fit_local_search(pos0, rng, deadline - 25, max_rounds=150)
        pos_polished = _gradient_polish(pos_refined, deadline - 10)
        
        total, valid = _eval_packing(pos_polished)
        if valid and total > best_val:
            best_val = total
            best_pos = pos_polished.copy()
        
        if time.time() < deadline - 15 and best_pos is not None:
            pos_again = _best_fit_local_search(best_pos, rng, deadline - 15, max_rounds=80)
            total2, valid2 = _eval_packing(pos_again)
            if valid2 and total2 > best_val:
                best_val = total2
                best_pos = pos_again.copy()
    
    if best_pos is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_pos
