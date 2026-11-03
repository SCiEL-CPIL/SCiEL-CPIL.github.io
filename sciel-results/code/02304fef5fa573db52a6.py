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
    that fits given all other circles. Vectorized over a grid of candidate positions.
    """
    pos = pos.copy()
    current_total, _ = _eval_packing(pos)
    
    for round_num in range(max_rounds):
        if time.time() > deadline:
            break
        
        improved = False
        # Shuffle circle order for diversity
        order = rng.permutation(N)
        
        for i in order:
            if time.time() > deadline:
                break
            
            # Get other circles' positions and radii
            others_idx = [j for j in range(N) if j != i]
            others_xy = pos[others_idx, :2]
            others_r = pos[others_idx, 2]
            
            # Current best for this circle
            best_r = pos[i, 2]
            best_x, best_y = pos[i, 0], pos[i, 1]
            
            # Try a grid of candidate positions around the current position
            # Adaptive grid: finer near the current best
            cx, cy = pos[i, 0], pos[i, 1]
            search_radius = max(0.15, 2 * pos[i, 2] + 0.05)
            
            # Multi-scale grid search
            best_candidate_r = best_r
            best_candidate_pos = (best_x, best_y)
            
            for scale in [1.0, 0.5, 0.25]:
                n_pts = int(30 * scale)
                if n_pts < 8:
                    n_pts = 8
                
                # Create grid centered on best position so far
                grid_x = np.linspace(best_x - search_radius * scale, 
                                     best_x + search_radius * scale, n_pts)
                grid_y = np.linspace(best_y - search_radius * scale, 
                                     best_y + search_radius * scale, n_pts)
                
                # Vectorize over grid
                gx, gy = np.meshgrid(grid_x, grid_y)
                gx = gx.ravel()
                gy = gy.ravel()
                
                # Filter to valid range
                valid_mask = (gx >= 0) & (gx <= 1) & (gy >= 0) & (gy <= 1)
                gx = gx[valid_mask]
                gy = gy[valid_mask]
                
                if len(gx) == 0:
                    continue
                
                # Compute max radius for each grid point
                radii = np.array([
                    _max_radius_for_circle(gx[k], gy[k], others_xy, others_r)
                    for k in range(len(gx))
                ])
                
                # Find best
                idx_max = np.argmax(radii)
                if radii[idx_max] > best_candidate_r + 1e-10:
                    best_candidate_r = radii[idx_max]
                    best_candidate_pos = (gx[idx_max], gy[idx_max])
            
            # Also try some random positions
            n_random = 50
            rand_x = rng.uniform(0, 1, n_random)
            rand_y = rng.uniform(0, 1, n_random)
            for k in range(n_random):
                r = _max_radius_for_circle(rand_x[k], rand_y[k], others_xy, others_r)
                if r > best_candidate_r + 1e-10:
                    best_candidate_r = r
                    best_candidate_pos = (rand_x[k], rand_y[k])
            
            # Update if improved
            if best_candidate_r > best_r + 1e-10:
                pos[i, 0] = best_candidate_pos[0]
                pos[i, 1] = best_candidate_pos[1]
                pos[i, 2] = best_candidate_r
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
    Use L-BFGS-B with penalty terms to polish the packing.
    """
    pos = pos.copy()
    
    def penalized_obj(x):
        p = x.reshape(N, 3)
        xs, ys, rs = p[:, 0], p[:, 1], p[:, 2]
        # Objective: maximize sum of radii
        obj = -np.sum(rs)
        
        # Penalty for boundary violations
        pen = 0.0
        pen += np.sum(np.maximum(0, rs - xs) ** 2)
        pen += np.sum(np.maximum(0, rs - (1 - xs)) ** 2)
        pen += np.sum(np.maximum(0, rs - ys) ** 2)
        pen += np.sum(np.maximum(0, rs - (1 - ys)) ** 2)
        
        # Penalty for overlaps
        for i in range(N):
            for j in range(i + 1, N):
                d2 = (xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2
                overlap = np.maximum(0, (rs[i] + rs[j]) ** 2 - d2)
                pen += overlap ** 2
        
        # Penalty for negative radii
        pen += np.sum(np.maximum(0, -rs) ** 2)
        
        return obj + 10.0 * pen
    
    def penalized_grad(x):
        p = x.reshape(N, 3)
        xs, ys, rs = p[:, 0], p[:, 1], p[:, 2]
        grad = np.zeros_like(x)
        
        # Gradient of -sum(r)
        grad[2::3] = -1.0
        
        # Gradient of boundary penalties
        # d/dx_i of max(0, r_i - x_i)^2
        viol_x = np.maximum(0, rs - xs)
        grad[0::3] += 20.0 * (-viol_x)
        viol_1mx = np.maximum(0, rs - (1 - xs))
        grad[0::3] += 20.0 * viol_1mx
        viol_y = np.maximum(0, rs - ys)
        grad[1::3] += 20.0 * (-viol_y)
        viol_1my = np.maximum(0, rs - (1 - ys))
        grad[1::3] += 20.0 * viol_1my
        
        # Gradient of overlap penalties
        for i in range(N):
            for j in range(i + 1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d2 = dx ** 2 + dy ** 2
                overlap = np.maximum(0, (rs[i] + rs[j]) ** 2 - d2)
                if overlap > 0:
                    # d/dx_i of overlap^2
                    grad[3*i] += 20.0 * 2 * overlap * (-2 * dx)
                    grad[3*j] += 20.0 * 2 * overlap * (2 * dx)
                    grad[3*i+1] += 20.0 * 2 * overlap * (-2 * dy)
                    grad[3*j+1] += 20.0 * 2 * overlap * (2 * dy)
                    # d/dr_i
                    grad[3*i+2] += 20.0 * 2 * overlap * (2 * (rs[i] + rs[j]))
                    grad[3*j+2] += 20.0 * 2 * overlap * (2 * (rs[i] + rs[j]))
        
        # Gradient of negative radius penalty
        neg_r = np.maximum(0, -rs)
        grad[2::3] += 20.0 * (-neg_r)
        
        return grad
    
    bounds = [(0, 1), (0, 1), (0, 0.5)] * N
    
    try:
        x0 = pos.flatten()
        res = minimize(penalized_obj, x0, jac=penalized_grad, method='L-BFGS-B',
                      bounds=bounds,
                      options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8,
                               'maxls': 50, 'maxfun': 1000})
        
        if res.success:
            new_pos = res.x.reshape(N, 3)
            # Validate
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
        # Set initial radius to half cell size minus small margin
        pos[i, 2] = 0.5 / side - 0.01
    # Clip to valid range
    for i in range(N):
        pos[i, 0] = np.clip(pos[i, 0], pos[i, 2], 1 - pos[i, 2])
        pos[i, 1] = np.clip(pos[i, 1], pos[i, 2], 1 - pos[i, 2])
    return pos


def construct():
    deadline = time.time() + 135  # 15s safety margin before 150s limit
    rng = np.random.default_rng(42)
    
    best_pos = None
    best_val = -np.inf
    
    # Try multiple restarts with different initial configurations
    n_restarts = 6
    for restart in range(n_restarts):
        if time.time() > deadline - 20:
            break
        
        # Different initializations
        if restart == 0:
            pos0 = _grid_init(rng)
        elif restart == 1:
            # Ring-based init
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
            # Perturbed grid with larger perturbation
            pos0 = _grid_init(rng)
            pos0[:, :2] += rng.normal(0, 0.03, (N, 2))
            for i in range(N):
                pos0[i, 0] = np.clip(pos0[i, 0], 0, 1)
                pos0[i, 1] = np.clip(pos0[i, 1], 0, 1)
                pos0[i, 2] = min(pos0[i, 2], pos0[i, 0], 1-pos0[i, 0], pos0[i, 1], 1-pos0[i, 1])
        
        # Run best-fit local search
        pos_refined = _best_fit_local_search(pos0, rng, deadline - 25, max_rounds=150)
        
        # Gradient polish
        pos_polished = _gradient_polish(pos_refined, deadline - 10)
        
        total, valid = _eval_packing(pos_polished)
        if valid and total > best_val:
            best_val = total
            best_pos = pos_polished.copy()
        
        # If we have time, do another round of local search on the polished result
        if time.time() < deadline - 15 and best_pos is not None:
            pos_again = _best_fit_local_search(best_pos, rng, deadline - 15, max_rounds=80)
            total2, valid2 = _eval_packing(pos_again)
            if valid2 and total2 > best_val:
                best_val = total2
                best_pos = pos_again.copy()
    
    if best_pos is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_pos
