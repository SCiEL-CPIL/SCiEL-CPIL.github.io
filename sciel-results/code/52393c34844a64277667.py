import numpy as np
from scipy.optimize import minimize
import time

N = 32

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    def get_score(xy, r):
        return np.sum(r)

    def penalty_and_grad(params):
        """
        Objective: minimize -sum(r) + penalty_terms
        Returns (value, gradient)
        """
        n = N
        xy = params[:2*n].reshape(n, 2)
        r = params[2*n:]
        
        # Boundary violations
        d_left = xy[:, 0] - r
        d_right = 1 - xy[:, 0] - r
        d_bottom = xy[:, 1] - r
        d_top = 1 - xy[:, 1] - r
        
        # We want d >= 0. Penalty if d < 0.
        # Using smooth hinge or quadratic penalty.
        # Let's use quadratic penalty for smoothness: max(0, -d)^2
        viol_left = np.maximum(0, -d_left)
        viol_right = np.maximum(0, -d_right)
        viol_bottom = np.maximum(0, -d_bottom)
        viol_top = np.maximum(0, -d_top)
        
        boundary_pen = np.sum(viol_left**2) + np.sum(viol_right**2) + \
                       np.sum(viol_bottom**2) + np.sum(viol_top**2)
        
        # Overlap violations
        # dist(i,j) >= r_i + r_j
        # Let f(i,j) = dist(i,j) - (r_i + r_j)
        # We want f >= 0. Penalty if f < 0.
        
        # Compute distances and sums
        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        dist = np.sqrt(dx**2 + dy**2)
        # Avoid division by zero in gradients
        dist_safe = np.where(dist > 1e-12, dist, 1e-12)
        
        r_sum = r[:, None] + r[None, :]
        
        # Only consider i < j
        iu = np.triu_indices(n, k=1)
        f_iu = dist[iu] - r_sum[iu]
        
        # Penalty: max(0, -f)^2
        viol_overlap = np.maximum(0, -f_iu)
        overlap_pen = np.sum(viol_overlap**2)
        
        # Total objective
        obj = -np.sum(r) + 100.0 * boundary_pen + 100.0 * overlap_pen
        
        # Gradient calculation
        # grad_obj = -grad_sum_r + 100 * grad_boundary_pen + 100 * grad_overlap_pen
        
        grad = np.zeros_like(params)
        
        # 1. Gradient of -sum(r)
        grad[2*n:] = -1.0
        
        # 2. Gradient of boundary penalties
        # d/dx_i (max(0, -(x_i - r_i))^2)
        # Let v = max(0, -(x_i - r_i)). d(v^2)/dx_i = 2v * dv/dx_i
        # If x_i - r_i < 0, v = -(x_i - r_i), dv/dx_i = -1. So d/dx_i = -2v.
        # If x_i - r_i >= 0, v = 0, d/dx_i = 0.
        
        # Left: x_i - r_i
        # v = max(0, r_i - x_i)
        # d/dx_i = -2 * (r_i - x_i) if r_i > x_i else 0
        # d/dr_i = 2 * (r_i - x_i) if r_i > x_i else 0
        
        mask_left = (xy[:, 0] < r)
        if np.any(mask_left):
            v = r[mask_left] - xy[mask_left, 0]
            grad[mask_left, 0] += 100.0 * (-2 * v)
            grad[2*n + np.where(mask_left)[0]] += 100.0 * (2 * v)
            
        # Right: 1 - x_i - r_i
        # v = max(0, x_i + r_i - 1)
        # d/dx_i = 2 * (x_i + r_i - 1) if x_i + r_i > 1 else 0
        # d/dr_i = 2 * (x_i + r_i - 1) if x_i + r_i > 1 else 0
        
        mask_right = (xy[:, 0] + r > 1)
        if np.any(mask_right):
            v = xy[mask_right, 0] + r[mask_right] - 1
            grad[mask_right, 0] += 100.0 * (2 * v)
            grad[2*n + np.where(mask_right)[0]] += 100.0 * (2 * v)
            
        # Bottom: y_i - r_i
        mask_bottom = (xy[:, 1] < r)
        if np.any(mask_bottom):
            v = r[mask_bottom] - xy[mask_bottom, 1]
            grad[mask_bottom, 1] += 100.0 * (-2 * v)
            grad[2*n + np.where(mask_bottom)[0]] += 100.0 * (2 * v)
            
        # Top: 1 - y_i - r_i
        mask_top = (xy[:, 1] + r > 1)
        if np.any(mask_top):
            v = xy[mask_top, 1] + r[mask_top] - 1
            grad[mask_top, 1] += 100.0 * (2 * v)
            grad[2*n + np.where(mask_top)[0]] += 100.0 * (2 * v)
            
        # 3. Gradient of overlap penalties
        # f_ij = dist_ij - (r_i + r_j)
        # v_ij = max(0, -(f_ij)) = max(0, r_i + r_j - dist_ij)
        # Penalty term: v_ij^2
        # d(v^2)/d(dist) = 2v * (-1) = -2v
        # d(v^2)/d(r_i) = 2v * (1) = 2v
        # d(v^2)/d(r_j) = 2v * (1) = 2v
        
        # d(dist)/dx_i = (x_i - x_j)/dist
        # d(dist)/dy_i = (y_i - y_j)/dist
        
        # We need to accumulate gradients for each i and j.
        # Iterate over pairs with violation
        v_iu = viol_overlap
        if np.any(v_iu > 0):
            idx_i = iu[0]
            idx_j = iu[1]
            
            # Only process active violations for efficiency
            active = v_iu > 0
            idx_i_act = idx_i[active]
            idx_j_act = idx_j[active]
            v_act = v_iu[active]
            
            # Get coordinates
            xi = xy[idx_i_act, 0]
            xj = xy[idx_j_act, 0]
            yi = xy[idx_i_act, 1]
            yj = xy[idx_j_act, 1]
            
            dist_act = dist[idx_i_act, idx_j_act]
            dist_act_safe = np.where(dist_act > 1e-12, dist_act, 1e-12)
            
            dx_act = xi - xj
            dy_act = yi - yj
            
            # d(dist)/dx_i = dx/dist
            # d(dist)/dx_j = -dx/dist
            
            # Gradient contribution to x_i:
            # d(Pen)/dx_i = d(Pen)/d(dist) * d(dist)/dx_i = (-2v) * (dx/dist)
            # Gradient contribution to x_j:
            # d(Pen)/dx_j = d(Pen)/d(dist) * d(dist)/dx_j = (-2v) * (-dx/dist) = 2v * (dx/dist)
            
            coeff = 100.0 * (-2 * v_act) / dist_act_safe
            
            # Accumulate for i
            # np.add.at is needed for repeated indices
            np.add.at(grad, idx_i_act * 2, coeff * dx_act)
            np.add.at(grad, idx_i_act * 2 + 1, coeff * dy_act)
            
            # Accumulate for j
            np.add.at(grad, idx_j_act * 2, -coeff * dx_act)
            np.add.at(grad, idx_j_act * 2 + 1, -coeff * dy_act)
            
            # Radii gradients
            # d(Pen)/dr_i = 2v
            # d(Pen)/dr_j = 2v
            np.add.at(grad, 2*n + idx_i_act, 100.0 * (2 * v_act))
            np.add.at(grad, 2*n + idx_j_act, 100.0 * (2 * v_act))
            
        return obj, grad

    def is_valid(xy, r, tol=1e-6):
        n = len(r)
        if np.any(r < -tol):
            return False
        if np.any(xy[:, 0] - r < -tol) or np.any(1 - xy[:, 0] - r < -tol):
            return False
        if np.any(xy[:, 1] - r < -tol) or np.any(1 - xy[:, 1] - r < -tol):
            return False
        
        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        if np.any(dist[iu] < min_dist[iu] - tol):
            return False
        return True

    def generate_init(rng, strategy):
        """Generate initial parameters (xy, r) based on strategy."""
        if strategy == 0:
            # Grid 6x6
            side = 6
            xy = np.zeros((N, 2))
            r = np.full(N, 0.5/side)
            for i in range(N):
                row = i // side
                col = i % side
                xy[i] = [(col + 0.5)/side, (row + 0.5)/side]
                r[i] = 0.5/side
        elif strategy == 1:
            # Hexagonal approximation
            # 4 rows of 8? Or 5 rows?
            # Let's try 4 rows, 8 cols, with offset
            rows = 4
            cols = 8
            xy = np.zeros((N, 2))
            # Estimate radius
            # Width per col = 1/8 = 0.125
            # Height per row = 1/4 = 0.25
            # Hex spacing: dx = 0.125, dy = 0.25 * sqrt(3)/2? No, square grid with offset.
            # Let's just use a grid with slightly larger R and let optimizer fix it.
            for i in range(N):
                row = i // cols
                col = i % cols
                offset = 0.5/cols if row % 2 == 1 else 0
                x = (col + 0.5)/cols + offset
                y = (row + 0.5)/rows
                xy[i] = [x, y]
            r = np.full(N, 0.1)
        elif strategy == 2:
            # Random
            r = np.full(N, 0.08)
            xy = rng.uniform(r, 1-r, size=(N, 2))
        elif strategy == 3:
            # Sparse grid, larger R
            side = 5
            # 25 circles, 7 extra?
            # Just use 6x6 grid but start with R=0.1
            side = 6
            xy = np.zeros((N, 2))
            for i in range(N):
                row = i // side
                col = i % side
                xy[i] = [(col + 0.5)/side, (row + 0.5)/side]
            r = np.full(N, 0.1)
        else:
            # Clustered
            r = np.full(N, 0.05)
            xy = rng.uniform(r, 1-r, size=(N, 2))
            
        # Clamp positions to be strictly inside given r
        xy[:, 0] = np.clip(xy[:, 0], r, 1-r)
        xy[:, 1] = np.clip(xy[:, 1], r, 1-r)
        
        params = np.concatenate([xy.flatten(), r])
        return params

    best_params = None
    best_score = -np.inf
    
    rng = np.random.RandomState(42)
    
    # Multi-start optimization
    # We have ~145 seconds.
    # Each L-BFGS-B run takes ~1-5 seconds depending on convergence.
    # Let's allow ~20-30 restarts.
    
    num_restarts = 25
    strategies = [0, 1, 2, 3, 4]
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        strategy = restart % len(strategies)
        params_init = generate_init(rng, strategy)
        
        # Add small noise to break symmetry
        params_init = params_init + rng.normal(0, 0.001, size=params_init.shape)
        
        try:
            result = minimize(penalty_and_grad, params_init, method='L-BFGS-B',
                             jac=True, # Use analytic gradient
                             options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
            
            params_opt = result.x
            xy = params_opt[:2*N].reshape(N, 2)
            r = params_opt[2*N:]
            
            # Check validity
            if is_valid(xy, r):
                score = get_score(xy, r)
                if score > best_score:
                    best_score = score
                    best_params = params_opt.copy()
            else:
                # If invalid, check if it's "close" and try a quick repair?
                # Or just ignore for now.
                # We can also try to project it back to validity.
                # For now, let's just record if it's better than current best valid,
                # but we can only return valid ones.
                # However, if the penalty is 0, it should be valid.
                # If penalty > 0, it's invalid.
                pass
                
        except Exception as e:
            continue
            
    if best_params is not None:
        xy = best_params[:2*N].reshape(N, 2)
        r = best_params[2*N:]
        return np.column_stack([xy, r])
    else:
        # Fallback: Grid 6x6
        side = 6
        cells = []
        for i in range(N):
            row = i // side
            col = i % side
            cells.append([(col + 0.5) / side, (row + 0.5) / side, 0.5 / side - 1e-3])
        return np.array(cells)
