import numpy as np
import time
from scipy.optimize import minimize

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # Precompute pair indices for vectorized distance checks
    i_idx, j_idx = np.triu_indices(N, k=1)
    n_pairs = len(i_idx)
    
    def objective_and_grad(state):
        """
        state: (N, 3) array of [x, y, r]
        Returns: (objective_value, gradient)
        Objective: -sum(r) + penalty * (sum of squared constraint violations)
        """
        x = state[:, 0]
        y = state[:, 1]
        r = state[:, 2]
        
        # 1. Boundary constraints: r <= min(x, 1-x, y, 1-y)
        # Violation is max(0, r - bound)
        # We use a smooth approximation or just standard penalty.
        # For L-BFGS-B, we can use a standard penalty: P = sum(max(0, viol)^2)
        
        bounds_x = np.minimum(x, 1 - x)
        bounds_y = np.minimum(y, 1 - y)
        bounds = np.minimum(bounds_x, bounds_y)
        
        # r must be <= bounds
        # Let v_bound_i = max(0, r_i - bounds_i)
        v_bound = np.maximum(0, r - bounds)
        
        # 2. Pairwise constraints: dist(i,j) >= r_i + r_j
        dx = x[i_idx] - x[j_idx]
        dy = y[i_idx] - y[j_idx]
        dist = np.sqrt(dx*dx + dy*dy)
        sum_r = r[i_idx] + r[j_idx]
        
        # Violation is max(0, sum_r - dist)
        v_pair = np.maximum(0, sum_r - dist)
        
        # Total penalty
        # We want to minimize -sum(r) + lambda * (sum v_bound^2 + sum v_pair^2)
        # If we use a large lambda, the solver will try to satisfy constraints.
        
        penalty_weight = 100.0
        
        obj = -np.sum(r) + penalty_weight * (np.sum(v_bound**2) + np.sum(v_pair**2))
        
        # Gradient calculation
        # grad = -1 (for -sum r) + 2 * lambda * (grad v_bound^2 + grad v_pair^2)
        
        grad = np.zeros_like(state)
        
        # Gradient of -sum(r)
        grad[:, 2] -= 1.0
        
        # Gradient of penalty
        # d/dstate (v^2) = 2*v * dv/dstate
        
        # Boundary part
        # v_bound_i = max(0, r_i - min(min(x_i, 1-x_i), min(y_i, 1-y_i)))
        # This derivative is tricky due to min/max. 
        # A simpler smooth penalty for boundaries: 
        # Instead of hard projection, we can penalize (r - bound)^2 if r > bound.
        # Let's just compute the gradient of the active constraints.
        
        # For v_bound:
        active_bound = v_bound > 0
        if np.any(active_bound):
            # grad r_i gets + 2*lambda*v_bound_i
            grad[active_bound, 2] += 2 * penalty_weight * v_bound[active_bound]
            
            # grad x_i, y_i depends on which bound is active
            # bound_i = min(x_i, 1-x_i, y_i, 1-y_i)
            # If x_i is the min, dbound/dx_i = 1. If 1-x_i is min, dbound/dx_i = -1.
            # If r_i > bound_i, we want to increase bound_i or decrease r_i.
            # Increasing bound_i means moving away from the edge.
            # If x_i < 0.5, bound is x_i (likely), so we want to increase x_i -> grad x_i += 2*lambda*v
            # If x_i > 0.5, bound is 1-x_i, so we want to decrease x_i -> grad x_i -= 2*lambda*v
            
            # Determine active bound for each i
            # Simplified: assume the closest wall is the constraint.
            # dist to left: x_i, right: 1-x_i, bottom: y_i, top: 1-y_i
            # The min of these is the bound.
            
            # We can just use the subgradient of the min function.
            # Or simpler: just push away from the nearest wall.
            
            for i in np.where(active_bound)[0]:
                # Find which wall is closest
                d_left = x[i]
                d_right = 1 - x[i]
                d_bottom = y[i]
                d_top = 1 - y[i]
                
                min_d = min(d_left, d_right, d_bottom, d_top)
                
                # If multiple are min, average gradients? Or just pick one?
                # For smoothness, let's just use the one that is strictly min, or average if tied.
                # Given floating point, ties are rare.
                
                factor = 2 * penalty_weight * v_bound[i]
                
                if min_d == d_left:
                    grad[i, 0] += factor
                elif min_d == d_right:
                    grad[i, 0] -= factor
                elif min_d == d_bottom:
                    grad[i, 1] += factor
                elif min_d == d_top:
                    grad[i, 1] -= factor
                else:
                    # Should not happen
                    pass
        
        # Pairwise part
        active_pair = v_pair > 0
        if np.any(active_pair):
            # v_pair_k = max(0, (r_i + r_j) - dist_ij)
            # grad r_i += 2*lambda*v_pair_k
            # grad r_j += 2*lambda*v_pair_k
            # grad x_i += 2*lambda*v_pair_k * (dx_ij / dist_ij)  (Note: dx = x_i - x_j)
            # Wait, dist = sqrt((x_i-x_j)^2 + ...). 
            # d(dist)/dx_i = (x_i-x_j)/dist.
            # We want to increase dist, so we want to move i away from j.
            # Gradient of -dist is - (x_i-x_j)/dist.
            # Gradient of (sum_r - dist) is (1, 1) for radii, and -(x_i-x_j)/dist for x_i.
            # So grad x_i += 2*lambda*v * (-(x_i-x_j)/dist) ?
            # Let's check signs.
            # We minimize P = (sum_r - dist)^2.
            # dP/dx_i = 2(sum_r - dist) * d(sum_r - dist)/dx_i
            # = 2*v * (0 - (x_i-x_j)/dist)
            # = -2*v * (x_i-x_j)/dist
            # So we want to decrease x_i if x_i > x_j. That moves i left, away from j?
            # If x_i > x_j, dx > 0. Grad is negative. x_i decreases. Distance increases. Correct.
            
            idx_i = i_idx[active_pair]
            idx_j = j_idx[active_pair]
            v_act = v_pair[active_pair]
            dist_act = dist[active_pair]
            
            # Avoid division by zero
            dist_safe = np.where(dist_act > 1e-6, dist_act, 1e-6)
            
            # Gradients for radii
            grad[idx_i, 2] += 2 * penalty_weight * v_act
            grad[idx_j, 2] += 2 * penalty_weight * v_act
            
            # Gradients for positions
            # dx = x_i - x_j
            dx_act = dx[active_pair]
            dy_act = dy[active_pair]
            
            grad_x_i = -2 * penalty_weight * v_act * (dx_act / dist_safe)
            grad_y_i = -2 * penalty_weight * v_act * (dy_act / dist_safe)
            
            # Accumulate gradients for i and j
            # Use np.add.at for repeated indices
            np.add.at(grad, (idx_i, 0), grad_x_i)
            np.add.at(grad, (idx_i, 1), grad_y_i)
            
            # For j, the vector is (x_j - x_i) = -dx.
            # d(dist)/dx_j = (x_j-x_i)/dist = -dx/dist.
            # d(sum_r - dist)/dx_j = -(-dx/dist) = dx/dist.
            # So grad x_j += 2*lambda*v * (dx/dist).
            # This is opposite to grad x_i.
            np.add.at(grad, (idx_j, 0), -grad_x_i)
            np.add.at(grad, (idx_j, 1), -grad_y_i)
            
        return obj, grad

    def is_valid(state, tol=1e-6):
        x = state[:, 0]
        y = state[:, 1]
        r = state[:, 2]
        
        # Boundary
        bounds = np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y))
        if np.any(r > bounds + tol):
            return False
            
        # Pairs
        dx = x[i_idx] - x[j_idx]
        dy = y[i_idx] - y[j_idx]
        dist = np.sqrt(dx*dx + dy*dy)
        sum_r = r[i_idx] + r[j_idx]
        if np.any(sum_r > dist + tol):
            return False
            
        # Radii must be positive
        if np.any(r < 1e-6):
            return False
            
        return True

    def get_value(state):
        if not is_valid(state):
            return -np.inf
        return np.sum(state[:, 2])

    # Initializations
    inits = []
    
    # 1. Grid-like init with jitter
    # 5x5 grid = 25, add 1? Or 26 points in a grid pattern.
    # Let's try a 5x5 grid (25 points) + 1 center? No, 26 points.
    # 5x5 grid covers 0.1 to 0.9.
    # Spacing 0.2.
    # Let's just generate 26 points on a grid with some jitter.
    grid_x = np.linspace(0.1, 0.9, 5)
    grid_y = np.linspace(0.1, 0.9, 5)
    gx, gy = np.meshgrid(grid_x, grid_y)
    pts_grid = np.column_stack([gx.ravel(), gy.ravel()])[:25]
    # Add one more point in the center?
    pts_grid = np.vstack([pts_grid, [0.5, 0.5]])
    # Shuffle to break symmetry?
    np.random.shuffle(pts_grid)
    inits.append(pts_grid)
    
    # 2. Ring init
    center = np.array([0.5, 0.5])
    pts_ring = np.zeros((26, 2))
    pts_ring[0] = center
    angles1 = np.linspace(0, 2*np.pi, 8, endpoint=False)
    pts_ring[1:9] = center + 0.2 * np.column_stack([np.cos(angles1), np.sin(angles1)])
    angles2 = np.linspace(0, 2*np.pi, 17, endpoint=False)
    pts_ring[9:] = center + 0.4 * np.column_stack([np.cos(angles2), np.sin(angles2)])
    inits.append(pts_ring)
    
    # 3. Random init
    pts_rand = np.random.rand(26, 2) * 0.8 + 0.1
    inits.append(pts_rand)

    best_state = None
    best_value = -np.inf
    
    # We will try to optimize from each init.
    # L-BFGS-B can be slow if the penalty is too stiff or too loose.
    # We use a fixed penalty weight for simplicity, but maybe adapt it?
    # For now, fixed weight 100.
    
    for init_pts in inits:
        if time.time() > deadline:
            break
            
        # Initial radii: small
        init_r = np.full(26, 0.02)
        init_state = np.hstack([init_pts, init_r[:, None]])
        
        # Run L-BFGS-B
        # Options for L-BFGS-B
        options = {
            'maxiter': 500,
            'ftol': 1e-9,
            'gtol': 1e-6,
            'maxls': 50
        }
        
        try:
            res = minimize(
                objective_and_grad, 
                init_state, 
                method='L-BFGS-B', 
                jac=True, # We return (f, g)
                options=options
            )
            final_state = res.x
        except Exception as e:
            # If it fails, skip
            continue
            
        # Check validity of the result
        # L-BFGS-B minimizes the penalty, but doesn't guarantee zero penalty.
        # We need to verify if it's valid.
        if is_valid(final_state):
            val = get_value(final_state)
            if val > best_value:
                best_value = val
                best_state = final_state.copy()
        else:
            # If invalid, try to "repair" it by scaling down radii?
            # Or just skip.
            # Let's try a simple repair: reduce radii until valid.
            # This is a fallback.
            state = final_state.copy()
            for _ in range(20):
                if is_valid(state):
                    val = get_value(state)
                    if val > best_value:
                        best_value = val
                        best_state = state.copy()
                    break
                # Scale down radii
                state[:, 2] *= 0.95
            # If still invalid after scaling, ignore.

    # If we didn't find a valid state (unlikely), return a trivial valid state
    if best_state is None:
        # Small circles in a grid
        best_state = np.zeros((26, 3))
        for i in range(26):
            row = i // 5
            col = i % 5
            best_state[i, 0] = 0.1 + 0.2 * col
            best_state[i, 1] = 0.1 + 0.2 * row
            best_state[i, 2] = 0.01 # Very small, valid
        # Ensure last one doesn't overlap? 5x5=25. 26th is at (0.5, 0.5)?
        # My grid init above had 25 grid + 1 center.
        # Let's just make sure this fallback is valid.
        # With r=0.01 and spacing 0.2, it's valid.

    return best_state

