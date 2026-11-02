"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit

    # Rectangle perimeter is 4, so 2*(w+h) = 4 => w+h = 2.
    # Let w = 2 * t, h = 2 * (1-t), where t in (0, 1).
    
    # Precompute pair indices for vectorized operations
    idx_i, idx_j = np.triu_indices(N, k=1)
    n_pairs = len(idx_i)

    def get_rect_dims(t):
        t = np.clip(t, 0.1, 0.9)
        w = 2.0 * t
        h = 2.0 * (1.0 - t)
        return w, h

    def extract_coords(state):
        """Extract t, x, y, r from state vector."""
        t = state[0]
        w, h = get_rect_dims(t)
        coords = np.zeros((N, 3))
        coords[:, 0] = state[1::3]
        coords[:, 1] = state[2::3]
        coords[:, 2] = state[3::3]
        return t, w, h, coords

    def evaluate(state, penalty_weight=100.0):
        """
        Evaluate score and penalty.
        Score is sum of radii.
        Penalty is sum of squared violations.
        """
        t, w, h, coords = extract_coords(state)
        radii = coords[:, 2]
        x = coords[:, 0]
        y = coords[:, 1]
        
        # Ensure radii are positive
        radii = np.maximum(radii, 1e-5)
        
        sum_r = np.sum(radii)
        
        # Boundary violations
        viol_left = np.maximum(0, radii - x)
        viol_right = np.maximum(0, x + radii - w)
        viol_bottom = np.maximum(0, radii - y)
        viol_top = np.maximum(0, y + radii - h)
        
        boundary_penalty = np.sum(viol_left**2) + np.sum(viol_right**2) + \
                           np.sum(viol_bottom**2) + np.sum(viol_top**2)
        
        # Overlap violations
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist_sq = dx*dx + dy*dy
        min_dist = radii[idx_i] + radii[idx_j]
        
        # Only compute sqrt for overlapping pairs to save time? 
        # For N=21, N_pairs=210, sqrt is cheap.
        dist = np.sqrt(dist_sq + 1e-12)
        overlap = np.maximum(0, min_dist - dist)
        overlap_penalty = np.sum(overlap**2)
        
        total_penalty = boundary_penalty + overlap_penalty
        score = sum_r - penalty_weight * total_penalty
        
        return score, total_penalty

    def check_valid(state, eps=1e-5):
        """
        Check if state is strictly valid.
        """
        t, w, h, coords = extract_coords(state)
        radii = coords[:, 2]
        x = coords[:, 0]
        y = coords[:, 1]
        
        if np.any(radii < 1e-6):
            return False
        
        # Boundary check
        if np.any(x - radii < -eps) or np.any(x + radii > w + eps) or \
           np.any(y - radii < -eps) or np.any(y + radii > h + eps):
            return False
        
        # Overlap check
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist_sq = dx*dx + dy*dy
        min_dist = radii[idx_i] + radii[idx_j]
        
        if np.any(dist_sq < (min_dist - eps)**2):
            return False
        
        return True

    def make_initial_state(t_init, jitter=0.0):
        """
        Create an initial feasible state.
        """
        w, h = get_rect_dims(t_init)
        
        # Estimate radius for hexagonal packing
        # Area of hexagonal lattice unit cell is 2*sqrt(3)*r^2
        # N cells * area = w*h * efficiency
        # r ~ sqrt(w*h / (N * 2*sqrt(3)))
        r_est = np.sqrt(w * h / (N * 2 * np.sqrt(3)))
        
        # Start with a slightly smaller radius to ensure feasibility
        r_current = r_est * 0.9
        
        state = np.zeros(1 + 3*N)
        state[0] = t_init
        
        positions = []
        y = r_current
        row_idx = 0
        while y <= h - r_current + 1e-9:
            if row_idx % 2 == 0:
                x_start = r_current
            else:
                x_start = r_current + r_current
            
            x = x_start
            while x <= w - r_current + 1e-9:
                positions.append((x, y, r_current))
                x += 2.0 * r_current
            y += np.sqrt(3) * r_current
            row_idx += 1
        
        # If we have more than N, take first N
        # If we have less, we need to add more or adjust
        if len(positions) < N:
            # Fallback: Grid
            side = int(np.ceil(np.sqrt(N)))
            idx = 0
            positions = []
            for i in range(side):
                for j in range(side):
                    if idx >= N:
                        break
                    gx = (i + 0.5) / side * w
                    gy = (j + 0.5) / side * h
                    r = 0.5 / side * min(w, h) * 0.9
                    positions.append((gx, gy, r))
                    idx += 1
        
        # Take first N
        positions = positions[:N]
        
        for i in range(N):
            if i < len(positions):
                x, y, r = positions[i]
                if jitter > 0:
                    x += random.uniform(-jitter, jitter)
                    y += random.uniform(-jitter, jitter)
                # Clamp to bounds
                x = max(r, min(w - r, x))
                y = max(r, min(h - r, y))
                state[1 + 3*i] = x
                state[1 + 3*i + 1] = y
                state[1 + 3*i + 2] = r
            else:
                # Should not happen if logic above is correct
                state[1 + 3*i] = w/2
                state[1 + 3*i + 1] = h/2
                state[1 + 3*i + 2] = 0.01
        
        return state

    def objective_and_grad(state, penalty_weight):
        """
        Compute objective and gradient for L-BFGS-B.
        Objective: -sum(r) + penalty_weight * penalty
        """
        t, w, h, coords = extract_coords(state)
        radii = coords[:, 2]
        x = coords[:, 0]
        y = coords[:, 1]
        
        # Clip radii for safety in gradient calc, but we rely on bounds
        radii_safe = np.maximum(radii, 1e-5)
        
        obj = -np.sum(radii)
        
        # --- Boundary Penalty ---
        # viol_l = r - x
        viol_l = radii_safe - x
        viol_r = x + radii_safe - w
        viol_b = radii_safe - y
        viol_t = y + radii_safe - h
        
        # Penalty is sum of squares of positive parts
        # P = sum (max(0, v)^2)
        # dP/dv = 2 * max(0, v)
        
        obj += penalty_weight * (np.sum(np.maximum(0, viol_l)**2) + 
                                  np.sum(np.maximum(0, viol_r)**2) + 
                                  np.sum(np.maximum(0, viol_b)**2) + 
                                  np.sum(np.maximum(0, viol_t)**2))
        
        # Gradient of Boundary Penalty
        grad_x_b = np.zeros(N)
        grad_y_b = np.zeros(N)
        grad_r_b = np.zeros(N)
        
        d_l = 2.0 * np.maximum(0, viol_l)
        d_r = 2.0 * np.maximum(0, viol_r)
        d_b = 2.0 * np.maximum(0, viol_b)
        d_t = 2.0 * np.maximum(0, viol_t)
        
        # d(viol_l)/dx = -1, d(viol_l)/dr = 1
        grad_x_b += -d_l
        grad_r_b += d_l
        
        # d(viol_r)/dx = 1, d(viol_r)/dr = 1
        grad_x_b += d_r
        grad_r_b += d_r
        
        # d(viol_b)/dy = -1, d(viol_b)/dr = 1
        grad_y_b += -d_b
        grad_r_b += d_b
        
        # d(viol_t)/dy = 1, d(viol_t)/dr = 1
        grad_y_b += d_t
        grad_r_b += d_t
        
        # --- Overlap Penalty ---
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist_sq = dx*dx + dy*dy
        dist = np.sqrt(dist_sq + 1e-12)
        min_dist = radii_safe[idx_i] + radii_safe[idx_j]
        
        overlap = np.maximum(0, min_dist - dist)
        
        obj += penalty_weight * np.sum(overlap**2)
        
        # Gradient of Overlap Penalty
        # d(overlap)/d(min_dist) = 1 (if overlap > 0)
        # d(overlap)/d(dist) = -1 (if overlap > 0)
        # d(dist)/dx_i = dx/dist, d(dist)/dx_j = -dx/dist
        
        mask = overlap > 0
        if np.any(mask):
            d_overlap = 2.0 * overlap[mask]
            
            # Distances for masked pairs
            dist_m = dist[mask]
            dx_m = dx[mask]
            dy_m = dy[mask]
            idx_i_m = idx_i[mask]
            idx_j_m = idx_j[mask]
            
            # d(dist)/dx_i = dx/dist
            d_dist_dx_i = dx_m / dist_m
            d_dist_dx_j = -dx_m / dist_m
            d_dist_dy_i = dy_m / dist_m
            d_dist_dy_j = -dy_m / dist_m
            
            # d(overlap)/dx_i = -1 * d(dist)/dx_i = -d_dist_dx_i
            # d(overlap)/dr_i = 1
            
            grad_x_o = np.zeros(N)
            grad_y_o = np.zeros(N)
            grad_r_o = np.zeros(N)
            
            np.add.at(grad_x_o, idx_i_m, -d_overlap * d_dist_dx_i)
            np.add.at(grad_x_o, idx_j_m, -d_overlap * d_dist_dx_j)
            
            np.add.at(grad_y_o, idx_i_m, -d_overlap * d_dist_dy_i)
            np.add.at(grad_y_o, idx_j_m, -d_overlap * d_dist_dy_j)
            
            np.add.at(grad_r_o, idx_i_m, d_overlap)
            np.add.at(grad_r_o, idx_j_m, d_overlap)
            
            grad_x_b += grad_x_o
            grad_y_b += grad_y_o
            grad_r_b += grad_r_o
        
        # --- Total Gradient ---
        grad = np.zeros_like(state)
        
        # t gradient (currently 0, as w,h are fixed during inner loop)
        # If we optimize t, we need d(w)/dt and d(h)/dt
        # d(viol_r)/dt = d(x+r - w)/dt = -dw/dt = -2
        # d(viol_t)/dt = d(y+r - h)/dt = -dh/dt = 2
        # This is complex. Let's fix t in the inner loop and optimize t in outer loop.
        
        grad[1::3] += penalty_weight * grad_x_b
        grad[2::3] += penalty_weight * grad_y_b
        grad[3::3] += penalty_weight * grad_r_b
        
        # Objective gradient: d(-sum r)/dr = -1
        grad[3::3] -= 1.0
        
        return obj, grad

    def optimize_state(state, time_limit, penalty_start=10.0, penalty_end=1000.0, steps=5):
        """
        Optimize a state using L-BFGS-B with annealed penalty weight.
        """
        best_state = state.copy()
        best_score, _ = evaluate(best_state, penalty_end)
        
        current_state = state.copy()
        
        bounds = [(0.1, 0.9)]
        for i in range(N):
            bounds.append((0, 2.0))  # x
            bounds.append((0, 2.0))  # y
            bounds.append((1e-4, 0.5))  # r
        
        for step in range(steps):
            if time.time() > time_limit:
                break
            
            # Anneal penalty weight
            frac = step / max(1, steps - 1)
            penalty_w = penalty_start * (penalty_end / penalty_start) ** frac
            
            def obj_func(s):
                return objective_and_grad(s, penalty_w)[0]
            
            def jac_func(s):
                return objective_and_grad(s, penalty_w)[1]
            
            try:
                result = minimize(
                    obj_func,
                    current_state,
                    jac=jac_func,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if result.fun < obj_func(current_state):
                    current_state = result.x
                
                # Check validity and update best
                score, pen = evaluate(current_state, penalty_end)
                if score > best_score:
                    best_state = current_state.copy()
                    best_score = score
                    
            except Exception:
                pass
            
            # If penalty is high, we might need to repair feasibility
            # Simple repair: shrink radii if invalid
            if not check_valid(current_state):
                # Shrink radii by 5%
                current_state[3::3] *= 0.95
                # Clamp positions
                t, w, h, _ = extract_coords(current_state)
                for i in range(N):
                    r = current_state[3 + 3*i]
                    x = current_state[1 + 3*i]
                    y = current_state[2 + 3*i]
                    x = max(r, min(w - r, x))
                    y = max(r, min(h - r, y))
                    current_state[1 + 3*i] = x
                    current_state[2 + 3*i] = y

        return best_state, best_score

    # Main Algorithm
    best_state = None
    best_score = -np.inf
    
    # Try a range of aspect ratios
    t_values = [0.5, 0.45, 0.55, 0.4, 0.6, 0.35, 0.65, 0.3, 0.7]
    random.shuffle(t_values)
    
    num_restarts = 15
    
    for restart in range(num_restarts):
        if time.time() > deadline - 20:
            break
        
        t_init = t_values[restart % len(t_values)]
        
        # Add some jitter to break symmetry
        jitter = 0.01
        state = make_initial_state(t_init, jitter=jitter)
        
        # Optimize
        remaining = deadline - 20 - time.time()
        if remaining < 5:
            continue
            
        # Allocate time for optimization
        opt_time = min(remaining * 0.8, 10.0)
        
        opt_state, opt_score = optimize_state(state, time.time() + opt_time)
        
        if opt_score > best_score:
            best_score = opt_score
            best_state = opt_state.copy()
            
        # If we have time, try a second pass with different jitter
        if time.time() < deadline - 10 and restart < num_restarts // 2:
            state2 = make_initial_state(t_init, jitter=jitter * 2)
            opt_time2 = min((deadline - 10 - time.time()) * 0.5, 5.0)
            opt_state2, opt_score2 = optimize_state(state2, time.time() + opt_time2)
            if opt_score2 > best_score:
                best_score = opt_score2
                best_state = opt_state2.copy()

    if best_state is None:
        best_state = make_initial_state(0.5)
    
    # Final Polish
    if time.time() < deadline - 5:
        final_state, final_score = optimize_state(best_state, deadline - 2, penalty_start=100.0, penalty_end=10000.0, steps=3)
        if final_score > best_score:
            best_state = final_state
            best_score = final_score

    # Ensure validity
    if not check_valid(best_state):
        # Repair by shrinking
        for i in range(50):
            if check_valid(best_state):
                break
            best_state[3::3] *= 0.95
            t, w, h, _ = extract_coords(best_state)
            for j in range(N):
                r = best_state[3 + 3*j]
                x = best_state[1 + 3*j]
                y = best_state[2 + 3*j]
                x = max(r, min(w - r, x))
                y = max(r, min(h - r, y))
                best_state[1 + 3*j] = x
                best_state[2 + 3*j] = y

    # Format output
    result = np.zeros((N, 3))
    t_final = best_state[0]
    w_final, h_final = get_rect_dims(t_final)
    
    for i in range(N):
        x = best_state[1 + 3*i]
        y = best_state[2 + 3*i]
        r = best_state[3 + 3*i]
        
        # Final clamping
        x = max(r, min(w_final - r, x))
        y = max(r, min(h_final - r, y))
        r = max(1e-6, r)
        
        result[i, 0] = x
        result[i, 1] = y
        result[i, 2] = r
        
    return result

# EVOLVE-BLOCK-END
