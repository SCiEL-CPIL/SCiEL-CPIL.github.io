"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 142  # Safety margin before 150s limit

    def get_rect_dims(t):
        t = np.clip(t, 0.1, 0.9)
        w = 2.0 * t
        h = 2.0 * (1.0 - t)
        return w, h

    def evaluate(state):
        """
        Returns (score, penalty). Score = sum_r - penalty.
        Higher score is better. Valid solutions have penalty=0.
        """
        t = state[0]
        w, h = get_rect_dims(t)
        
        penalty = 0.0
        x = state[1::3]
        y = state[2::3]
        radii = state[3::3]
        
        if np.any(radii < 1e-5):
            penalty += 0.1 * np.sum(radii[radii < 1e-5])
            radii = np.maximum(radii, 1e-5)
        
        sum_r = np.sum(radii)
        
        # Boundary violations
        viol_left = np.maximum(0, radii - x)
        viol_right = np.maximum(0, x + radii - w)
        viol_bottom = np.maximum(0, radii - y)
        viol_top = np.maximum(0, y + radii - h)
        
        penalty += 500.0 * (np.sum(viol_left) + np.sum(viol_right) + np.sum(viol_bottom) + np.sum(viol_top))
        
        # Overlap violations - vectorized
        idx_i, idx_j = np.triu_indices(N, k=1)
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist_sq = dx*dx + dy*dy
        min_dist = radii[idx_i] + radii[idx_j]
        
        overlap_mask = dist_sq < min_dist * min_dist
        if np.any(overlap_mask):
            dist = np.sqrt(dist_sq[overlap_mask])
            overlaps = min_dist[overlap_mask] - dist
            penalty += 1000.0 * np.sum(overlaps)
        
        score = sum_r - penalty
        return score, penalty

    def check_valid(state, eps=1e-7):
        """
        Check if state is valid: all circles inside rectangle, no overlaps.
        """
        t = state[0]
        w, h = get_rect_dims(t)
        
        x = state[1::3]
        y = state[2::3]
        radii = state[3::3]
        
        if np.any(radii < 1e-6):
            return False
        
        # Boundary check
        if np.any(x - radii < -eps) or np.any(x + radii > w + eps) or \
           np.any(y - radii < -eps) or np.any(y + radii > h + eps):
            return False
        
        # Overlap check
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                min_dist = radii[i] + radii[j] - eps
                if dist_sq < min_dist * min_dist:
                    return False
        return True

    def make_initial_hexagonal(t_init=0.5, jitter=0.0):
        """
        Create an initial state using a hexagonal lattice pattern.
        """
        w, h = get_rect_dims(t_init)
        state = np.zeros(1 + 3*N)
        state[0] = t_init
        r_est = np.sqrt(0.9069 * w * h / (N * np.pi))
        r_current = r_est
        positions = []
        
        for attempt in range(20):
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
            if len(positions) >= N:
                break
            else:
                r_current *= 0.95
        
        if len(positions) < N:
            side = 5
            idx = 0
            positions = []
            for i in range(side):
                for j in range(side):
                    if idx >= N:
                        break
                    gx = (i + 0.5) / side * w
                    gy = (j + 0.5) / side * h
                    r = 0.5 / side * min(w, h) * 0.95
                    positions.append((gx, gy, r))
                    idx += 1
        
        for i in range(N):
            if i < len(positions):
                x, y, r = positions[i]
                if jitter > 0:
                    x += random.uniform(-jitter, jitter)
                    y += random.uniform(-jitter, jitter)
                state[1 + 3*i] = x
                state[1 + 3*i + 1] = y
                state[1 + 3*i + 2] = r
            else:
                state[1 + 3*i] = w/2
                state[1 + 3*i + 1] = h/2
                state[1 + 3*i + 2] = 0.01
        return state

    def make_initial_grid(t_init=0.5, jitter=0.0):
        """
        Create an initial state using a square grid pattern.
        """
        w, h = get_rect_dims(t_init)
        state = np.zeros(1 + 3*N)
        state[0] = t_init
        
        # Try different grid configurations
        best_positions = None
        for rows in range(3, 8):
            cols = int(np.ceil(N / rows))
            if rows * cols < N:
                continue
            positions = []
            for i in range(rows):
                for j in range(cols):
                    if len(positions) >= N:
                        break
                    gx = (j + 0.5) / cols * w
                    gy = (i + 0.5) / rows * h
                    r = 0.5 * min(w / cols, h / rows) * 0.9
                    positions.append((gx, gy, r))
            if len(positions) >= N:
                if best_positions is None or np.mean([p[2] for p in positions]) > np.mean([p[2] for p in best_positions]):
                    best_positions = positions
        
        if best_positions is None:
            side = 5
            positions = []
            for i in range(side):
                for j in range(side):
                    if len(positions) >= N:
                        break
                    gx = (i + 0.5) / side * w
                    gy = (j + 0.5) / side * h
                    r = 0.5 / side * min(w, h) * 0.95
                    positions.append((gx, gy, r))
            best_positions = positions
        
        for i in range(N):
            x, y, r = best_positions[i]
            if jitter > 0:
                x += random.uniform(-jitter, jitter)
                y += random.uniform(-jitter, jitter)
            state[1 + 3*i] = x
            state[1 + 3*i + 1] = y
            state[1 + 3*i + 2] = r
        return state

    def make_initial_random(t_init=0.5):
        """
        Create a random initial state with non-overlapping circles.
        """
        w, h = get_rect_dims(t_init)
        state = np.zeros(1 + 3*N)
        state[0] = t_init
        
        positions = []
        max_attempts = 1000
        r_est = np.sqrt(0.9069 * w * h / (N * np.pi))
        
        for i in range(N):
            placed = False
            for attempt in range(max_attempts):
                r = r_est * random.uniform(0.8, 1.2)
                x = random.uniform(r, w - r)
                y = random.uniform(r, h - r)
                
                valid = True
                for (px, py, pr) in positions:
                    dx = x - px
                    dy = y - py
                    if dx*dx + dy*dy < (r + pr)**2:
                        valid = False
                        break
                
                if valid:
                    positions.append((x, y, r))
                    placed = True
                    break
            
            if not placed:
                positions.append((w/2, h/2, 0.01))
        
        for i in range(N):
            x, y, r = positions[i]
            state[1 + 3*i] = x
            state[1 + 3*i + 1] = y
            state[1 + 3*i + 2] = r
        return state

    def polish_gradient(state, time_limit, max_iter=2000, beta_scale=1.0):
        """
        Gradient-based polish using smooth penalty formulation.
        beta_scale controls the softness of the penalties.
        """
        idx_i, idx_j = np.triu_indices(N, k=1)
        n_pairs = len(idx_i)
        
        def objective_and_grad(s):
            t = s[0]
            w, h = get_rect_dims(t)
            
            x = s[1::3]
            y = s[2::3]
            radii = s[3::3]
            
            # Negative sum of radii (we minimize)
            obj = -np.sum(radii)
            
            # Small penalty for radius variance to encourage equal radii
            mean_r = np.mean(radii)
            obj += 0.05 * np.sum((radii - mean_r)**2)
            
            # Smooth boundary penalties using softplus
            beta_b = 0.005 * beta_scale
            viol_l = radii - x
            viol_r = x + radii - w
            viol_b = radii - y
            viol_t = y + radii - h
            
            sl = beta_b * np.log1p(np.exp(np.clip(viol_l / beta_b, -500, 500)))
            sr = beta_b * np.log1p(np.exp(np.clip(viol_r / beta_b, -500, 500)))
            sb = beta_b * np.log1p(np.exp(np.clip(viol_b / beta_b, -500, 500)))
            st = beta_b * np.log1p(np.exp(np.clip(viol_t / beta_b, -500, 500)))
            
            obj += 1000.0 * (np.sum(sl**2) + np.sum(sr**2) + np.sum(sb**2) + np.sum(st**2))
            
            # Overlap penalty - smooth
            dx = x[idx_i] - x[idx_j]
            dy = y[idx_i] - y[idx_j]
            dist_sq = dx*dx + dy*dy
            min_dist = radii[idx_i] + radii[idx_j]
            
            overlap_amount = min_dist**2 - dist_sq
            beta_o = 0.001 * beta_scale
            so = beta_o * np.log1p(np.exp(np.clip(overlap_amount / beta_o, -500, 500)))
            obj += 5000.0 * np.sum(so**2)
            
            # Gradient calculation
            grad = np.zeros_like(s)
            
            # d(-sum_r)/dr_i = -1
            grad[3::3] += -1.0
            
            # Gradient for variance penalty: 0.10 * (r_i - mean_r)
            grad[3::3] += 0.10 * (radii - mean_r)

            # Boundary gradients
            sig_l = 1.0 / (1.0 + np.exp(-np.clip(viol_l / beta_b, -500, 500)))
            sig_r = 1.0 / (1.0 + np.exp(-np.clip(viol_r / beta_b, -500, 500)))
            sig_b = 1.0 / (1.0 + np.exp(-np.clip(viol_b / beta_b, -500, 500)))
            sig_t = 1.0 / (1.0 + np.exp(-np.clip(viol_t / beta_b, -500, 500)))
            
            # x gradients
            grad[1::3] += 2000.0 * sl * sig_l * (-1.0)
            grad[1::3] += 2000.0 * sr * sig_r * (1.0)
            # y gradients
            grad[2::3] += 2000.0 * sb * sig_b * (-1.0)
            grad[2::3] += 2000.0 * st * sig_t * (1.0)
            # r gradients from boundaries
            grad[3::3] += 2000.0 * sl * sig_l * (1.0)
            grad[3::3] += 2000.0 * sr * sig_r * (1.0)
            grad[3::3] += 2000.0 * sb * sig_b * (1.0)
            grad[3::3] += 2000.0 * st * sig_t * (1.0)
            
            # Overlap gradients
            sig_o = 1.0 / (1.0 + np.exp(-np.clip(overlap_amount / beta_o, -500, 500)))
            factor = 10000.0 * so * sig_o
            
            for k in range(n_pairs):
                i = idx_i[k]
                j = idx_j[k]
                
                grad[1 + 3*i] += factor[k] * (-2.0 * dx[k])
                grad[1 + 3*j] += factor[k] * (2.0 * dx[k])
                grad[2 + 3*i] += factor[k] * (-2.0 * dy[k])
                grad[2 + 3*j] += factor[k] * (2.0 * dy[k])
                grad[3 + 3*i] += factor[k] * (2.0 * min_dist[k])
                grad[3 + 3*j] += factor[k] * (2.0 * min_dist[k])
            
            return obj, grad
        
        def objective(s):
            return objective_and_grad(s)[0]
        
        def gradient(s):
            return objective_and_grad(s)[1]
        
        bounds = [(0.1, 0.9)]
        for i in range(N):
            bounds.append((0, 2.0))  # x
            bounds.append((0, 2.0))  # y
            bounds.append((1e-4, 0.5))  # r
        
        try:
            result = minimize(
                objective,
                state,
                jac=gradient,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
            )
            if result.success or result.fun < objective(state):
                return result.x
        except Exception:
            pass
        
        return state

    def force_relax(state, time_limit, max_iter=5000):
        """
        Force-directed relaxation: vectorized version for speed.
        """
        best_state = state.copy()
        best_score, _ = evaluate(best_state)
        
        idx_i, idx_j = np.triu_indices(N, k=1)
        n_pairs = len(idx_i)
        
        step_size = 0.005
        patience = 0
        max_patience = 200
        
        for it in range(max_iter):
            if time.time() > time_limit:
                break
            
            t = best_state[0]
            w, h = get_rect_dims(t)
            
            x = best_state[1::3].copy()
            y = best_state[2::3].copy()
            radii = best_state[3::3].copy()
            
            fx = np.zeros(N)
            fy = np.zeros(N)
            fr = np.zeros(N)
            
            # Boundary forces (vectorized)
            viol_l = np.maximum(0, radii - x)
            viol_r = np.maximum(0, x + radii - w)
            viol_b = np.maximum(0, radii - y)
            viol_t = np.maximum(0, y + radii - h)
            fx += 10.0 * viol_l - 10.0 * viol_r
            fy += 10.0 * viol_b - 10.0 * viol_t
            
            # Pair forces (vectorized)
            dx = x[idx_i] - x[idx_j]
            dy = y[idx_i] - y[idx_j]
            dist_sq = dx*dx + dy*dy
            dist = np.sqrt(dist_sq + 1e-12)
            target = radii[idx_i] + radii[idx_j]
            diff = dist - target
            
            # Force magnitude
            k_spring = np.where(diff < 0, 50.0, 2.0)
            force_mag = k_spring * diff
            
            # Direction unit vectors
            inv_dist = 1.0 / np.maximum(dist, 1e-10)
            fx_dir = dx * inv_dist
            fy_dir = dy * inv_dist
            
            # Apply forces to pairs using np.add.at for accumulation
            np.add.at(fx, idx_i, force_mag * fx_dir)
            np.add.at(fy, idx_i, force_mag * fy_dir)
            np.add.at(fx, idx_j, -force_mag * fx_dir)
            np.add.at(fy, idx_j, -force_mag * fy_dir)
            
            # Radii forces: encourage equal radii and growth if space allows
            mean_r = np.mean(radii)
            fr += -0.1 * (radii - mean_r)
            fr_inc = np.where(diff > 0.01, 0.1, np.where(diff < -0.01, -0.5, 0.0))
            np.add.at(fr, idx_i, fr_inc)
            np.add.at(fr, idx_j, fr_inc)
            
            # Update
            x_new = x + step_size * fx
            y_new = y + step_size * fy
            r_new = radii + step_size * 0.5 * fr
            
            # Clamp
            max_r = 0.5 * min(w, h)
            x_new = np.clip(x_new, radii, w - radii)
            y_new = np.clip(y_new, radii, h - radii)
            r_new = np.clip(r_new, 1e-4, max_r)
            
            # Build new state
            new_state = best_state.copy()
            new_state[1::3] = x_new
            new_state[2::3] = y_new
            new_state[3::3] = r_new
            
            new_score, _ = evaluate(new_state)
            
            if new_score > best_score:
                best_state = new_state
                best_score = new_score
                patience = 0
                step_size = min(0.01, step_size * 1.02)
            else:
                patience += 1
                step_size = max(0.001, step_size * 0.95)
            
            if patience > max_patience:
                break
        
        return best_state

    def local_optimize(state, time_limit):
        """
        Run a fast local optimization using pairwise repulsion and attraction.
        """
        best_state = state.copy()
        best_score, _ = evaluate(best_state)
        
        idx_i, idx_j = np.triu_indices(N, k=1)
        n_pairs = len(idx_i)
        
        step_size = 0.01
        patience = 0
        max_patience = 500
        
        for it in range(20000):
            if time.time() > time_limit - 1:
                break
            
            t = best_state[0]
            w, h = get_rect_dims(t)
            
            x = best_state[1::3].copy()
            y = best_state[2::3].copy()
            radii = best_state[3::3].copy()
            
            fx = np.zeros(N)
            fy = np.zeros(N)
            fr = np.zeros(N)
            
            # Boundary forces
            viol_l = np.maximum(0, radii - x)
            viol_r = np.maximum(0, x + radii - w)
            viol_b = np.maximum(0, radii - y)
            viol_t = np.maximum(0, y + radii - h)
            fx += 20.0 * viol_l - 20.0 * viol_r
            fy += 20.0 * viol_b - 20.0 * viol_t
            
            # Pair forces - only push apart when overlapping, gentle pull when close
            dx = x[idx_i] - x[idx_j]
            dy = y[idx_i] - y[idx_j]
            dist_sq = dx*dx + dy*dy
            dist = np.sqrt(dist_sq + 1e-12)
            target = radii[idx_i] + radii[idx_j]
            diff = dist - target
            
            # Strong repulsion when overlapping, weak attraction when too far
            force_mag = np.where(diff < 0, 100.0 * diff, np.where(diff < 0.05, 5.0 * diff, 0.0))
            
            inv_dist = 1.0 / np.maximum(dist, 1e-10)
            fx_dir = dx * inv_dist
            fy_dir = dy * inv_dist
            
            np.add.at(fx, idx_i, force_mag * fx_dir)
            np.add.at(fy, idx_i, force_mag * fy_dir)
            np.add.at(fx, idx_j, -force_mag * fx_dir)
            np.add.at(fy, idx_j, -force_mag * fy_dir)
            
            # Radii: grow if space, shrink if tight
            mean_r = np.mean(radii)
            fr += -0.2 * (radii - mean_r)
            
            # Update with adaptive step
            x_new = x + step_size * fx
            y_new = y + step_size * fy
            r_new = radii + step_size * 0.3 * fr
            
            max_r = 0.5 * min(w, h)
            x_new = np.clip(x_new, radii, w - radii)
            y_new = np.clip(y_new, radii, h - radii)
            r_new = np.clip(r_new, 1e-4, max_r)
            
            new_state = best_state.copy()
            new_state[1::3] = x_new
            new_state[2::3] = y_new
            new_state[3::3] = r_new
            
            new_score, _ = evaluate(new_state)
            
            if new_score > best_score:
                best_state = new_state
                best_score = new_score
                patience = 0
                step_size = min(0.02, step_size * 1.01)
            else:
                patience += 1
                step_size = max(0.002, step_size * 0.97)
            
            if patience > max_patience:
                break
        
        return best_state, best_score

    def multi_phase_polish(state, time_limit):
        """
        Multi-phase polish with increasing beta for tighter convergence.
        """
        current_state = state
        phases = [1.0, 0.5, 0.25, 0.1]
        time_per_phase = (time_limit - time.time()) / len(phases)
        
        for phase_idx, beta_scale in enumerate(phases):
            phase_deadline = time.time() + time_per_phase
            if time.time() >= time_limit - 2:
                break
            polished = polish_gradient(current_state, phase_deadline, max_iter=3000, beta_scale=beta_scale)
            pol_score, _ = evaluate(polished)
            cur_score, _ = evaluate(current_state)
            if pol_score > cur_score:
                current_state = polished
        
        return current_state

    # Main algorithm
    best_state = None
    best_score = -np.inf
    
    # Multiple restarts with different aspect ratios and initializations
    num_restarts = 20
    
    for restart in range(num_restarts):
        if time.time() > deadline - 25:
            break
        
        # Vary aspect ratio for diversity
        t_init = 0.5 + random.uniform(-0.25, 0.25)
        
        # Choose initialization strategy based on restart index
        init_strategy = restart % 3
        if init_strategy == 0:
            state = make_initial_hexagonal(t_init, jitter=0.02)
        elif init_strategy == 1:
            state = make_initial_grid(t_init, jitter=0.02)
        else:
            state = make_initial_random(t_init)
        
        # Time budget for this restart
        remaining = deadline - 25 - time.time()
        if remaining < 8:
            break
        
        # Allocate time for local opt and polish
        local_time = max(3, remaining * 0.30)
        local_deadline = time.time() + local_time
        
        state, score = local_optimize(state, local_deadline)
        
        # Force-directed relaxation phase
        if time.time() < deadline - 18:
            force_deadline = time.time() + min(5, (deadline - 18 - time.time()) * 0.3)
            force_state = force_relax(state, force_deadline)
            force_score, _ = evaluate(force_state)
            if force_score > score:
                state = force_state
                score = force_score
        
        # Multi-phase gradient polish
        if time.time() < deadline - 12:
            polish_deadline = time.time() + min(10, (deadline - 12 - time.time()) * 0.7)
            polished = multi_phase_polish(state, polish_deadline)
            pol_score, _ = evaluate(polished)
            if pol_score > score:
                state = polished
                score = pol_score
        
        if score > best_score:
            best_score = score
            best_state = state.copy()
    
    # Final high-precision polish on the best solution
    if best_state is not None and time.time() < deadline - 8:
        # Force relaxation first to potentially find better topology
        if time.time() < deadline - 10:
            force_final = force_relax(best_state, time.time() + 4)
            ff_score, _ = evaluate(force_final)
            if ff_score > best_score:
                best_state = force_final
                best_score = ff_score
        
        # Multi-phase final polish with tighter tolerance
        if time.time() < deadline - 5:
            final_polished = multi_phase_polish(best_state, deadline - 3)
            fp_score, _ = evaluate(final_polished)
            if fp_score > best_score:
                best_state = final_polished
                best_score = fp_score
    
    # Final validity check and repair if needed
    if best_state is None:
        best_state = make_initial_hexagonal(0.5)
    elif not check_valid(best_state):
        # Try to repair by shrinking radii and re-polishing
        repair = best_state.copy()
        repaired = False
        for shrink in range(50):
            if check_valid(repair):
                repaired = True
                break
            for i in range(N):
                repair[1 + 3*i + 2] *= 0.96
            # Quick reposition
            t_r = repair[0]
            w_r, h_r = get_rect_dims(t_r)
            for i in range(N):
                r_r = repair[1 + 3*i + 2]
                x_r = repair[1 + 3*i]
                y_r = repair[1 + 3*i + 1]
                x_r = max(r_r, min(w_r - r_r, x_r))
                y_r = max(r_r, min(h_r - r_r, y_r))
                repair[1 + 3*i] = x_r
                repair[1 + 3*i + 1] = y_r
        if repaired:
            best_state = repair
        else:
            best_state = make_initial_hexagonal(0.5)
    
    # Extract result
    result = np.zeros((N, 3))
    for i in range(N):
        result[i, 0] = best_state[1 + 3*i]
        result[i, 1] = best_state[1 + 3*i+1]
        result[i, 2] = best_state[1 + 3*i+2]
    
    # Final clamp to ensure validity
    t_final = best_state[0]
    w_final, h_final = get_rect_dims(t_final)
    for i in range(N):
        x = result[i, 0]
        y = result[i, 1]
        r = result[i, 2]
        
        if x - r < 0:
            x = r
        if x + r > w_final:
            x = w_final - r
        result[i, 0] = max(0, min(w_final, x))
        
        if y - r < 0:
            y = r
        if y + r > h_final:
            y = h_final - r
        result[i, 1] = max(0, min(h_final, y))
        
        max_r = 0.5 * min(w_final, h_final)
        result[i, 2] = max(1e-6, min(max_r, r))
    
    # Final validity verification - if invalid, use safe fallback
    if not check_valid(best_state):
        # Use the simplest valid configuration
        t_safe = 0.5
        w_safe, h_safe = get_rect_dims(t_safe)
        result = np.zeros((N, 3))
        rows = 5
        cols = 5
        for i in range(N):
            row = i // cols
            col = i % cols
            result[i, 0] = (col + 0.5) / cols * w_safe
            result[i, 1] = (row + 0.5) / rows * h_safe
            result[i, 2] = 0.5 * min(w_safe / cols, h_safe / rows) * 0.9
    
    return result

# EVOLVE-BLOCK-END
