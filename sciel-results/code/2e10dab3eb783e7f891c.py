"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
import time
import random

N = 21

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Rectangle perimeter is 4, so 2*(w+h) = 4 => w+h = 2.
    # Let w = 2 * t, h = 2 * (1-t), where t in (0, 1).
    # We want to maximize sum of radii of 21 disjoint circles inside [0,w] x [0,h].
    
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
        
        coords = np.zeros((N, 3))
        for i in range(N):
            coords[i, 0] = state[1 + 3*i]
            coords[i, 1] = state[1 + 3*i+1]
            coords[i, 2] = state[1 + 3*i+2]
        
        radii = coords[:, 2]
        if np.any(radii < 1e-5):
            radii = np.maximum(radii, 1e-5)
            penalty += 0.1 * np.sum(radii[radii < 1e-5])
        
        sum_r = np.sum(radii)
        
        # Boundary violations
        x = coords[:, 0]
        y = coords[:, 1]
        
        viol_left = np.maximum(0, radii - x)
        viol_right = np.maximum(0, x + radii - w)
        viol_bottom = np.maximum(0, radii - y)
        viol_top = np.maximum(0, y + radii - h)
        
        penalty += 500.0 * (np.sum(viol_left) + np.sum(viol_right) + np.sum(viol_bottom) + np.sum(viol_top))
        
        # Overlap violations - vectorized using squared distances
        # For pairs (i,j) with i < j
        idx_i, idx_j = np.triu_indices(N, k=1)
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist_sq = dx*dx + dy*dy
        min_dist = radii[idx_i] + radii[idx_j]
        
        # Only compute sqrt where needed (where dist_sq < min_dist^2)
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
        
        coords = np.zeros((N, 3))
        for i in range(N):
            coords[i, 0] = state[1 + 3*i]
            coords[i, 1] = state[1 + 3*i+1]
            coords[i, 2] = state[1 + 3*i+2]
        
        radii = coords[:, 2]
        if np.any(radii < 1e-6):
            return False
        
        x = coords[:, 0]
        y = coords[:, 1]
        
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

    def make_initial():
        """
        Create an initial state using a grid pattern in a square rectangle (t=0.5).
        """
        t_init = 0.5
        w, h = get_rect_dims(t_init)
        
        state = np.zeros(1 + 3*N)
        state[0] = t_init
        
        # Use a 5x5 grid, fill 21 cells
        side = 5
        idx = 0
        for i in range(side):
            for j in range(side):
                if idx >= N:
                    break
                gx = (i + 0.5) / side * w
                gy = (j + 0.5) / side * h
                # Radius that fits in cell with small margin
                r = 0.5 / side * min(w, h) * 0.95
                state[1 + 3*idx] = gx
                state[1 + 3*idx + 1] = gy
                state[1 + 3*idx + 2] = r
                idx += 1
        
        return state

    def polish_gradient(state, time_limit, max_iter=2000):
        """
        Gradient-based polish using smooth penalty formulation with proper gradients.
        """
        from scipy.optimize import minimize
        
        # Precompute pair indices
        idx_i, idx_j = np.triu_indices(N, k=1)
        
        def objective_and_grad(s):
            t = s[0]
            w, h = get_rect_dims(t)
            
            x = s[1:1+3*N:3]
            y = s[2:2+3*N:3]
            r = s[3:3+3*N:3]
            
            # Negative sum of radii (we minimize)
            obj = -np.sum(r)
            grad = np.zeros_like(s)
            grad[3:3+3*N:3] = -1.0  # d(-sum_r)/dr_i = -1
            
            # Smooth boundary penalties using softplus: log(1+exp(x/eps))*eps
            eps_b = 1e-4
            def softplus(v):
                return eps_b * np.log1p(np.exp(v / eps_b))
            
            # Left: violation = r_i - x_i
            v_l = r - x
            obj += 1000.0 * np.sum(softplus(v_l))
            # d/dx_i = -1000 * sigmoid(v_l/eps_b), d/dr_i = 1000 * sigmoid(v_l/eps_b)
            sig_l = 1.0 / (1.0 + np.exp(-v_l / eps_b))
            grad[1:1+3*N:3] += -1000.0 * sig_l
            grad[3:3+3*N:3] += 1000.0 * sig_l
            
            # Right: violation = x_i + r_i - w
            v_r = x + r - w
            obj += 1000.0 * np.sum(softplus(v_r))
            sig_r = 1.0 / (1.0 + np.exp(-v_r / eps_b))
            grad[1:1+3*N:3] += 1000.0 * sig_r
            grad[3:3+3*N:3] += 1000.0 * sig_r
            
            # Bottom: violation = r_i - y_i
            v_b = r - y
            obj += 1000.0 * np.sum(softplus(v_b))
            sig_b = 1.0 / (1.0 + np.exp(-v_b / eps_b))
            grad[2:2+3*N:3] += -1000.0 * sig_b
            grad[3:3+3*N:3] += 1000.0 * sig_b
            
            # Top: violation = y_i + r_i - h
            v_t = y + r - h
            obj += 1000.0 * np.sum(softplus(v_t))
            sig_t = 1.0 / (1.0 + np.exp(-v_t / eps_b))
            grad[2:2+3*N:3] += 1000.0 * sig_t
            grad[3:3+3*N:3] += 1000.0 * sig_t
            
            # Overlap penalty using smooth approximation
            dx = x[idx_i] - x[idx_j]
            dy = y[idx_i] - y[idx_j]
            dist_sq = dx*dx + dy*dy
            min_dist = r[idx_i] + r[idx_j]
            
            # Smooth penalty: use max(0, min_dist - dist) with smoothing
            # Let g = min_dist - sqrt(dist_sq + eps^2) (smooth version of min_dist - dist)
            dist_smooth = np.sqrt(dist_sq + 1e-8)
            g = min_dist - dist_smooth
            
            # softplus(g) = log(1+exp(g/eps_o))*eps_o
            eps_o = 1e-4
            sp = eps_o * np.log1p(np.exp(g / eps_o))
            obj += 5000.0 * np.sum(sp)
            
            # d(sp)/dg = sigmoid(g/eps_o)
            sig_o = 1.0 / (1.0 + np.exp(-g / eps_o))
            
            # d(g)/d(dist_smooth) = -1
            # d(dist_smooth)/d(dist_sq) = 1/(2*dist_smooth)
            # d(dist_sq)/dx_i = -2*dx, d(dist_sq)/dx_j = 2*dx
            # d(min_dist)/dr_i = 1, d(min_dist)/dr_j = 1
            
            # d(obj)/dx_i = 5000 * sig_o * (-1) * (1/(2*dist_smooth)) * (-2*dx) = 5000 * sig_o * dx / dist_smooth
            # d(obj)/dx_j = 5000 * sig_o * (-1) * (1/(2*dist_smooth)) * (2*dx) = -5000 * sig_o * dx / dist_smooth
            # d(obj)/dr_i = 5000 * sig_o * 1
            # d(obj)/dr_j = 5000 * sig_o * 1
            
            coeff = 5000.0 * sig_o / dist_smooth
            
            np.add.at(grad[1:1+3*N:3], idx_i, coeff * dx)
            np.add.at(grad[1:1+3*N:3], idx_j, -coeff * dx)
            np.add.at(grad[2:2+3*N:3], idx_i, coeff * dy)
            np.add.at(grad[2:2+3*N:3], idx_j, -coeff * dy)
            np.add.at(grad[3:3+3*N:3], idx_i, 5000.0 * sig_o)
            np.add.at(grad[3:3+3*N:3], idx_j, 5000.0 * sig_o)
            
            return obj, grad
        
        def objective(s):
            return objective_and_grad(s)[0]
        
        def gradient(s):
            return objective_and_grad(s)[1]
        
        # Use scipy minimize with L-BFGS-B
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
            if result.fun < objective(state):
                return result.x
        except Exception:
            pass
        
        return state
    
    def local_optimize(state, time_limit):
        """
        Run simulated annealing to improve state.
        """
        best_state = state.copy()
        best_score, _ = evaluate(best_state)
        
        T_start = 0.3
        T_end = 1e-6
        # Use a slower cooling schedule
        num_steps_estimate = 5000
        alpha = (T_end / T_start) ** (1.0 / num_steps_estimate)
        
        T = T_start
        current_state = best_state.copy()
        current_score = best_score
        
        while time.time() < time_limit - 1:
            # Generate neighbor
            new_state = current_state.copy()
            var_idx = random.randint(0, len(new_state)-1)
            
            if var_idx == 0:
                # Perturb aspect ratio
                new_state[0] += random.gauss(0, 0.03)
                if new_state[0] < 0.05 or new_state[0] > 0.95:
                    T *= alpha
                    if T < T_end:
                        T = T_end
                    continue
            else:
                circle_idx = (var_idx - 1) // 3
                comp = (var_idx - 1) % 3
                
                # Scale perturbation by current temperature
                scale = 0.05 * (T / T_start) + 0.005
                
                if comp == 0:  # x
                    new_state[var_idx] += random.gauss(0, scale)
                elif comp == 1:  # y
                    new_state[var_idx] += random.gauss(0, scale)
                else:  # r
                    new_state[var_idx] += random.gauss(0, scale * 0.5)
                    if new_state[var_idx] < 1e-4:
                        new_state[var_idx] = 1e-4
                    w, h = get_rect_dims(new_state[0])
                    max_r = 0.5 * min(w, h)
                    if new_state[var_idx] > max_r:
                        new_state[var_idx] = max_r
            
            new_score, new_penalty = evaluate(new_state)
            
            # Acceptance criterion
            delta = new_score - current_score
            if delta > 0 or random.random() < np.exp(delta / T):
                current_state = new_state
                current_score = new_score
                if new_score > best_score:
                    best_state = new_state
                    best_score = new_score
            
            T *= alpha
            if T < T_end:
                T = T_end
        
        return best_state, best_score

    # Main algorithm
    best_state = None
    best_score = -np.inf
    
    # Fewer restarts with more thorough optimization each
    num_restarts = 4
    
    for restart in range(num_restarts):
        if time.time() > deadline - 40:
            break
        
        # Create initial state with some randomization
        if restart == 0:
            state = make_initial()
        else:
            state = make_initial()
            # Add random perturbation
            w, h = get_rect_dims(state[0])
            for i in range(N):
                state[1 + 3*i] += random.uniform(-0.08, 0.08) * min(w, h)
                state[1 + 3*i + 1] += random.uniform(-0.08, 0.08) * min(w, h)
                state[1 + 3*i + 2] *= random.uniform(0.7, 1.0)
            state[0] += random.uniform(-0.15, 0.15)
        
        # Time budget for this restart
        remaining = deadline - 40 - time.time()
        if remaining < 20:
            break
        
        # Allocate time for SA (leave room for polish)
        sa_time = max(10, remaining * 0.45)
        sa_deadline = time.time() + sa_time
        
        state, score = local_optimize(state, sa_deadline)
        
        # Gradient polish phase - run multiple times for better convergence
        if time.time() < deadline - 15:
            for polish_iter in range(4):
                if time.time() > deadline - 12:
                    break
                polish_deadline = time.time() + min(12, (deadline - 12 - time.time()) / (4 - polish_iter))
                polished = polish_gradient(state, polish_deadline)
                pol_score, pol_pen = evaluate(polished)
                if pol_score > score:
                    state = polished
                    score = pol_score
                else:
                    break
        
        if score > best_score:
            best_score = score
            best_state = state.copy()
        
        # If we found a valid solution, try to refine it
        if best_state is not None and check_valid(best_state):
            # Try to increase radii slightly and re-optimize
            refined = best_state.copy()
            for i in range(N):
                r_idx = 1 + 3*i + 2
                refined[r_idx] *= 1.02
            # Quick local optimization
            quick_deadline = min(time.time() + 8, deadline - 12)
            refined, ref_score = local_optimize(refined, quick_deadline)
            # Polish the refined solution
            if time.time() < deadline - 12:
                polished_ref = polish_gradient(refined, min(time.time() + 10, deadline - 8))
                pr_score, _ = evaluate(polished_ref)
                if pr_score > ref_score:
                    refined = polished_ref
                    ref_score = pr_score
            if ref_score > best_score and check_valid(refined):
                best_score = ref_score
                best_state = refined.copy()
    
    # Final high-precision polish on the best solution
    if best_state is not None and time.time() < deadline - 8:
        final_polished = polish_gradient(best_state, deadline - 4)
        fp_score, fp_pen = evaluate(final_polished)
        if fp_score > best_score:
            best_state = final_polished
            best_score = fp_score
        
        # One more polish pass if time allows
        if time.time() < deadline - 6:
            second_polish = polish_gradient(best_state, deadline - 2)
            sp_score, sp_pen = evaluate(second_polish)
            if sp_score > best_score:
                best_state = second_polish
                best_score = sp_score
    
    # Final validity check and repair if needed
    if best_state is None:
        # Fallback: return a simple valid grid
        best_state = make_initial()
    elif not check_valid(best_state):
        # Try to repair by shrinking radii
        repair = best_state.copy()
        for _ in range(100):
            if check_valid(repair):
                break
            for i in range(N):
                repair[1 + 3*i + 2] *= 0.97
        if check_valid(repair):
            best_state = repair
        else:
            # Last resort: use the initial grid
            best_state = make_initial()
    
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
        
        # Clamp x
        if x - r < 0:
            x = r
        if x + r > w_final:
            x = w_final - r
        result[i, 0] = max(0, min(w_final, x))
        
        # Clamp y
        if y - r < 0:
            y = r
        if y + r > h_final:
            y = h_final - r
        result[i, 1] = max(0, min(h_final, y))
        
        # Clamp radius
        max_r = 0.5 * min(w_final, h_final)
        result[i, 2] = max(1e-6, min(max_r, r))
    
    return result

# EVOLVE-BLOCK-END
