"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
import time
import random

N = 21

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    def get_rect_dims(t):
        t = np.clip(t, 0.05, 0.95)
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
        
        coords = np.zeros((N, 3))
        for i in range(N):
            coords[i, 0] = state[1 + 3*i]
            coords[i, 1] = state[1 + 3*i+1]
            coords[i, 2] = state[1 + 3*i+2]
        
        radii = coords[:, 2]
        if np.any(radii < 1e-5):
            radii = np.maximum(radii, 1e-5)
        
        sum_r = np.sum(radii)
        
        # Boundary violations
        x = coords[:, 0]
        y = coords[:, 1]
        
        viol_left = np.maximum(0, radii - x)
        viol_right = np.maximum(0, x + radii - w)
        viol_bottom = np.maximum(0, radii - y)
        viol_top = np.maximum(0, y + radii - h)
        
        penalty = 500.0 * (np.sum(viol_left) + np.sum(viol_right) + np.sum(viol_bottom) + np.sum(viol_top))
        
        # Overlap violations
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                min_dist = radii[i] + radii[j]
                if dist_sq < min_dist * min_dist:
                    dist = np.sqrt(dist_sq)
                    overlap = min_dist - dist
                    penalty += 1000.0 * overlap
        
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
    
    def make_initial(t_val=0.5):
        """
        Create an initial state using a grid pattern.
        """
        w, h = get_rect_dims(t_val)
        
        state = np.zeros(1 + 3*N)
        state[0] = t_val
        
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
                r = 0.5 / side * min(w, h) * 0.9
                state[1 + 3*idx] = gx
                state[1 + 3*idx + 1] = gy
                state[1 + 3*idx + 2] = r
                idx += 1
        
        return state
    
    def local_optimize(state, time_limit, T_start=0.3, T_end=1e-4, alpha=0.999):
        """
        Run simulated annealing to improve state.
        """
        best_state = state.copy()
        best_score, _ = evaluate(best_state)
        curr_state = state.copy()
        curr_score, _ = evaluate(curr_state)
        
        T = T_start
        
        while time.time() < time_limit:
            if time.time() > time_limit - 1:
                break
            
            # Generate neighbor
            new_state = curr_state.copy()
            var_idx = random.randint(0, len(new_state)-1)
            
            if var_idx == 0:
                # Perturb aspect ratio
                new_state[0] += random.gauss(0, 0.015)
                if new_state[0] < 0.05 or new_state[0] > 0.95:
                    T *= alpha
                    if T < T_end:
                        T = T_end
                    continue
            else:
                circle_idx = (var_idx - 1) // 3
                comp = (var_idx - 1) % 3
                
                # Scale perturbation by current temperature
                scale = max(0.005, T / T_start * 0.08)
                
                if comp == 0:  # x
                    new_state[var_idx] += random.gauss(0, scale)
                elif comp == 1:  # y
                    new_state[var_idx] += random.gauss(0, scale)
                else:  # r
                    new_state[var_idx] += random.gauss(0, scale * 0.25)
                    if new_state[var_idx] < 1e-4:
                        new_state[var_idx] = 1e-4
                    w, h = get_rect_dims(new_state[0])
                    max_r = 0.5 * min(w, h)
                    if new_state[var_idx] > max_r:
                        new_state[var_idx] = max_r
            
            new_score, new_penalty = evaluate(new_state)
            
            # Acceptance criterion
            if new_score > curr_score or random.random() < np.exp((new_score - curr_score) / max(T, 1e-10)):
                curr_state = new_state
                curr_score = new_score
                
                if curr_score > best_score:
                    best_score = curr_score
                    best_state = curr_state.copy()
            
            T *= alpha
            if T < T_end:
                T = T_end
        
        return best_state, best_score
    
    # Main algorithm
    best_state = None
    best_score = -np.inf
    
    # Multiple restarts with different initial configurations
    num_restarts = 8
    
    for restart in range(num_restarts):
        if time.time() > deadline - 20:
            break
        
        # Create initial state with some randomization
        if restart == 0:
            t_init = 0.5
            state = make_initial(t_init)
        elif restart == 1:
            t_init = 0.6
            state = make_initial(t_init)
        elif restart == 2:
            t_init = 0.4
            state = make_initial(t_init)
        elif restart == 3:
            t_init = 0.55
            state = make_initial(t_init)
        else:
            # Random aspect ratio
            t_init = random.uniform(0.35, 0.65)
            state = make_initial(t_init)
            # Add random perturbation
            w, h = get_rect_dims(state[0])
            for i in range(N):
                state[1 + 3*i] += random.uniform(-0.03, 0.03) * min(w, h)
                state[1 + 3*i + 1] += random.uniform(-0.03, 0.03) * min(w, h)
                state[1 + 3*i + 2] *= random.uniform(0.75, 0.95)
        
        # Time budget for this restart
        remaining = deadline - 20 - time.time()
        if remaining < 5:
            break
        
        # Allocate time for local optimization
        time_for_sa = max(3, remaining * (num_restarts - restart) / (num_restarts + 1))
        sa_deadline = time.time() + min(time_for_sa, remaining - 3)
        
        state, score = local_optimize(state, sa_deadline)
        
        if score > best_score:
            best_score = score
            best_state = state.copy()
        
        # If we found a valid solution, try to refine it
        if check_valid(best_state):
            # Try to increase radii slightly and re-optimize
            refined = best_state.copy()
            for i in range(N):
                r_idx = 1 + 3*i + 2
                # Increase radius by a small amount if it doesn't violate boundaries
                refined[r_idx] *= 1.03
            
            # Quick local optimization
            quick_deadline = min(time.time() + 8, deadline - 15)
            if quick_deadline > time.time() + 2:
                refined, ref_score = local_optimize(refined, quick_deadline)
                if ref_score > best_score and check_valid(refined):
                    best_score = ref_score
                    best_state = refined.copy()
    
    # Final validity check and repair if needed
    if best_state is None:
        # Fallback: return a simple valid grid
        best_state = make_initial(0.5)
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
            best_state = make_initial(0.5)
    
    # Extract result
    result = np.zeros((N, 3))
    for i in range(N):
        result[i, 0] = best_state[1 + 3*i]
        result[i, 1] = best_state[1 + 3*i+1]
        result[i, 2] = best_state[1 + 3*i+2]
    
    return result

# EVOLVE-BLOCK-END
