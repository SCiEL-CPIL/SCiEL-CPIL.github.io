"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
import time
import random

N = 21

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

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
        
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        for i in range(N):
            x[i] = state[1 + 3*i]
            y[i] = state[1 + 3*i+1]
            r[i] = state[1 + 3*i+2]
        
        # Enforce minimum radius
        min_r = 1e-6
        for i in range(N):
            if r[i] < min_r:
                r[i] = min_r
                penalty += 0.1
        
        sum_r = np.sum(r)
        
        # Boundary violations
        viol_left = np.maximum(0, r - x)
        viol_right = np.maximum(0, x + r - w)
        viol_bottom = np.maximum(0, r - y)
        viol_top = np.maximum(0, y + r - h)
        
        penalty += 500.0 * (np.sum(viol_left) + np.sum(viol_right) + np.sum(viol_bottom) + np.sum(viol_top))
        
        # Overlap violations
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                min_dist = r[i] + r[j]
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
        
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        for i in range(N):
            x[i] = state[1 + 3*i]
            y[i] = state[1 + 3*i+1]
            r[i] = state[1 + 3*i+2]
        
        if np.any(r < 1e-6):
            return False
        
        # Boundary check
        if np.any(x - r < -eps) or np.any(x + r > w + eps) or \
           np.any(y - r < -eps) or np.any(y + r > h + eps):
            return False
        
        # Overlap check
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                min_dist = r[i] + r[j] - eps
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
        
        # Estimate radius
        area = w * h
        r_est = np.sqrt(0.9069 * area / (N * np.pi))
        r_current = r_est * 0.95
        
        positions = []
        
        # Try to fit hexagonal packing
        for attempt in range(30):
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
            # Fallback to grid
            side = 5
            positions = []
            for i in range(side):
                for j in range(side):
                    if len(positions) >= N:
                        break
                    gx = (i + 0.5) / side * w
                    gy = (j + 0.5) / side * h
                    r = 0.5 / side * min(w, h) * 0.9
                    positions.append((gx, gy, r))
        
        for i in range(N):
            if i < len(positions):
                px, py, pr = positions[i]
                if jitter > 0:
                    px += random.uniform(-jitter, jitter)
                    py += random.uniform(-jitter, jitter)
                state[1 + 3*i] = px
                state[1 + 3*i + 1] = py
                state[1 + 3*i + 2] = pr
            else:
                state[1 + 3*i] = w/2
                state[1 + 3*i + 1] = h/2
                state[1 + 3*i + 2] = 0.01
        return state

    def local_optimize(state, max_iter=10000, time_limit=2.0):
        """
        Perform a local optimization using gradient descent / hill climbing.
        This refines a given state to find a better local optimum.
        """
        current_state = state.copy()
        current_score, _ = evaluate(current_state)
        
        # Simple hill climbing with random restarts within the local neighborhood
        # Or a more sophisticated approach: coordinate ascent / gradient steps
        
        # For now, let's do a simple simulated annealing step with a lower temperature
        # to polish the solution.
        
        T = 0.01
        alpha = 0.999
        step_size = 0.01
        
        start = time.time()
        for _ in range(max_iter):
            if time.time() - start > time_limit:
                break
            
            new_state = current_state.copy()
            var_idx = random.randint(0, len(new_state)-1)
            
            if var_idx == 0:
                new_state[0] += random.gauss(0, 0.005)
                if new_state[0] < 0.1 or new_state[0] > 0.9:
                    continue
            else:
                circle_idx = (var_idx - 1) // 3
                comp = (var_idx - 1) % 3
                
                if comp == 0: # x
                    new_state[var_idx] += random.gauss(0, step_size)
                elif comp == 1: # y
                    new_state[var_idx] += random.gauss(0, step_size)
                else: # r
                    new_state[var_idx] += random.gauss(0, step_size * 0.5)
                    if new_state[var_idx] < 1e-6:
                        new_state[var_idx] = 1e-6
                    w, h = get_rect_dims(new_state[0])
                    max_r = 0.5 * min(w, h)
                    if new_state[var_idx] > max_r:
                        new_state[var_idx] = max_r

            new_score, _ = evaluate(new_state)
            
            # Accept if better, or with probability exp((new-old)/T)
            if new_score > current_score or random.random() < np.exp((new_score - current_score) / T):
                current_state = new_state
                current_score = new_score
            
            T *= alpha
            if T < 1e-6:
                T = 1e-6
        
        return current_state, current_score

    # Main Search Loop
    # We will run multiple restarts with different initializations and aspect ratios.
    
    best_state = None
    best_score = -np.inf
    
    # List of aspect ratios to try
    # t = w/2. w+h=2 => h=2-w => h=2-2t.
    # Square: t=0.5, w=1, h=1.
    # Rectangle: t=0.6, w=1.2, h=0.8.
    # t=0.4, w=0.8, h=1.2.
    
    t_candidates = [0.5, 0.6, 0.4, 0.7, 0.3, 0.55, 0.45, 0.65, 0.35]
    
    # Shuffle candidates to randomize order
    random.shuffle(t_candidates)
    
    # We will allocate time to each restart
    # Roughly 10 restarts * 10s each = 100s.
    
    for t_init in t_candidates:
        if time.time() > deadline - 5:
            break
            
        # Initialize
        state = make_initial_hexagonal(t_init, jitter=0.01)
        
        # Local optimization
        state, score = local_optimize(state, max_iter=20000, time_limit=5.0)
        
        if score > best_score:
            best_score = score
            best_state = state.copy()
            
            # If we found a valid solution, we can potentially try to improve it further
            # but for now we just keep it.
            
        # Also try a grid initialization for this t
        # (Optional: skip if time is tight)
        if time.time() < deadline - 10:
            # Simple grid init
            w, h = get_rect_dims(t_init)
            state_grid = np.zeros(1 + 3*N)
            state_grid[0] = t_init
            side = 5
            idx = 0
            for i in range(side):
                for j in range(side):
                    if idx >= N:
                        break
                    gx = (i + 0.5) / side * w
                    gy = (j + 0.5) / side * h
                    r = 0.5 / side * min(w, h) * 0.9
                    state_grid[1 + 3*idx] = gx
                    state_grid[1 + 3*idx + 1] = gy
                    state_grid[1 + 3*idx + 2] = r
                    idx += 1
            
            state_grid, score_grid = local_optimize(state_grid, max_iter=20000, time_limit=5.0)
            if score_grid > best_score:
                best_score = score_grid
                best_state = state_grid.copy()

    # Final check and cleanup
    if best_state is None:
        # Fallback to a simple valid grid
        t_final = 0.5
        w, h = get_rect_dims(t_final)
        best_state = np.zeros(1 + 3*N)
        best_state[0] = t_final
        side = 5
        idx = 0
        for i in range(side):
            for j in range(side):
                if idx >= N:
                    break
                gx = (i + 0.5) / side * w
                gy = (j + 0.5) / side * h
                r = 0.5 / side * min(w, h) * 0.9
                best_state[1 + 3*idx] = gx
                best_state[1 + 3*idx + 1] = gy
                best_state[1 + 3*idx + 2] = r
                idx += 1
    else:
        # Repair if invalid
        if not check_valid(best_state):
            # Shrink radii
            repair_state = best_state.copy()
            for _ in range(100):
                if check_valid(repair_state):
                    break
                for i in range(N):
                    repair_state[1 + 3*i + 2] *= 0.99
            if check_valid(repair_state):
                best_state = repair_state
            else:
                # Last resort fallback
                t_final = 0.5
                w, h = get_rect_dims(t_final)
                best_state = np.zeros(1 + 3*N)
                best_state[0] = t_final
                side = 5
                idx = 0
                for i in range(side):
                    for j in range(side):
                        if idx >= N:
                            break
                        gx = (i + 0.5) / side * w
                        gy = (j + 0.5) / side * h
                        r = 0.5 / side * min(w, h) * 0.9
                        best_state[1 + 3*idx] = gx
                        best_state[1 + 3*idx + 1] = gy
                        best_state[1 + 3*idx + 2] = r
                        idx += 1

    # Extract final result
    t_final = best_state[0]
    
    result = np.zeros((N, 3))
    for i in range(N):
        result[i, 0] = best_state[1 + 3*i]
        result[i, 1] = best_state[1 + 3*i + 1]
        result[i, 2] = best_state[1 + 3*i + 2]
        
    return result

# EVOLVE-BLOCK-END
