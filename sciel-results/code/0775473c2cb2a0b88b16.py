"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit (10s buffer)
    
    # Rectangle perimeter is 4, so 2*(w+h) = 4 => w+h = 2.
    # Let w = 2 * t, h = 2 * (1-t), where t in (0, 1).
    
    def get_rect_dims(t):
        t = np.clip(t, 0.1, 0.9)
        w = 2.0 * t
        h = 2.0 * (1.0 - t)
        return w, h

    def evaluate(state):
        """
        Returns a score (higher is better).
        Score = sum(radii) - penalty
        """
        t = state[0]
        circles = state[1:]
        
        w, h = get_rect_dims(t)
        
        sum_r = 0.0
        penalty = 0.0
        
        coords = []
        for i in range(N):
            xi = circles[3*i]
            yi = circles[3*i+1]
            ri = circles[3*i+2]
            if ri < 1e-5:
                ri = 1e-5
            sum_r += ri
            coords.append((xi, yi, ri))
            
            # Boundaries
            if xi < ri:
                penalty += (ri - xi) ** 2 * 500
            if xi > w - ri:
                penalty += (xi - (w - ri)) ** 2 * 500
            if yi < ri:
                penalty += (ri - yi) ** 2 * 500
            if yi > h - ri:
                penalty += (yi - (h - ri)) ** 2 * 500
        
        # Overlaps
        for i in range(N):
            xi, yi, ri = coords[i]
            for j in range(i+1, N):
                xj, yj, rj = coords[j]
                dist_sq = (xi - xj)**2 + (yi - yj)**2
                min_dist = ri + rj
                if dist_sq < min_dist * min_dist:
                    dist = np.sqrt(dist_sq)
                    overlap = min_dist - dist
                    penalty += overlap ** 2 * 1000
        
        score = sum_r - penalty
        return score

    def check_valid(state, eps=1e-5):
        """
        Check if the state is strictly valid for the final return.
        """
        t = state[0]
        circles = state[1:]
        w, h = get_rect_dims(t)
        
        coords = []
        for i in range(N):
            xi = circles[3*i]
            yi = circles[3*i+1]
            ri = circles[3*i+2]
            if ri < 1e-6:
                return False
            coords.append((xi, yi, ri))
            
            # Strict boundary check with small epsilon
            if xi - ri < -eps or xi + ri > w + eps or yi - ri < -eps or yi + ri > h + eps:
                return False
        
        # Strict overlap check
        for i in range(N):
            xi, yi, ri = coords[i]
            for j in range(i+1, N):
                xj, yj, rj = coords[j]
                dist_sq = (xi - xj)**2 + (yi - yj)**2
                if dist_sq < (ri + rj - eps)**2:
                    return False
        return True

    def repair(state, max_iters=1000):
        """
        Attempt to repair an invalid state by shrinking radii and adjusting positions.
        Returns a repaired state or None if failed.
        """
        state = state.copy()
        for _ in range(max_iters):
            if check_valid(state):
                return state
            
            t = state[0]
            w, h = get_rect_dims(t)
            
            # Find worst violation and fix it
            # Simple strategy: shrink all radii by a small factor
            for i in range(N):
                r_idx = 1 + 3*i + 2
                state[r_idx] *= 0.995
                
            # Also nudge positions towards center if near boundary
            for i in range(N):
                x_idx = 1 + 3*i
                y_idx = 1 + 3*i + 1
                r_idx = 1 + 3*i + 2
                xi = state[x_idx]
                yi = state[y_idx]
                ri = state[r_idx]
                
                if xi < ri:
                    state[x_idx] = ri
                if xi > w - ri:
                    state[x_idx] = w - ri
                if yi < ri:
                    state[y_idx] = ri
                if yi > h - ri:
                    state[y_idx] = h - ri
        
        if check_valid(state):
            return state
        return None

    # Initialization
    def initialize():
        """
        Generate an initial state.
        """
        state = np.zeros(1 + 3*N)
        
        # Random aspect ratio
        state[0] = random.uniform(0.3, 0.7)
        w, h = get_rect_dims(state[0])
        
        # Place circles in a grid pattern initially
        side = int(np.ceil(np.sqrt(N)))
        idx = 0
        for i in range(side):
            for j in range(side):
                if idx >= N:
                    break
                gx = (i + 0.5) / side * w
                gy = (j + 0.5) / side * h
                r_init = 0.4 / side * min(w, h)
                
                state[1 + 3*idx] = gx + random.uniform(-0.01, 0.01)
                state[1 + 3*idx + 1] = gy + random.uniform(-0.01, 0.01)
                state[1 + 3*idx + 2] = r_init
                idx += 1
        
        return state

    # Simulated Annealing
    def simulated_annealing(initial_state, T_start, T_end, alpha, max_iter=None):
        """
        Run simulated annealing from initial_state.
        Returns best_state found.
        """
        state = initial_state.copy()
        best_state = state.copy()
        best_score = evaluate(state)
        
        T = T_start
        iter_count = 0
        
        while time.time() < deadline - 2:
            if max_iter is not None and iter_count >= max_iter:
                break
            iter_count += 1
            
            # Generate neighbor
            new_state = state.copy()
            
            # Randomly perturb a variable
            var_idx = random.randint(0, len(new_state)-1)
            
            if var_idx == 0:
                # Perturb aspect ratio
                new_state[0] += random.gauss(0, 0.02)
                if new_state[0] < 0.1 or new_state[0] > 0.9:
                    continue
            else:
                circle_idx = (var_idx - 1) // 3
                comp = (var_idx - 1) % 3
                
                if comp == 0: # x
                    new_state[var_idx] += random.gauss(0, 0.05)
                elif comp == 1: # y
                    new_state[var_idx] += random.gauss(0, 0.05)
                else: # r
                    new_state[var_idx] += random.gauss(0, 0.01)
                    if new_state[var_idx] < 1e-4:
                        new_state[var_idx] = 1e-4
                    w, h = get_rect_dims(new_state[0])
                    max_r = 0.5 * min(w, h)
                    if new_state[var_idx] > max_r:
                        new_state[var_idx] = max_r

            new_score = evaluate(new_state)
            
            # Acceptance criterion
            if new_score > best_score or random.random() < np.exp((new_score - best_score) / T):
                state = new_state
                if new_score > best_score:
                    best_state = new_state.copy()
                    best_score = new_score
            
            T *= alpha
            if T < T_end:
                T = T_end
        
        return best_state, best_score

    # Main loop: multiple restarts
    best_overall_state = None
    best_overall_score = -np.inf
    
    # First, try a few quick initializations to get a baseline
    for restart in range(5):
        if time.time() > deadline - 5:
            break
        
        init_state = initialize()
        state, score = simulated_annealing(
            init_state,
            T_start=0.5,
            T_end=1e-3,
            alpha=0.9995,
            max_iter=5000
        )
        
        if score > best_overall_score:
            best_overall_state = state
            best_overall_score = score
        
        # Try to repair if invalid
        if not check_valid(state):
            repaired = repair(state)
            if repaired is not None:
                rep_score = evaluate(repaired)
                if rep_score > best_overall_score:
                    best_overall_state = repaired
                    best_overall_score = rep_score

    # If we still have time, do a longer, more careful optimization from the best state
    if best_overall_state is not None and time.time() < deadline - 10:
        # Try to improve the best state further
        state, score = simulated_annealing(
            best_overall_state,
            T_start=0.1,
            T_end=1e-4,
            alpha=0.9999,
            max_iter=20000
        )
        
        if score > best_overall_score:
            best_overall_state = state
            best_overall_score = score
        
        # Final repair attempt
        if not check_valid(best_overall_state):
            repaired = repair(best_overall_state, max_iters=5000)
            if repaired is not None:
                rep_score = evaluate(repaired)
                if rep_score > best_overall_score:
                    best_overall_state = repaired
                    best_overall_score = rep_score

    # Fallback: if no valid state found, use a simple grid
    if best_overall_state is None or not check_valid(best_overall_state):
        # Simple grid fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                gx = (i + 0.5) / side
                gy = (j + 0.5) / side
                r = 0.5 / side - 1e-3
                cells.append([gx, gy, r])
        
        best_overall_state = np.zeros(1 + 3*N)
        best_overall_state[0] = 0.5
        for i in range(N):
            best_overall_state[1 + 3*i] = cells[i][0]
            best_overall_state[1 + 3*i + 1] = cells[i][1]
            best_overall_state[1 + 3*i + 2] = cells[i][2]

    # Extract final result
    result = np.zeros((N, 3))
    for i in range(N):
        result[i, 0] = best_overall_state[1 + 3*i]
        result[i, 1] = best_overall_state[1 + 3*i + 1]
        result[i, 2] = best_overall_state[1 + 3*i + 2]
        
    return result

# EVOLVE-BLOCK-END
