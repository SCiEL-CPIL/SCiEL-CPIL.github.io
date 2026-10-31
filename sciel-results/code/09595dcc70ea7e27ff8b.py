import numpy as np
import time

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin for 150s limit
    
    # We will use a Multi-Start Local Search strategy.
    # Instead of one long SA run, we run many short, aggressive local optimizations
    # on different initializations.
    
    # Precompute pair indices for vectorized distance checks
    i_idx, j_idx = np.triu_indices(N, k=1)
    pairs = np.column_stack([i_idx, j_idx])
    num_pairs = len(pairs)
    
    def evaluate_and_project(state):
        """
        Takes a state (N, 3), ensures it's valid by scaling down radii if necessary,
        and returns the valid state and its score.
        """
        x = state[:, 0]
        y = state[:, 1]
        r = state[:, 2].copy()
        
        # 1. Boundary constraint: r <= min(x, 1-x, y, 1-y)
        max_r_bound = np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y))
        # If any radius exceeds boundary, scale down globally to be safe
        if np.any(r > max_r_bound):
            # Find the worst violation ratio
            # We want r_i * scale <= max_r_bound_i for all i
            # scale <= max_r_bound_i / r_i
            # scale = min(max_r_bound_i / r_i)
            ratios = max_r_bound / np.maximum(r, 1e-9)
            scale = np.min(ratios)
            r *= scale
        
        # 2. Pairwise constraint: dist(i,j) >= r_i + r_j
        # We need to ensure this. A simple global scaling might break this if positions are bad,
        # but usually, we start with valid positions. 
        # For robustness, we do a few iterations of "shrink to fit" if overlaps exist.
        
        for _ in range(10):
            dx = x[i_idx] - x[j_idx]
            dy = y[i_idx] - y[j_idx]
            dist = np.sqrt(dx*dx + dy*dy)
            sum_r = r[i_idx] + r[j_idx]
            
            # Check for overlap
            violations = sum_r - dist
            if np.max(violations) < 1e-9:
                break
            
            # Simple repair: Scale down all radii by a factor
            # This is conservative but ensures validity.
            # A better repair would move centers, but that's complex.
            # Let's try a slight shrink.
            r *= 0.95
            
            # Re-check bounds after shrink
            if np.any(r > max_r_bound):
                ratios = max_r_bound / np.maximum(r, 1e-9)
                r *= np.min(ratios)
        
        # Final state
        valid_state = np.column_stack([x, y, r])
        score = np.sum(r)
        return valid_state, score

    def local_optimize(state, time_limit):
        """
        Aggressive local search for a single candidate.
        Strategy: 
        1. Try to grow all radii by a small epsilon.
        2. If invalid, identify "worst" overlaps and push circles apart.
        3. Repeat until no improvement or time limit.
        """
        current_state, current_score = evaluate_and_project(state)
        best_state = current_state.copy()
        best_score = current_score
        
        # Parameters for local search
        grow_step = 0.005
        push_step = 0.01
        no_improve_count = 0
        max_no_improve = 50
        
        while time.time() < time_limit and no_improve_count < max_no_improve:
            # 1. Attempt to grow
            new_state = current_state.copy()
            new_state[:, 2] += grow_step
            
            # Check validity of new_state
            x = new_state[:, 0]
            y = new_state[:, 1]
            r = new_state[:, 2]
            
            # Boundary check
            max_r_bound = np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y))
            if np.any(r > max_r_bound + 1e-9):
                # Boundary violation. Shrink to fit boundary first.
                ratios = max_r_bound / np.maximum(r, 1e-9)
                r *= np.min(ratios)
                new_state[:, 2] = r
            
            # Pairwise check
            dx = x[i_idx] - x[j_idx]
            dy = y[i_idx] - y[j_idx]
            dist = np.sqrt(dx*dx + dy*dy)
            sum_r = r[i_idx] + r[j_idx]
            violations = sum_r - dist
            
            if np.max(violations) < 1e-9:
                # Valid! Accept the growth.
                current_state = new_state
                current_score = np.sum(r)
                if current_score > best_score:
                    best_state = current_state.copy()
                    best_score = current_score
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                # Invalid. We need to resolve overlaps.
                # Identify the most violated pair
                max_viol_idx = np.argmax(violations)
                i, j = i_idx[max_viol_idx], j_idx[max_viol_idx]
                
                # Push i and j apart
                # Vector from j to i
                vec = current_state[i, :2] - current_state[j, :2]
                norm = np.linalg.norm(vec)
                if norm > 1e-9:
                    vec = vec / norm
                else:
                    vec = np.array([1.0, 0.0])
                
                # Move i in direction of vec, j in opposite
                # Move amount proportional to violation
                move_amt = violations[max_viol_idx] * 0.5 + 0.001
                
                new_state[i, 0] += vec[0] * move_amt
                new_state[i, 1] += vec[1] * move_amt
                new_state[j, 0] -= vec[0] * move_amt
                new_state[j, 1] -= vec[1] * move_amt
                
                # Ensure they stay in bounds
                # Simple projection to [0,1]
                new_state[i, 0] = np.clip(new_state[i, 0], 0, 1)
                new_state[i, 1] = np.clip(new_state[i, 1], 0, 1)
                new_state[j, 0] = np.clip(new_state[j, 0], 0, 1)
                new_state[j, 1] = np.clip(new_state[j, 1], 0, 1)
                
                # Re-evaluate validity of this pushed state (without growing yet)
                # If this push fixes the overlap, we can try to grow again next iteration.
                # If not, we might need to shrink radii.
                
                # For simplicity, accept the push if it reduces the total violation,
                # otherwise shrink the larger of the two circles slightly.
                
                # Calculate new violations for this specific pair
                dx_new = new_state[i, 0] - new_state[j, 0]
                dy_new = new_state[i, 1] - new_state[j, 1]
                dist_new = np.sqrt(dx_new**2 + dy_new**2)
                sum_r_new = r[i] + r[j]
                
                if dist_new >= sum_r_new:
                    # Push worked, accept position change
                    current_state = new_state
                    no_improve_count += 1 # We didn't grow, so count as no improve
                else:
                    # Push wasn't enough, shrink the larger circle
                    if r[i] > r[j]:
                        new_state[i, 2] *= 0.95
                    else:
                        new_state[j, 2] *= 0.95
                    current_state = new_state
                    no_improve_count += 1
        
        return best_state, best_score

    # Multi-Start Loop
    best_global_state = None
    best_global_score = 0.0
    
    # Generate diverse initializations
    # 1. Grid-based
    # 2. Random
    # 3. Ring-based
    
    num_starts = 20
    time_per_start = (deadline - start_time) / num_starts
    # Ensure minimum time per start
    time_per_start = max(time_per_start, 2.0)
    
    for k in range(num_starts):
        if time.time() > deadline - 5.0:
            break
            
        start_k = time.time()
        end_k = min(start_k + time_per_start, deadline - 5.0)
        
        # Generate initial state
        if k % 3 == 0:
            # Grid init
            side = int(np.ceil(np.sqrt(N)))
            cells = []
            for i in range(side):
                for j in range(side):
                    if len(cells) < N:
                        cells.append([ (i+0.5)/side, (j+0.5)/side, 0.1 ])
            state = np.array(cells[:N])
        elif k % 3 == 1:
            # Random init
            state = np.random.rand(N, 3)
            state[:, 0] *= 0.8 + 0.1 # [0.1, 0.9]
            state[:, 1] *= 0.8 + 0.1
            state[:, 2] = 0.05
        else:
            # Ring init
            pts = np.zeros((N, 2))
            pts[0] = [0.5, 0.5]
            angles1 = np.linspace(0, 2*np.pi, 8, endpoint=False)
            ring1 = 0.5 + 0.2 * np.column_stack([np.cos(angles1), np.sin(angles1)])
            pts[1:9] = ring1
            angles2 = np.linspace(0, 2*np.pi, 17, endpoint=False)
            ring2 = 0.5 + 0.4 * np.column_stack([np.cos(angles2), np.sin(angles2)])
            pts[9:] = ring2
            state = np.hstack([pts, np.full((N,1), 0.05)])
        
        # Add small noise to break symmetry
        state += np.random.normal(0, 0.01, state.shape)
        
        # Run local optimization
        candidate_state, candidate_score = local_optimize(state, end_k)
        
        if candidate_score > best_global_score:
            best_global_score = candidate_score
            best_global_state = candidate_state.copy()
    
    # If no valid state found (shouldn't happen), return a trivial valid one
    if best_global_state is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    cells.append([ (i+0.5)/side, (j+0.5)/side, 0.1 ])
        best_global_state = np.array(cells[:N])
        
    # Final safety projection
    final_state, _ = evaluate_and_project(best_global_state)
    return final_state

# EVOLVE-BLOCK-END
