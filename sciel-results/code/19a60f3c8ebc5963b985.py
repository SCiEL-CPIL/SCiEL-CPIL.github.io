import numpy as np
import time
from scipy.optimize import minimize

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin for 150s limit
    
    # Precompute pair indices for vectorized operations
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    def get_score(state):
        """
        Calculate the sum of radii for a valid state.
        If invalid, returns a penalized score.
        """
        x = state[:, 0]
        y = state[:, 1]
        r = state[:, 2]
        
        # Boundary check
        max_r_bound = np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y))
        if np.any(r > max_r_bound + 1e-9):
            # If boundary violated, we need to shrink to fit
            # This is a repair step for scoring
            ratios = max_r_bound / np.maximum(r, 1e-9)
            scale = np.min(ratios)
            r = r * scale
            
        # Pairwise check
        dx = x[i_idx] - x[j_idx]
        dy = y[i_idx] - y[j_idx]
        dist = np.sqrt(dx*dx + dy*dy)
        sum_r = r[i_idx] + r[j_idx]
        violations = sum_r - dist
        
        if np.max(violations) > 1e-9:
            # If overlaps exist, the state is technically invalid for the "maximize sum" goal
            # unless we shrink. For the optimizer, we use a penalty.
            # For this raw score function used in the final return, we ensure validity.
            # We perform a quick shrink to ensure the returned array is valid.
            for _ in range(20):
                dx = x[i_idx] - x[j_idx]
                dy = y[i_idx] - y[j_idx]
                dist = np.sqrt(dx*dx + dy*dy)
                sum_r = r[i_idx] + r[j_idx]
                violations = sum_r - dist
                if np.max(violations) < 1e-9:
                    break
                # Shrink the largest violators or global shrink
                # Global shrink is safer for validity
                r *= 0.95
                # Re-check bounds
                max_r_bound = np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y))
                if np.any(r > max_r_bound):
                    ratios = max_r_bound / np.maximum(r, 1e-9)
                    r *= np.min(ratios)
        
        return np.sum(r)

    def force_directed_solve(init_state, time_limit):
        """
        Uses a force-directed approach to separate circles and then grows them.
        1. Start with small radii.
        2. Apply repulsive forces to separate centers.
        3. Once separated, gradually increase radii while maintaining separation.
        """
        state = init_state.copy()
        # Start with small radii to allow movement
        state[:, 2] = np.full(N, 0.01)
        
        # Initialize velocities
        v = np.zeros((N, 2))
        
        # Parameters
        dt = 0.005
        damping = 0.9
        repulsion_k = 1.0
        attraction_k = 0.1 # Weak attraction to center to keep them in box
        growth_rate = 0.001
        
        current_time = time.time()
        steps = 0
        
        while current_time < time_limit:
            # Calculate forces
            forces = np.zeros((N, 2))
            
            # Repulsion between circles
            dx = state[i_idx, 0] - state[j_idx, 0]
            dy = state[i_idx, 1] - state[j_idx, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            # Avoid division by zero
            dist = np.maximum(dist, 1e-6)
            
            sum_r = state[i_idx, 2] + state[j_idx, 2]
            
            # Overlap amount
            overlap = sum_r - dist
            
            # Only apply force if overlapping or very close
            mask = overlap > -0.01 # Small buffer
            
            if np.any(mask):
                # Force is proportional to overlap
                f_mag = repulsion_k * overlap[mask]
                # Direction
                dir_x = dx[mask] / dist[mask]
                dir_y = dy[mask] / dist[mask]
                
                f_x = f_mag * dir_x
                f_y = f_mag * dir_y
                
                # Apply to i (positive) and j (negative)
                # We need to accumulate these into the forces array
                # Since we have vectorized indices, we use add.at
                np.add.at(forces, i_idx[mask], [f_x, f_y])
                np.add.at(forces, j_idx[mask], [-f_x, -f_y])
            
            # Boundary forces (soft walls)
            # Push away from walls if too close
            margin = 0.05
            # Left wall
            mask_l = state[:, 0] < margin
            forces[mask_l, 0] += attraction_k * (margin - state[mask_l, 0])
            # Right wall
            mask_r = state[:, 0] > 1 - margin
            forces[mask_r, 0] -= attraction_k * (state[mask_r, 0] - (1 - margin))
            # Bottom wall
            mask_b = state[:, 1] < margin
            forces[mask_b, 1] += attraction_k * (margin - state[mask_b, 1])
            # Top wall
            mask_t = state[:, 1] > 1 - margin
            forces[mask_t, 1] -= attraction_k * (state[mask_t, 1] - (1 - margin))
            
            # Attraction to center (very weak, to prevent drifting)
            forces[:, 0] += -0.01 * (state[:, 0] - 0.5)
            forces[:, 1] += -0.01 * (state[:, 1] - 0.5)
            
            # Update velocities
            v += forces * dt
            v *= damping
            
            # Update positions
            state[:, 0] += v[:, 0] * dt
            state[:, 1] += v[:, 1] * dt
            
            # Hard clamp to [0,1]
            state[:, 0] = np.clip(state[:, 0], 0, 1)
            state[:, 1] = np.clip(state[:, 1], 0, 1)
            
            # Try to grow radii
            # Check if valid
            dx = state[i_idx, 0] - state[j_idx, 0]
            dy = state[i_idx, 1] - state[j_idx, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            sum_r = state[i_idx, 2] + state[j_idx, 2]
            
            # Also check bounds
            max_r_bound = np.minimum(np.minimum(state[:, 0], 1-state[:, 0]), np.minimum(state[:, 1], 1-state[:, 1]))
            
            # Max possible growth
            min_dist_margin = np.min(dist - sum_r)
            min_bound_margin = np.min(max_r_bound - state[:, 2])
            
            possible_growth = min(min_dist_margin, min_bound_margin)
            
            if possible_growth > 0:
                # Grow by a fraction of the possible growth, capped by growth_rate
                grow_amt = min(growth_rate, possible_growth * 0.5)
                state[:, 2] += grow_amt
            else:
                # If not possible to grow, shrink slightly to allow movement?
                # Or just wait for positions to change
                pass
            
            steps += 1
            current_time = time.time()
            
            # Early stopping if no movement and no growth
            if steps % 100 == 0:
                if np.max(np.abs(v)) < 1e-6 and possible_growth < 1e-6:
                    break
                    
        return state

    def polish(state, time_limit):
        """
        L-BFGS-B polish to maximize sum of radii.
        Variables: x, y, r (3N)
        Objective: -sum(r)
        Constraints:
        1. x_i + r_i <= 1, x_i - r_i >= 0, etc. (Handled by bounds or penalty)
        2. dist(i,j) >= r_i + r_j (Inequality constraints)
        
        Since scipy minimize handles bounds easily but inequality constraints are tricky,
        we use a penalty method or SLSQP. Given N=26, SLSQP might be slow.
        Let's use a custom penalty gradient for L-BFGS-B or just SLSQP with tight bounds.
        Actually, let's use a simpler approach: 
        Fix positions from FD, optimize radii only? No, positions need to move too.
        
        Let's use SLSQP with a time limit check.
        """
        x0 = state.flatten()
        
        def objective(z):
            return -np.sum(z[2::3]) # Maximize sum of radii (indices 2, 5, 8...)
            
        def gradient(z):
            g = np.zeros_like(z)
            g[2::3] = -1.0
            return g
            
        # Bounds for x, y, r
        # x, y in [0, 1]
        # r in [0, 0.5] (max radius in unit square is 0.5)
        bounds = []
        for i in range(N):
            bounds.append((0, 1)) # x
            bounds.append((0, 1)) # y
            bounds.append((0, 0.5)) # r
            
        # Constraints: dist(i,j) >= r_i + r_j
        # This is non-convex and many constraints.
        # SLSQP might struggle. 
        # Alternative: Use the fact that FD gives a good local minimum.
        # Just do a few steps of "push apart and grow" with smaller steps.
        
        # Let's stick to a vectorized local search for the polish phase to be safe and fast.
        # It's more robust than SLSQP for this many constraints.
        
        best_state = state.copy()
        best_score = get_score(best_state)
        
        step_size = 0.001
        no_improve = 0
        max_no_improve = 200
        current_time = time.time()
        
        while current_time < time_limit and no_improve < max_no_improve:
            # Try to grow all radii
            new_state = best_state.copy()
            new_state[:, 2] += step_size
            
            # Check validity and repair
            x = new_state[:, 0]
            y = new_state[:, 1]
            r = new_state[:, 2]
            
            # Boundary
            max_r_bound = np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y))
            if np.any(r > max_r_bound):
                ratios = max_r_bound / np.maximum(r, 1e-9)
                r *= np.min(ratios)
                new_state[:, 2] = r
            
            # Pairwise
            dx = x[i_idx] - x[j_idx]
            dy = y[i_idx] - y[j_idx]
            dist = np.sqrt(dx*dx + dy*dy)
            sum_r = r[i_idx] + r[j_idx]
            violations = sum_r - dist
            
            if np.max(violations) < 1e-9:
                # Valid
                score = np.sum(r)
                if score > best_score:
                    best_state = new_state
                    best_score = score
                    no_improve = 0
                    # Try to increase step size slightly if stable?
                    # step_size = min(step_size * 1.1, 0.01)
                else:
                    no_improve += 1
                    step_size *= 0.9
            else:
                # Invalid. Push apart the worst violator.
                max_viol_idx = np.argmax(violations)
                i, j = i_idx[max_viol_idx], j_idx[max_viol_idx]
                
                vec = best_state[i, :2] - best_state[j, :2]
                norm = np.linalg.norm(vec)
                if norm > 1e-9:
                    vec = vec / norm
                else:
                    vec = np.array([1.0, 0.0])
                
                move_amt = violations[max_viol_idx] * 0.5 + 0.0005
                
                new_state[i, 0] += vec[0] * move_amt
                new_state[i, 1] += vec[1] * move_amt
                new_state[j, 0] -= vec[0] * move_amt
                new_state[j, 1] -= vec[1] * move_amt
                
                new_state[i, 0] = np.clip(new_state[i, 0], 0, 1)
                new_state[i, 1] = np.clip(new_state[i, 1], 0, 1)
                new_state[j, 0] = np.clip(new_state[j, 0], 0, 1)
                new_state[j, 1] = np.clip(new_state[j, 1], 0, 1)
                
                # Check if this push allows growth or at least maintains validity
                # Re-evaluate violations for this specific state
                dx = new_state[i, 0] - new_state[j, 0]
                dy = new_state[i, 1] - new_state[j, 1]
                dist_new = np.sqrt(dx*dx + dy*dy)
                sum_r_new = new_state[i, 2] + new_state[j, 2]
                
                if dist_new >= sum_r_new - 1e-9:
                    # Check full validity
                    x_c = new_state[:, 0]
                    y_c = new_state[:, 1]
                    r_c = new_state[:, 2]
                    max_r_bound_c = np.minimum(np.minimum(x_c, 1-x_c), np.minimum(y_c, 1-y_c))
                    if np.any(r_c > max_r_bound_c):
                         # Shrink to fit
                         ratios = max_r_bound_c / np.maximum(r_c, 1e-9)
                         r_c *= np.min(ratios)
                         new_state[:, 2] = r_c
                         
                    dx_all = x_c[i_idx] - x_c[j_idx]
                    dy_all = y_c[i_idx] - y_c[j_idx]
                    dist_all = np.sqrt(dx_all*dx_all + dy_all*dy_all)
                    sum_r_all = r_c[i_idx] + r_c[j_idx]
                    viol_all = sum_r_all - dist_all
                    
                    if np.max(viol_all) < 1e-9:
                         best_state = new_state
                         best_score = np.sum(r_c)
                         no_improve = 0
                    else:
                         no_improve += 1
                         step_size *= 0.9
                else:
                    no_improve += 1
                    step_size *= 0.9
                    
            current_time = time.time()
            
        return best_state, best_score

    # Multi-Start Loop
    best_global_state = None
    best_global_score = 0.0
    
    num_starts = 10
    time_per_start = (deadline - start_time) / num_starts
    time_per_start = max(time_per_start, 5.0)
    
    for k in range(num_starts):
        if time.time() > deadline - 5.0:
            break
            
        start_k = time.time()
        end_k = min(start_k + time_per_start, deadline - 5.0)
        
        # Generate initial state
        if k % 2 == 0:
            # Grid init
            side = int(np.ceil(np.sqrt(N)))
            cells = []
            for i in range(side):
                for j in range(side):
                    if len(cells) < N:
                        cells.append([ (i+0.5)/side, (j+0.5)/side, 0.01 ])
            state = np.array(cells[:N])
        else:
            # Random init
            state = np.random.rand(N, 3)
            state[:, 0] *= 0.8 + 0.1
            state[:, 1] *= 0.8 + 0.1
            state[:, 2] = 0.01
        
        # Add noise
        state += np.random.normal(0, 0.01, state.shape)
        
        # Phase 1: Force Directed
        fd_state = force_directed_solve(state, end_k - 2.0)
        
        # Phase 2: Polish
        candidate_state, candidate_score = polish(fd_state, end_k)
        
        if candidate_score > best_global_score:
            best_global_score = candidate_score
            best_global_state = candidate_state.copy()
            
    # Fallback
    if best_global_state is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    cells.append([ (i+0.5)/side, (j+0.5)/side, 0.01 ])
        best_global_state = np.array(cells[:N])
        
    # Final safety projection
    final_state, _ = evaluate_and_project(best_global_state)
    return final_state

def evaluate_and_project(state):
    """
    Takes a state (N, 3), ensures it's valid by scaling down radii if necessary,
    and returns the valid state and its score.
    """
    x = state[:, 0]
    y = state[:, 1]
    r = state[:, 2].copy()
    
    # 1. Boundary constraint
    max_r_bound = np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y))
    if np.any(r > max_r_bound):
        ratios = max_r_bound / np.maximum(r, 1e-9)
        scale = np.min(ratios)
        r *= scale
    
    # 2. Pairwise constraint
    i_idx, j_idx = np.triu_indices(N, k=1)
    for _ in range(20):
        dx = x[i_idx] - x[j_idx]
        dy = y[i_idx] - y[j_idx]
        dist = np.sqrt(dx*dx + dy*dy)
        sum_r = r[i_idx] + r[j_idx]
        violations = sum_r - dist
        if np.max(violations) < 1e-9:
            break
        r *= 0.95
        if np.any(r > max_r_bound):
            ratios = max_r_bound / np.maximum(r, 1e-9)
            r *= np.min(ratios)
            
    valid_state = np.column_stack([x, y, r])
    score = np.sum(r)
    return valid_state, score

# EVOLVE-BLOCK-END
