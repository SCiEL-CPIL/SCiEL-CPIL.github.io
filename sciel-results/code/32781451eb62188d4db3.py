"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 21

def _calculate_penalty_and_score(w, positions, radii):
    h = 2.0 - w
    # Calculate sum of radii
    score = np.sum(radii)
    
    # Calculate penalty for constraint violations
    penalty = 0.0
    
    # Boundary violations
    # For each circle, check distance to walls
    # dist_left = x, dist_right = w-x, dist_bottom = y, dist_top = h-y
    # Violation if r > dist
    dists = np.stack([
        positions[:, 0],
        w - positions[:, 0],
        positions[:, 1],
        h - positions[:, 1]
    ], axis=1)
    
    # We want r_i <= min(dists_i).
    # Violation = max(0, r_i - min(dists_i))
    min_dists = np.min(dists, axis=1)
    boundary_violations = np.maximum(0, radii - min_dists)
    penalty += np.sum(boundary_violations) ** 2 # Squared penalty for smoothness
    
    # Overlap violations
    # For each pair (i, j), dist_ij >= r_i + r_j
    # dist_ij = sqrt((xi-xj)^2 + (yi-yj)^2)
    # Violation = max(0, r_i + r_j - dist_ij)
    
    # Vectorized distance calculation
    # positions is (N, 2)
    # diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :] # (N, N, 2)
    # dists_matrix = np.linalg.norm(diff, axis=2)
    
    # To save memory/time, we can do it in chunks or just use loops if N is small (21 is small)
    # N=21 is small enough for O(N^2) in Python if optimized, but vectorization is faster.
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists_matrix = np.linalg.norm(diff, axis=2)
    
    # radii sum matrix
    r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    
    # Only consider upper triangle to avoid double counting and self-distance
    # dists_matrix[i, i] is 0, r_sum[i, i] is 2*r_i, so violation is 2*r_i. We should mask diagonal.
    mask = ~np.eye(N, dtype=bool)
    
    violations = np.maximum(0, r_sum - dists_matrix)
    violations = violations * mask
    penalty += np.sum(violations) ** 2
    
    # Total energy to minimize: Penalty - Score (or Penalty - lambda * Score)
    # We want to maximize Score while keeping Penalty 0.
    # Let's return (Score, Penalty) and handle logic outside or define a combined cost.
    return score, penalty

def _simulate_annealing(w_init, max_iters=5000, T_init=1.0, T_min=1e-4, alpha=0.999):
    """
    Runs Simulated Annealing for a fixed initial width w_init.
    Returns best (w, positions, radii, score).
    """
    h_init = 2.0 - w_init
    
    # Initialize positions in a grid
    # Estimate grid size
    cols = int(np.sqrt(N * w_init / h_init)) + 1
    rows = (N + cols - 1) // cols
    
    positions = np.zeros((N, 2))
    for i in range(N):
        row = i // cols
        col = i % cols
        positions[i, 0] = (col + 0.5) * w_init / cols
        positions[i, 1] = (row + 0.5) * h_init / rows
        
    # Initial radii: small
    radii = np.full(N, 0.1)
    
    w = w_init
    
    best_score = -np.inf
    best_state = (w, positions.copy(), radii.copy())
    
    current_score, current_penalty = _calculate_penalty_and_score(w, positions, radii)
    # Cost function: We want to minimize Penalty and maximize Score.
    # Let's define Cost = Penalty - Score. We want to minimize Cost.
    current_cost = current_penalty - current_score
    
    T = T_init
    
    for it in range(max_iters):
        T *= alpha
        if T < T_min:
            break
            
        # Generate perturbation
        r = np.random.rand()
        
        if r < 0.1:
            # Perturb width
            dw = np.random.normal(0, 0.05)
            w_new = w + dw
            if 0.2 < w_new < 1.8: # Keep aspect ratio reasonable
                w = w_new
        elif r < 0.6:
            # Perturb position of a random circle
            i = np.random.randint(N)
            positions[i, 0] += np.random.normal(0, 0.05)
            positions[i, 1] += np.random.normal(0, 0.05)
        else:
            # Perturb radius of a random circle
            i = np.random.randint(N)
            radii[i] += np.random.normal(0, 0.02)
            if radii[i] < 0: radii[i] = 0
            if radii[i] > 1.0: radii[i] = 1.0 # Max possible is < 1.0
            
        # Ensure positions are somewhat inside (soft constraint handled by penalty, but avoid huge values)
        positions[:, 0] = np.clip(positions[:, 0], -1, 3)
        positions[:, 1] = np.clip(positions[:, 1], -1, 3)
        
        new_score, new_penalty = _calculate_penalty_and_score(w, positions, radii)
        new_cost = new_penalty - new_score
        
        delta_cost = new_cost - current_cost
        
        # Metropolis criterion
        if delta_cost < 0 or np.random.rand() < np.exp(-delta_cost / max(T, 1e-10)):
            current_cost = new_cost
            current_score = new_score
            current_penalty = new_penalty
            
            if current_penalty < 1e-6 and current_score > best_score:
                best_score = current_score
                best_state = (w, positions.copy(), radii.copy())
        
    return best_state, best_score

def construct():
    best_overall_score = -np.inf
    best_overall_state = None
    
    # Try a few different initial widths
    # Optimal aspect ratio for packing is often near 1.0 (square) or specific ratios.
    # Perimeter 4 => Square is 1x1.
    # Let's try w in [0.8, 1.2]
    
    start_time = time.time()
    time_limit = 110.0 # Leave 10s for overhead
    
    # Use multiple restarts
    num_restarts = 5
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Vary initial width
        w_init = 1.0 + np.random.uniform(-0.2, 0.2)
        
        state, score = _simulate_annealing(w_init, max_iters=2000)
        
        if score > best_overall_score:
            best_overall_score = score
            best_overall_state = state
    
    if best_overall_state is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    w, positions, radii = best_overall_state
    
    # Final cleanup: ensure strict validity
    # If penalty is not zero, we might need to shrink radii slightly to ensure validity
    # The evaluator checks for disjoint interiors and bounding box.
    # If our SA left us with tiny violations, we must fix them.
    
    h = 2.0 - w
    
    # Shrink radii to ensure no overlap and inside bounds
    # This is a conservative step
    valid = True
    # We can try to iteratively shrink any violating radii
    for _ in range(10):
        for i in range(N):
            # Check bounds
            max_r_bound = min(positions[i, 0], w - positions[i, 0], positions[i, 1], h - positions[i, 1])
            if radii[i] > max_r_bound:
                radii[i] = max(0, max_r_bound - 1e-6)
                
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sqrt(np.sum((positions[i] - positions[j])**2))
                required = radii[i] + radii[j]
                if dist < required:
                    # Shrink both slightly
                    overlap = required - dist
                    # Distribute shrinkage
                    shrink_i = overlap / 2.0 + 1e-6
                    shrink_j = overlap / 2.0 + 1e-6
                    radii[i] = max(0, radii[i] - shrink_i)
                    radii[j] = max(0, radii[j] - shrink_j)
                    
    # Final check
    for i in range(N):
        if positions[i, 0] < 0 or positions[i, 0] > w or positions[i, 1] < 0 or positions[i, 1] > h:
            return np.zeros((N, 3)) # Invalid, return empty or fallback
        if radii[i] < 0:
            return np.zeros((N, 3))
            
        max_r_bound = min(positions[i, 0], w - positions[i, 0], positions[i, 1], h - positions[i, 1])
        if radii[i] > max_r_bound + 1e-9:
             return np.zeros((N, 3))
             
    for i in range(N):
        for j in range(i+1, N):
            dist = np.sqrt(np.sum((positions[i] - positions[j])**2))
            if dist < radii[i] + radii[j] - 1e-9:
                return np.zeros((N, 3))
                
    result = np.column_stack([positions, radii])
    return result

# EVOLVE-BLOCK-END
