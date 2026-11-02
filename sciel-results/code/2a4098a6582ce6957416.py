"""Improved circle packing with better SA and final radius growth."""
import numpy as np
import time

N = 21

def _calculate_penalty_and_score(w, positions, radii):
    h = 2.0 - w
    score = np.sum(radii)
    
    penalty = 0.0
    
    # Boundary violations
    dists = np.stack([
        positions[:, 0],
        w - positions[:, 0],
        positions[:, 1],
        h - positions[:, 1]
    ], axis=1)
    
    min_dists = np.min(dists, axis=1)
    boundary_violations = np.maximum(0, radii - min_dists)
    penalty += np.sum(boundary_violations) ** 2
    
    # Overlap violations
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists_matrix = np.linalg.norm(diff, axis=2)
    
    r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = ~np.eye(N, dtype=bool)
    
    violations = np.maximum(0, r_sum - dists_matrix)
    violations = violations * mask
    penalty += np.sum(violations) ** 2
    
    return score, penalty

def _grow_radii(w, positions, radii, max_iters=1000):
    """
    Given valid positions, try to grow radii to maximize sum while maintaining validity.
    Uses iterative local search on radii.
    """
    h = 2.0 - w
    best_radii = radii.copy()
    best_score = np.sum(best_radii)
    
    for _ in range(max_iters):
        # Try to grow each radius slightly
        improved = False
        for i in range(N):
            # Calculate max possible radius for circle i given current positions and other radii
            max_r = min(positions[i, 0], w - positions[i, 0], positions[i, 1], h - positions[i, 1])
            
            # Check against all other circles
            for j in range(N):
                if i == j:
                    continue
                dist = np.sqrt(np.sum((positions[i] - positions[j])**2))
                max_r = min(max_r, dist - radii[j])
            
            # Try to increase radius a bit
            if radii[i] < max_r - 1e-8:
                new_r = min(max_r - 1e-8, radii[i] + 0.01)
                # Verify this doesn't violate any constraints
                valid = True
                if new_r > max_r + 1e-9:
                    valid = False
                if valid:
                    old_r = radii[i]
                    radii[i] = new_r
                    if np.sum(radii) > best_score:
                        best_score = np.sum(radii)
                        best_radii = radii.copy()
                        improved = True
                    else:
                        radii[i] = old_r
        
        if not improved:
            # Try random perturbations
            for _ in range(10):
                i = np.random.randint(N)
                delta = np.random.uniform(-0.02, 0.02)
                old_r = radii[i]
                new_r = max(0, old_r + delta)
                
                # Check if new_r is valid
                max_r = min(positions[i, 0], w - positions[i, 0], positions[i, 1], h - positions[i, 1])
                for j in range(N):
                    if i == j:
                        continue
                    dist = np.sqrt(np.sum((positions[i] - positions[j])**2))
                    max_r = min(max_r, dist - radii[j])
                
                if new_r <= max_r + 1e-9:
                    radii[i] = new_r
                    if np.sum(radii) > best_score:
                        best_score = np.sum(radii)
                        best_radii = radii.copy()
                        improved = True
                    else:
                        radii[i] = old_r
            
            if not improved:
                break
    
    return best_radii, best_score

def _simulate_annealing(w_init, max_iters=3000, T_init=0.5, T_min=1e-5, alpha=0.9995):
    """
    Runs Simulated Annealing for a fixed initial width w_init.
    Returns best (w, positions, radii, score).
    """
    h_init = 2.0 - w_init
    
    # Better initialization: place circles in a grid with reasonable radii
    # Estimate grid size based on aspect ratio
    aspect = w_init / h_init
    if aspect >= 1:
        cols = int(np.ceil(np.sqrt(N * aspect)))
        rows = (N + cols - 1) // cols
    else:
        rows = int(np.ceil(np.sqrt(N / aspect)))
        cols = (N + rows - 1) // rows
    
    positions = np.zeros((N, 2))
    for i in range(N):
        row = i // cols
        col = i % cols
        positions[i, 0] = (col + 0.5) * w_init / cols
        positions[i, 1] = (row + 0.5) * h_init / rows
    
    # Initial radii: slightly less than half the cell size
    cell_w = w_init / cols
    cell_h = h_init / rows
    initial_r = 0.4 * min(cell_w, cell_h)
    radii = np.full(N, initial_r)
    
    w = w_init
    
    best_score = -np.inf
    best_state = (w, positions.copy(), radii.copy())
    
    current_score, current_penalty = _calculate_penalty_and_score(w, positions, radii)
    # Cost: heavily penalize violations, then maximize score
    current_cost = current_penalty * 1000.0 - current_score
    
    T = T_init
    
    for it in range(max_iters):
        T *= alpha
        if T < T_min:
            break
        
        # Generate perturbation
        r = np.random.rand()
        
        if r < 0.05:
            # Perturb width
            dw = np.random.normal(0, 0.03)
            w_new = w + dw
            if 0.3 < w_new < 1.7:
                w = w_new
        elif r < 0.55:
            # Perturb position of a random circle
            i = np.random.randint(N)
            positions[i, 0] += np.random.normal(0, 0.03)
            positions[i, 1] += np.random.normal(0, 0.03)
        elif r < 0.9:
            # Perturb radius of a random circle
            i = np.random.randint(N)
            radii[i] += np.random.normal(0, 0.01)
            if radii[i] < 0:
                radii[i] = 0
            if radii[i] > 0.5:
                radii[i] = 0.5
        else:
            # Perturb multiple positions
            for _ in range(3):
                i = np.random.randint(N)
                positions[i, 0] += np.random.normal(0, 0.02)
                positions[i, 1] += np.random.normal(0, 0.02)
        
        # Keep positions in reasonable range
        positions[:, 0] = np.clip(positions[:, 0], -0.5, 2.5)
        positions[:, 1] = np.clip(positions[:, 1], -0.5, 2.5)
        
        new_score, new_penalty = _calculate_penalty_and_score(w, positions, radii)
        new_cost = new_penalty * 1000.0 - new_score
        
        delta_cost = new_cost - current_cost
        
        # Metropolis criterion
        if delta_cost < 0 or np.random.rand() < np.exp(-delta_cost / max(T, 1e-10)):
            current_cost = new_cost
            current_score = new_score
            current_penalty = new_penalty
            
            # Track best valid solution
            if current_penalty < 1e-8 and current_score > best_score:
                best_score = current_score
                best_state = (w, positions.copy(), radii.copy())
        
        # Periodically try to grow radii if we have a valid solution
        if it % 200 == 0 and current_penalty < 1e-8:
            grown_radii, grown_score = _grow_radii(w, positions, radii, max_iters=50)
            if grown_score > best_score:
                best_score = grown_score
                best_state = (w, positions.copy(), grown_radii.copy())
                radii = grown_radii.copy()
    
    # Final growth attempt
    if current_penalty < 1e-8:
        grown_radii, grown_score = _grow_radii(w, positions, radii, max_iters=200)
        if grown_score > best_score:
            best_score = grown_score
            best_state = (w, positions.copy(), grown_radii.copy())
    
    return best_state, best_score

def construct():
    best_overall_score = -np.inf
    best_overall_state = None
    
    start_time = time.time()
    time_limit = 110.0
    
    # Try multiple initial widths and restarts
    w_inits = [0.8, 0.9, 1.0, 1.1, 1.2, 0.85, 0.95, 1.05, 1.15]
    
    for restart in range(15):
        if time.time() - start_time > time_limit:
            break
        
        w_init = w_inits[restart % len(w_inits)] + np.random.uniform(-0.05, 0.05)
        w_init = np.clip(w_init, 0.5, 1.5)
        
        state, score = _simulate_annealing(w_init, max_iters=2500)
        
        if score > best_overall_score:
            best_overall_score = score
            best_overall_state = state
    
    if best_overall_state is None:
        # Fallback: grid packing
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.45 / side) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    w, positions, radii = best_overall_state
    
    # Final cleanup: ensure strict validity
    h = 2.0 - w
    
    # Shrink radii to ensure no overlap and inside bounds
    for _ in range(20):
        for i in range(N):
            max_r_bound = min(positions[i, 0], w - positions[i, 0], positions[i, 1], h - positions[i, 1])
            if radii[i] > max_r_bound:
                radii[i] = max(0, max_r_bound - 1e-7)
                
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sqrt(np.sum((positions[i] - positions[j])**2))
                required = radii[i] + radii[j]
                if dist < required:
                    overlap = required - dist
                    shrink_i = overlap / 2.0 + 1e-7
                    shrink_j = overlap / 2.0 + 1e-7
                    radii[i] = max(0, radii[i] - shrink_i)
                    radii[j] = max(0, radii[j] - shrink_j)
    
    # Final check
    for i in range(N):
        if positions[i, 0] < -1e-9 or positions[i, 0] > w + 1e-9 or positions[i, 1] < -1e-9 or positions[i, 1] > h + 1e-9:
            # Try to fix by clipping
            positions[i, 0] = np.clip(positions[i, 0], 0, w)
            positions[i, 1] = np.clip(positions[i, 1], 0, h)
        
        max_r_bound = min(positions[i, 0], w - positions[i, 0], positions[i, 1], h - positions[i, 1])
        if radii[i] > max_r_bound + 1e-9:
            radii[i] = max(0, max_r_bound - 1e-7)
        if radii[i] < 0:
            radii[i] = 0
    
    for i in range(N):
        for j in range(i+1, N):
            dist = np.sqrt(np.sum((positions[i] - positions[j])**2))
            if dist < radii[i] + radii[j] - 1e-9:
                # Shrink both
                overlap = radii[i] + radii[j] - dist
                radii[i] = max(0, radii[i] - overlap/2 - 1e-7)
                radii[j] = max(0, radii[j] - overlap/2 - 1e-7)
    
    # Try one more growth pass on the final valid configuration
    grown_radii, grown_score = _grow_radii(w, positions, radii, max_iters=500)
    if grown_score > np.sum(radii):
        radii = grown_radii
    
    result = np.column_stack([positions, radii])
    return result

# EVOLVE-BLOCK-END
