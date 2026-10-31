"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _is_valid(pos, radii, tol=1e-9):
    """Check if circles are valid (inside square, disjoint interiors)."""
    n = len(radii)
    for i in range(n):
        if pos[i, 0] - radii[i] < -tol or pos[i, 0] + radii[i] > 1.0 + tol:
            return False
        if pos[i, 1] - radii[i] < -tol or pos[i, 1] + radii[i] > 1.0 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < radii[i] + radii[j] - tol:
                return False
    return True

def _objective(x):
    """Objective: minimize negative sum of radii + penalty for violations."""
    n = N
    pos = x[:2*n].reshape(n, 2)
    radii = x[2*n:]
    
    radii = np.maximum(radii, 1e-6)
    
    penalty = 0.0
    # Boundary penalties
    left_viol = np.maximum(0.0, radii - pos[:, 0])
    right_viol = np.maximum(0.0, pos[:, 0] - (1.0 - radii))
    bottom_viol = np.maximum(0.0, radii - pos[:, 1])
    top_viol = np.maximum(0.0, pos[:, 1] - (1.0 - radii))
    penalty += np.sum(left_viol**2) + np.sum(right_viol**2) + np.sum(bottom_viol**2) + np.sum(top_viol**2)
    
    # Pairwise overlap penalties (vectorized)
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    min_dist_sq = min_dist * min_dist
    
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    # Penalty for dist_sq < min_dist_sq
    # We use a smooth approximation: max(0, min_dist_sq - dist_sq)
    overlap_sq = np.maximum(0.0, min_dist_sq - dist_sq)
    # Scale to avoid huge gradients
    penalty += np.sum(overlap_sq[mask])
    
    return -np.sum(radii) + 1000.0 * penalty

def _initial_grid():
    """Create a 5x5 grid of circles with the 26th placed in a corner gap."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    r = 0.09
    for i in range(5):
        for j in range(5):
            idx = i * 5 + j
            pos[idx, 0] = (i + 0.5) / 5.0
            pos[idx, 1] = (j + 0.5) / 5.0
            radii[idx] = r
    
    pos[25, 0] = 0.5
    pos[25, 1] = 0.95
    radii[25] = 0.05
    
    return pos, radii

def _initial_hexagonal():
    """Create a hexagonal packing pattern."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    r = 0.095
    y_spacing = r * np.sqrt(3)
    
    # Center vertically
    total_height = 4 * y_spacing + 2 * r
    y_start = (1.0 - total_height) / 2.0
    
    # Row 0: 5 circles
    x_start_5 = (1.0 - 4 * 2 * r) / 2.0
    for i in range(5):
        idx = i
        pos[idx, 0] = x_start_5 + i * 2 * r
        pos[idx, 1] = y_start + r
        radii[idx] = r
    
    # Row 1: 4 circles
    x_start_4 = (1.0 - 3 * 2 * r) / 2.0
    for i in range(4):
        idx = 5 + i
        pos[idx, 0] = x_start_4 + i * 2 * r
        pos[idx, 1] = y_start + r + y_spacing
        radii[idx] = r
        
    # Row 2: 5 circles
    for i in range(5):
        idx = 9 + i
        pos[idx, 0] = x_start_5 + i * 2 * r
        pos[idx, 1] = y_start + r + 2 * y_spacing
        radii[idx] = r
        
    # Row 3: 4 circles
    for i in range(4):
        idx = 14 + i
        pos[idx, 0] = x_start_4 + i * 2 * r
        pos[idx, 1] = y_start + r + 3 * y_spacing
        radii[idx] = r
        
    # Row 4: 5 circles
    for i in range(5):
        idx = 18 + i
        pos[idx, 0] = x_start_5 + i * 2 * r
        pos[idx, 1] = y_start + r + 4 * y_spacing
        radii[idx] = r
        
    # 26th circle: small circle in a gap
    pos[25, 0] = 0.5
    pos[25, 1] = 0.5
    radii[25] = 0.02
    
    return pos, radii

def _grow_and_push(pos, radii, deadline, step_size=0.005, max_iters=5000):
    """
    Hard-constraint grow-and-push algorithm.
    Iteratively grows the circle with the most clearance and pushes neighbors to maintain validity.
    """
    pos = pos.copy()
    radii = radii.copy()
    
    # Start with slightly smaller radii to allow growth
    radii *= 0.5
    
    current_score = np.sum(radii)
    
    for it in range(max_iters):
        if time.time() > deadline - 10:
            break
            
        # 1. Calculate clearance for each circle
        # Clearance = min(distance to boundary, distance to other circles) - radius
        clearances = np.zeros(N)
        
        # Distance to boundaries
        dist_bounds = np.minimum(
            np.minimum(pos[:, 0], 1.0 - pos[:, 0]),
            np.minimum(pos[:, 1], 1.0 - pos[:, 1])
        )
        clearances = dist_bounds - radii
        
        # Distance to other circles
        for i in range(N):
            # Vectorized distance to all other circles
            dx = pos[:, 0] - pos[i, 0]
            dy = pos[:, 1] - pos[i, 1]
            dists = np.sqrt(dx*dx + dy*dy)
            dists[i] = np.inf # Ignore self
            
            # Minimum distance to any other circle
            min_dist_other = np.min(dists)
            
            # The clearance relative to circle i is min_dist_other - radii[i]
            # But we also need to consider that the other circle has a radius.
            # The "gap" is min_dist_other - radii[i] - radii[j_min]
            # Actually, the maximum growth for i is limited by:
            # dist(i, j) >= r_i + r_j  =>  r_i <= dist(i, j) - r_j
            # So max r_i = min_j (dist(i, j) - r_j)
            # And also r_i <= dist_to_bound
            # So clearance = min(dist_to_bound, min_j(dist(i,j) - r_j)) - r_i
            
            max_possible_r = dist_bounds[i]
            for j in range(N):
                if i == j: continue
                limit = dists[j] - radii[j]
                if limit < max_possible_r:
                    max_possible_r = limit
                    
            clearances[i] = max_possible_r - radii[i]
            
        # 2. Find circle with max clearance
        idx_max = np.argmax(clearances)
        
        if clearances[idx_max] < 1e-6:
            # No room to grow any circle significantly
            break
            
        # 3. Grow the circle
        dr = min(step_size, clearances[idx_max])
        radii[idx_max] += dr
        
        # 4. Resolve overlaps (Push)
        # We need to ensure that for all j, dist(idx_max, j) >= radii[idx_max] + radii[j]
        # And boundaries are respected.
        
        # Boundary check for idx_max
        if pos[idx_max, 0] < radii[idx_max]:
            pos[idx_max, 0] = radii[idx_max]
        elif pos[idx_max, 0] > 1.0 - radii[idx_max]:
            pos[idx_max, 0] = 1.0 - radii[idx_max]
        if pos[idx_max, 1] < radii[idx_max]:
            pos[idx_max, 1] = radii[idx_max]
        elif pos[idx_max, 1] > 1.0 - radii[idx_max]:
            pos[idx_max, 1] = 1.0 - radii[idx_max]
            
        # Pairwise check
        for j in range(N):
            if i == j: continue
            dx = pos[idx_max, 0] - pos[j, 0]
            dy = pos[idx_max, 1] - pos[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = radii[idx_max] + radii[j]
            
            if dist < min_dist:
                # Overlap! Push j away from idx_max
                if dist < 1e-9:
                    # Random push
                    angle = np.random.rand() * 2 * np.pi
                    pos[j, 0] += np.cos(angle) * 0.01
                    pos[j, 1] += np.sin(angle) * 0.01
                    # Clip to bounds
                    pos[j, 0] = np.clip(pos[j, 0], radii[j], 1.0 - radii[j])
                    pos[j, 1] = np.clip(pos[j, 1], radii[j], 1.0 - radii[j])
                else:
                    # Move j along the vector from idx_max to j
                    overlap = min_dist - dist
                    # Move j by overlap in the direction away from idx_max
                    # To be safe, move j by half the overlap and idx_max by half?
                    # No, idx_max just grew, so we should push j.
                    # But pushing j might cause j to overlap with k.
                    # Simple approach: Push j by the full overlap amount.
                    unit_vec_x = dx / dist
                    unit_vec_y = dy / dist
                    pos[j, 0] += unit_vec_x * overlap
                    pos[j, 1] += unit_vec_y * overlap
                    
                    # Clip j to bounds
                    pos[j, 0] = np.clip(pos[j, 0], radii[j], 1.0 - radii[j])
                    pos[j, 1] = np.clip(pos[j, 1], radii[j], 1.0 - radii[j])
                    
                    # If clipping caused a new overlap or boundary violation, 
                    # we might need to shrink j or idx_max slightly.
                    # For now, we trust the iterative process to resolve.
                    # However, if j is pushed out of bounds, we must shrink.
                    if pos[j, 0] <= radii[j] or pos[j, 0] >= 1.0 - radii[j] or pos[j, 1] <= radii[j] or pos[j, 1] >= 1.0 - radii[j]:
                         # If j is stuck at boundary, we can't push it further.
                         # We must shrink idx_max or j to satisfy constraint.
                         # Shrink idx_max a bit
                         radii[idx_max] = max(1e-4, radii[idx_max] - dr/2.0)
                         # And retry this iteration? Or just continue.
                         # To avoid infinite loops, we just continue.
                         break

        # 5. Check if score improved
        new_score = np.sum(radii)
        if new_score > current_score + 1e-9:
            current_score = new_score
        else:
            # If score didn't improve (likely due to pushes reducing radii or no growth), 
            # we might be stuck. Try a smaller step or break.
            # To prevent getting stuck in a loop where we grow then push back,
            # we can reduce the step size.
            step_size *= 0.9
            if step_size < 1e-5:
                break

    return pos, radii

def _polish(pos, radii, deadline):
    """High-precision local search to squeeze out more sum of radii."""
    x0 = np.concatenate([pos.flatten(), radii])
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((1e-5, 0.5))
        
    best_score = np.sum(radii)
    best_pos = pos.copy()
    best_radii = radii.copy()
    
    try:
        result = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-10})
        
        if result.fun < _objective(x0):
            x_opt = result.x
            pos_opt = x_opt[:2*N].reshape(N, 2)
            radii_opt = np.maximum(x_opt[2*N:], 1e-5)
            
            # Ensure validity by shrinking if necessary
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            score = np.sum(radii_pp)
            
            if _is_valid(pos_pp, radii_pp) and score > best_score:
                return pos_pp, radii_pp, score
    except Exception:
        pass
        
    return best_pos, best_radii, best_score

def _post_process(pos, radii):
    """Iteratively shrink circles to ensure validity."""
    pos = pos.copy()
    radii = radii.copy()
    
    for iteration in range(100):
        valid = True
        
        for i in range(N):
            if pos[i, 0] < radii[i]:
                radii[i] = max(pos[i, 0] * 0.99, 1e-4)
                valid = False
            if pos[i, 0] > 1.0 - radii[i]:
                radii[i] = min(radii[i], (1.0 - pos[i, 0]) * 0.99)
                radii[i] = max(radii[i], 1e-4)
                valid = False
            if pos[i, 1] < radii[i]:
                radii[i] = min(radii[i], pos[i, 1] * 0.99)
                radii[i] = max(radii[i], 1e-4)
                valid = False
            if pos[i, 1] > 1.0 - radii[i]:
                radii[i] = min(radii[i], (1.0 - pos[i, 1]) * 0.99)
                radii[i] = max(radii[i], 1e-4)
                valid = False
        
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    scale = dist / (min_dist + 1e-12) * 0.99
                    radii[i] = max(radii[i] * scale, 1e-4)
                    radii[j] = max(radii[j] * scale, 1e-4)
                    valid = False
        
        if valid:
            break
    
    return pos, radii

def _optimize_from(pos, radii, deadline):
    """Run optimization from given initial configuration."""
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    # Phase 1: Grow and Push
    if time.time() < deadline - 20:
        pos_gp, radii_gp = _grow_and_push(best_pos, best_radii, deadline)
        score_gp = np.sum(radii_gp)
        
        # Post-process to ensure strict validity
        pos_pp, radii_pp = _post_process(pos_gp, radii_gp)
        score_pp = np.sum(radii_pp)
        
        if _is_valid(pos_pp, radii_pp) and score_pp > best_score:
            best_score = score_pp
            best_pos = pos_pp.copy()
            best_radii = radii_pp.copy()
            
    # Phase 2: Polish
    if time.time() < deadline - 5:
        best_pos, best_radii, best_score = _polish(best_pos, best_radii, deadline)
        
    return best_pos, best_radii, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_pos = None
    best_radii = None
    best_score = -np.inf
    
    # Initial configurations
    init_configs = []
    
    pos, radii = _initial_grid()
    init_configs.append((pos, radii))
    
    pos, radii = _initial_hexagonal()
    init_configs.append((pos, radii))
    
    # Random starts
    rng = np.random.RandomState(42)
    for seed in [1, 2, 3]:
        pos_r = rng.uniform(0.2, 0.8, size=(N, 2))
        radii_r = np.full(N, 0.05)
        init_configs.append((pos_r, radii_r))
        
    # Run optimization from each
    for idx, (pos, radii) in enumerate(init_configs):
        if time.time() > deadline - 15:
            break
            
        pos_opt, radii_opt, score = _optimize_from(pos, radii, deadline)
        
        if score > best_score:
            best_score = score
            best_pos = pos_opt.copy()
            best_radii = radii_opt.copy()
            
    # Fallback
    if best_pos is None or best_score < 0:
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        for i in range(5):
            for j in range(5):
                idx2 = i * 5 + j
                if idx2 < N:
                    pos[idx2, 0] = (i + 0.5) / 5.0
                    pos[idx2, 1] = (j + 0.5) / 5.0
                    radii[idx2] = 0.09
        pos[25, 0] = 0.1
        pos[25, 1] = 0.9
        radii[25] = 0.07
        best_pos = pos
        best_radii = radii
        best_score = np.sum(radii)
        
    # Final check
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
