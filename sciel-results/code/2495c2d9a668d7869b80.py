"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _objective(x, penalty_weight=500.0):
    """
    Objective function: minimize negative sum of radii + penalty for violations.
    x is a flattened array of [x_1, y_1, ..., x_N, y_N, r_1, ..., r_N]
    """
    pos = x[:2*N].reshape(N, 2)
    radii = x[2*N:]
    
    # Ensure radii are positive
    radii = np.maximum(radii, 1e-6)
    
    # 1. Boundary Violations (Smooth Squared Penalty)
    left_viol = np.maximum(0.0, radii - pos[:, 0])
    right_viol = np.maximum(0.0, pos[:, 0] - (1.0 - radii))
    bottom_viol = np.maximum(0.0, radii - pos[:, 1])
    top_viol = np.maximum(0.0, pos[:, 1] - (1.0 - radii))
    
    boundary_penalty = np.sum(left_viol**2) + np.sum(right_viol**2) + \
                       np.sum(bottom_viol**2) + np.sum(top_viol**2)
    
    # 2. Pairwise Overlap Violations (Vectorized)
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    min_dist_sq = min_dist * min_dist
    
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    # Smoother penalty: (violation)^2 / (min_dist^2 + epsilon)
    violation = np.maximum(0.0, min_dist_sq - dist_sq)
    denom = min_dist_sq + 1e-8
    smooth_overlap = (violation ** 2) / denom
    
    overlap_penalty = np.sum(smooth_overlap[mask])
    
    total_penalty = boundary_penalty + overlap_penalty
    
    return -np.sum(radii) + penalty_weight * total_penalty

def _is_valid(pos, radii, tol=1e-7):
    """Strict validity check for final result."""
    n = len(radii)
    # Check boundaries
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False
        
    # Check overlaps
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    # dist_sq < min_dist^2  =>  dist < min_dist
    if np.any(dist_sq[mask] < (min_dist[mask] - tol)**2):
        return False
        
    return True

def _post_process(pos, radii):
    """
    Shrink radii slightly to ensure strict validity if the optimizer
    left them on the boundary or slightly overlapping due to numerical noise.
    """
    pos = pos.copy()
    radii = radii.copy()
    
    for _ in range(200):
        valid = True
        
        # 1. Boundary fix
        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            max_r = min(x, 1.0-x, y, 1.0-y)
            if r > max_r:
                radii[i] = max(max_r * 0.9999, 1e-5)
                valid = False
        
        # 2. Overlap fix
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    # Scale both radii to make them just touch
                    k = dist / (sum_r + 1e-12) * 0.9999
                    if k < 1.0:
                        radii[i] = max(radii[i] * k, 1e-5)
                        radii[j] = max(radii[j] * k, 1e-5)
                        valid = False
                        
        if valid:
            break
            
    return pos, radii

def _initial_best_fit():
    """
    Generate a tight initial packing using a greedy best-fit algorithm.
    """
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # Start with a large central circle
    pos[0] = [0.5, 0.5]
    radii[0] = 0.25
    
    for i in range(1, N):
        best_x, best_y, best_r = 0.5, 0.5, 0.0
        found = False
        
        rng = np.random.RandomState(i * 1000 + 42)
        
        # Sample 3000 random points for better coverage
        for _ in range(3000):
            x = rng.uniform(0.02, 0.98)
            y = rng.uniform(0.02, 0.98)
            
            # Max radius allowed by walls
            max_r_wall = min(x, 1.0-x, y, 1.0-y)
            
            # Max radius allowed by existing circles
            max_r_circ = max_r_wall
            for j in range(i):
                dx = x - pos[j, 0]
                dy = y - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                possible_r = dist - radii[j]
                if possible_r < max_r_circ:
                    max_r_circ = possible_r
            
            if max_r_circ > best_r and max_r_circ > 0.003:
                best_r = max_r_circ
                best_x = x
                best_y = y
                found = True
        
        if not found or best_r < 0.003:
            best_x, best_y = 0.5, 0.5
            best_r = 0.003
            
        pos[i] = [best_x, best_y]
        radii[i] = best_r * 0.97
        
    return pos, radii

def _initial_lattice():
    """
    Generate a hexagonal lattice-like initial state.
    """
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    r = 0.11
    dx = r * np.sqrt(3)
    dy = r * 1.5
    
    k = 0
    for row in range(6):
        if row == 5:
            x = 0.5
            y = row * dy + r
            if k < N:
                pos[k] = [x, y]
                radii[k] = r
                k += 1
            continue
            
        offset = dx / 2 if row % 2 == 1 else 0
        start_x = r + offset
        
        for col in range(5):
            if k >= N:
                break
            x = start_x + col * dx
            y = row * dy + r
            
            if x + r <= 1.0 and x - r >= 0.0 and y + r <= 1.0 and y - r >= 0.0:
                pos[k] = [x, y]
                radii[k] = r
                k += 1
        
        if k >= N:
            break
            
    while k < N:
        placed = False
        for _ in range(100):
            x = np.random.uniform(0.1, 0.9)
            y = np.random.uniform(0.1, 0.9)
            min_dist = min(x, 1-x, y, 1-y)
            for j in range(k):
                d = np.sqrt((x-pos[j,0])**2 + (y-pos[j,1])**2)
                min_dist = min(min_dist, d - radii[j])
            if min_dist > 0.02:
                pos[k] = [x, y]
                radii[k] = min_dist * 0.95
                k += 1
                placed = True
                break
        if not placed:
            pos[k] = [0.5, 0.5]
            radii[k] = 0.01
            k += 1
            
    return pos, radii

def _initial_random(seed):
    """Generate a random valid-ish initial state."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, (N, 2))
    radii = rng.uniform(0.04, 0.12, N)
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _vectorized_anneal(pos, radii, steps=1000, start_temp=0.05, end_temp=0.0005, rng=None):
    """
    Vectorized Simulated Annealing.
    Faster than loop-based force directed. Uses vectorized distance calculations.
    """
    if rng is None:
        rng = np.random.RandomState(42)
    
    pos = pos.copy()
    radii = radii.copy()
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    for step in range(steps):
        # Cooldown
        temp = start_temp * (end_temp / start_temp) ** (step / steps)
        
        # 1. Calculate Forces (Vectorized)
        # Pairwise differences
        dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
        dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
        dist_sq = dx * dx + dy * dy
        dist = np.sqrt(dist_sq + 1e-12)
        
        # Repulsion: only for overlapping pairs (dist < r_i + r_j)
        sum_r = radii[:, np.newaxis] + radii[np.newaxis, :]
        # Overlap amount: max(0, sum_r - dist)
        overlap = np.maximum(0, sum_r - dist)
        
        # Force magnitude proportional to overlap.
        # We want to push them apart.
        # Force vector direction is (dx, dy) / dist.
        # Since we only care about i != j, and force is symmetric, we can sum contributions.
        # F_ij = overlap * (dx, dy) / dist
        # Note: if dist is 0, direction is undefined, but overlap is 0 if radii are 0? No, if radii > 0 and dist=0, overlap > 0.
        # Add small noise to break symmetry if dist is very small.
        safe_dist = np.where(dist < 1e-6, 1e-6, dist)
        
        # Vector of forces from j to i
        # F_x[i] = sum_j overlap[i,j] * dx[i,j] / safe_dist[i,j]
        # Since overlap is symmetric and dx is antisymmetric, the sum over j works out.
        fx = np.sum(overlap * dx / safe_dist, axis=1)
        fy = np.sum(overlap * dy / safe_dist, axis=1)
        
        # Attraction to center (weak) to prevent drifting
        ax = (0.5 - pos[:, 0]) * 0.5
        ay = (0.5 - pos[:, 1]) * 0.5
        
        forces_x = fx + ax
        forces_y = fy + ay
        
        # 2. Update Positions
        # Step size depends on temperature
        step_size = 0.05 * temp
        noise_x = rng.normal(0, temp, N)
        noise_y = rng.normal(0, temp, N)
        
        new_pos = pos.copy()
        new_pos[:, 0] += forces_x * step_size + noise_x
        new_pos[:, 1] += forces_y * step_size + noise_y
        
        # Clip to bounds (approximate, will be refined)
        # We don't clip to radii yet, we let the radius expansion handle it
        new_pos = np.clip(new_pos, 0.0, 1.0)
        
        # 3. Grow Radii
        # For each circle, find max possible radius given new positions
        # Max radius is min(
        #   dist to walls,
        #   min_j (dist(i,j) - r_j)
        # )
        
        # Dist to walls
        r_wall_x = np.minimum(new_pos[:, 0], 1.0 - new_pos[:, 0])
        r_wall_y = np.minimum(new_pos[:, 1], 1.0 - new_pos[:, 1])
        max_r_wall = np.minimum(r_wall_x, r_wall_y)
        
        # Dist to other circles
        # We need min_j (dist(i,j) - r_j)
        # dist is already calculated for old pos, but we moved.
        # Recalculate distances for new_pos? That's expensive.
        # Approximation: use old distances but adjust?
        # Better: Recalculate pairwise distances for new_pos.
        
        ndx = new_pos[:, 0, np.newaxis] - new_pos[np.newaxis, :, 0]
        ndy = new_pos[:, 1, np.newaxis] - new_pos[np.newaxis, :, 1]
        ndist = np.sqrt(ndx*ndx + ndy*ndy + 1e-12)
        
        # max_r_circ[i] = min_j (ndist[i,j] - radii[j])
        # Note: we subtract the OTHER circle's radius.
        # We want to grow r_i such that r_i + r_j <= ndist[i,j] => r_i <= ndist[i,j] - r_j
        # So for each i, we look at all j != i.
        
        # Create a matrix of potential radii: M[i,j] = ndist[i,j] - radii[j]
        M = ndist - radii[np.newaxis, :]
        
        # Set diagonal to infinity (ignore self)
        np.fill_diagonal(M, np.inf)
        
        # min over j for each i
        max_r_circ = np.min(M, axis=1)
        
        # Combine constraints
        max_possible_r = np.minimum(max_r_wall, max_r_circ)
        
        # Grow towards max_possible_r
        # We want to increase radii.
        # new_radii = radii + alpha * (max_possible_r - radii)
        # But we must ensure we don't exceed max_possible_r.
        # Also, if max_possible_r < radii, we are in violation.
        
        # Strategy:
        # If max_possible_r > radii: grow a bit.
        # If max_possible_r < radii: shrink to max_possible_r (to restore validity).
        
        growth_rate = 0.1 # 10% of gap per step
        # If we are below max, grow.
        # If we are above max, shrink.
        
        diff = max_possible_r - radii
        # If diff > 0, grow by growth_rate * diff
        # If diff < 0, shrink by 1.0 * diff (immediate fix)
        
        # Actually, immediate fix is safer.
        # Let's try: new_radii = radii + growth_rate * diff
        # This is a soft constraint.
        
        new_radii = radii + growth_rate * diff
        
        # Ensure non-negative
        new_radii = np.maximum(new_radii, 1e-5)
        
        # 4. Acceptance Criterion
        # Calculate score
        new_score = np.sum(new_radii)
        old_score = np.sum(radii)
        
        # Penalty for invalidity?
        # We already enforced r <= max_r, so it should be valid?
        # Wait, max_r_circ was calculated using OLD radii.
        # If we grow r_i, it might violate r_i + r_j <= dist for some j.
        # Because M[i,j] = dist - r_j.
        # If we set r_i = M[i,j], then r_i + r_j = dist.
        # But if we grow r_k (k != i, j), it might affect i.
        # So the packing might be invalid.
        
        # We need a validity check or a penalty.
        # For speed, let's assume the "max_r" calculation is a good heuristic for a valid-ish state.
        # But to be safe, we can do a quick check or just rely on post_process.
        
        # Let's add a penalty term to the score for the annealing decision.
        # Penalty = sum of violations.
        
        # Violations:
        # 1. Boundary: r > min(x, 1-x, y, 1-y)
        #    We enforced r <= max_r_wall, so this is 0.
        # 2. Overlap: r_i + r_j > dist
        #    We used old radii to calc max_r.
        
        # Let's just use the score and accept if better, or with probability exp(-delta/T).
        # But if the state is invalid, the "score" is misleading.
        # Let's compute a penalty.
        
        # Overlap penalty
        # dist is ndist
        # sum_r_new = new_radii[:, np.newaxis] + new_radii[np.newaxis, :]
        # viol = max(0, sum_r_new - ndist)
        # penalty = sum(viol^2)
        
        sum_r_new = new_radii[:, np.newaxis] + new_radii[np.newaxis, :]
        viol = np.maximum(0, sum_r_new - ndist)
        # Exclude diagonal
        np.fill_diagonal(viol, 0)
        penalty = np.sum(viol * viol)
        
        # Boundary penalty (should be 0)
        b_viol = np.maximum(0, new_radii - max_r_wall)
        b_penalty = np.sum(b_viol * b_viol)
        
        total_penalty = penalty + b_penalty
        
        # Objective: maximize sum_r - penalty
        new_obj = new_score - 10.0 * total_penalty
        old_obj = old_score # Old state was assumed valid or close
        
        delta = new_obj - old_obj
        
        if delta > 0 or rng.uniform() < np.exp(delta / temp):
            pos = new_pos
            radii = new_radii
            
            # Update best if valid
            if total_penalty < 1e-6:
                if new_score > best_score:
                    best_score = new_score
                    best_pos = pos.copy()
                    best_radii = radii.copy()
                    
    return best_pos, best_radii

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Strategy: Multi-start with Vectorized Annealing
    inits = []
    
    # 1. Best Fit
    pos_bf, rad_bf = _initial_best_fit()
    inits.append((pos_bf, rad_bf))
    
    # 2. Lattice
    pos_lat, rad_lat = _initial_lattice()
    inits.append((pos_lat, rad_lat))
    
    # 3. Random (6 variations)
    for i in range(6):
        pos_r, rad_r = _initial_random(100 + i)
        inits.append((pos_r, rad_r))
    
    best_pos = pos_bf.copy()
    best_radii = rad_bf.copy()
    best_score = np.sum(best_radii)
    
    # Post-process initial states to ensure validity
    for i, (pos_init, rad_init) in enumerate(inits):
        pos_init, rad_init = _post_process(pos_init, rad_init)
        inits[i] = (pos_init, rad_init)
        if np.sum(rad_init) > best_score:
            best_score = np.sum(rad_init)
            best_pos = pos_init.copy()
            best_radii = rad_init.copy()
            
    # Run Vectorized Annealing on each initialization
    for idx, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 10:
            break
            
        # Allocate time based on remaining budget
        # Give each run ~15-20 seconds if we have 8 runs?
        # 145s / 8 runs ~ 18s per run.
        # Vectorized anneal is fast. 1000 steps is likely < 1s.
        # We can run multiple anneals or longer ones.
        
        # Let's run 3 anneals with different temps
        for run in range(3):
            if time.time() > deadline - 5:
                break
                
            # Vary temperature for diversity
            start_temp = 0.05 + 0.01 * run
            steps = 2000
            
            pos_a, rad_a = _vectorized_anneal(pos_init, rad_init, steps=steps, start_temp=start_temp, end_temp=0.0005)
            
            # Post-process
            pos_a, rad_a = _post_process(pos_a, rad_a)
            
            if _is_valid(pos_a, rad_a):
                score_a = np.sum(rad_a)
                if score_a > best_score:
                    best_score = score_a
                    best_pos = pos_a.copy()
                    best_radii = rad_a.copy()
                    
            # Use the result as new init for next run (chaining)
            pos_init, rad_init = pos_a, rad_a
            
    # Final validation and post-processing
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
    # Fallback if invalid
    if not _is_valid(best_pos, best_radii):
         pos = np.zeros((N, 2))
         radii = np.zeros(N)
         k = 0
         for i in range(5):
             for j in range(5):
                 if k < N-1:
                     pos[k] = [(i+0.5)/5.0, (j+0.5)/5.0]
                     radii[k] = 0.095
                     k += 1
         pos[k] = [0.5, 0.02]
         radii[k] = 0.01
         best_pos, best_radii = pos, radii
         best_pos, best_radii = _post_process(best_pos, best_radii)

    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
