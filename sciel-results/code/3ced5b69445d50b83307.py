"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 26

# EVOLVE-BLOCK-START

def _compute_energy(pos, radii, penalty_weight=100.0):
    """
    Compute the energy of the configuration.
    Energy = -sum(radii) + penalty_weight * (boundary_violations + overlap_violations)
    We minimize this energy.
    """
    # Boundary violations
    # x - r < 0  => r - x > 0
    # x + r > 1  => x + r - 1 > 0
    bv_left = np.maximum(0, radii - pos[:, 0])
    bv_right = np.maximum(0, pos[:, 0] + radii - 1.0)
    bv_bottom = np.maximum(0, radii - pos[:, 1])
    bv_top = np.maximum(0, pos[:, 1] + radii - 1.0)
    
    boundary_viol = np.sum(bv_left**2) + np.sum(bv_right**2) + np.sum(bv_bottom**2) + np.sum(bv_top**2)
    
    # Overlap violations
    # d_ij < r_i + r_j
    # We use squared distance for speed in the SA loop? No, SA needs frequent evals.
    # N=26 is small enough that O(N^2) sqrt is fine.
    # dist_ij < r_i + r_j  => (r_i + r_j) - dist_ij > 0
    # Let's use linear violation for smoother SA landscape? Or quadratic?
    # Quadratic is standard for penalty methods.
    
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    
    # Only consider i < j
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    # dist = sqrt(dist_sq)
    # viol = max(0, min_dist - dist)
    # To avoid sqrt in the main SA loop if possible, we can compare squares?
    # But viol is linear in distance for the penalty.
    # Let's compute sqrt. N=26 -> 325 pairs. sqrt is fast.
    dist = np.sqrt(dist_sq)
    
    viol = np.maximum(0, min_dist - dist)
    overlap_viol = np.sum(viol[mask] ** 2)
    
    total_viol = boundary_viol + overlap_viol
    score = -np.sum(radii) + penalty_weight * total_viol
    return score

def _is_valid(pos, radii, tol=1e-8):
    """Check if configuration is strictly valid."""
    # Boundary
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False
    
    # Overlap
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    # Check if dist_sq < min_dist^2 - tol^2 (approx)
    # Better: dist < min_dist - tol
    dist = np.sqrt(dist_sq)
    if np.any(dist[mask] < min_dist[mask] - tol):
        return False
        
    return True

def _enforce_validity(pos, radii):
    """
    Greedily shrink radii to ensure validity.
    Returns valid pos, radii.
    """
    pos = pos.copy()
    radii = radii.copy()
    
    # 1. Boundary constraints
    for i in range(N):
        max_r = min(pos[i,0], 1-pos[i,0], pos[i,1], 1-pos[i,1])
        if radii[i] > max_r:
            radii[i] = max(0.0, max_r)
            
    # 2. Overlap constraints
    # Iterate until no changes
    for _ in range(100):
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i,0] - pos[j,0]
                dy = pos[i,1] - pos[j,1]
                dist = np.sqrt(dx*dx + dy*dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    # Shrink both to fit
                    # Ideally, we shrink the smaller one more? Or proportional?
                    # Proportional to current radius is a good heuristic.
                    if sum_r > 1e-10:
                        scale = dist / sum_r
                        radii[i] *= scale
                        radii[j] *= scale
                    else:
                        radii[i] = 0.0
                        radii[j] = 0.0
                    changed = True
        if not changed:
            break
            
    # Final safety clip
    for i in range(N):
        max_r = min(pos[i,0], 1-pos[i,0], pos[i,1], 1-pos[i,1])
        if radii[i] > max_r:
            radii[i] = max_r
            
    return pos, radii

def _simulated_annealing(pos, radii, time_limit, rng):
    """
    Simulated Annealing to maximize sum of radii.
    """
    start_time = time.time()
    T = 0.1 # Initial temperature
    cooling_rate = 0.9995
    min_T = 1e-6
    
    current_pos = pos.copy()
    current_radii = radii.copy()
    current_energy = _compute_energy(current_pos, current_radii)
    
    best_pos = current_pos.copy()
    best_radii = current_radii.copy()
    best_energy = current_energy
    
    step_size_pos = 0.05
    step_size_rad = 0.05
    
    while time.time() - start_time < time_limit and T > min_T:
        # Generate a neighbor
        new_pos = current_pos.copy()
        new_radii = current_radii.copy()
        
        # Perturb 1 or 2 circles
        n_perturb = 1 if rng.random() < 0.7 else 2
        indices = rng.choice(N, size=n_perturb, replace=False)
        
        for idx in indices:
            # Perturb position
            new_pos[idx, 0] += rng.normal(0, step_size_pos)
            new_pos[idx, 1] += rng.normal(0, step_size_pos)
            
            # Perturb radius
            new_radii[idx] += rng.normal(0, step_size_rad)
            
            # Clamp to reasonable bounds to keep search local
            new_pos[idx, 0] = np.clip(new_pos[idx, 0], 0, 1)
            new_pos[idx, 1] = np.clip(new_pos[idx, 1], 0, 1)
            new_radii[idx] = np.clip(new_radii[idx], 0, 0.5)
            
        new_energy = _compute_energy(new_pos, new_radii)
        
        # Acceptance criterion
        delta_E = new_energy - current_energy
        if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
            current_pos = new_pos
            current_radii = new_radii
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_pos = current_pos.copy()
                best_radii = current_radii.copy()
                
        T *= cooling_rate
        
        # Adapt step size? 
        # If we accept a lot, increase step size. If we reject a lot, decrease.
        # Simple heuristic: keep constant for now, or slow decrease.
        step_size_pos *= 0.9999
        step_size_rad *= 0.9999
        
    return best_pos, best_radii

def _polish_configuration(pos, radii, time_limit=5.0):
    """
    Local search to maximize sum of radii starting from a valid or near-valid configuration.
    Strategy:
    1. Enforce validity.
    2. Iteratively try to increase radii of the smallest circles, adjusting positions if needed.
    """
    pos = pos.copy()
    radii = radii.copy()
    start_time = time.time()
    
    pos, radii = _enforce_validity(pos, radii)
    
    rng = np.random.RandomState(42)
    
    # Gradient-like local search / Hill Climbing
    # We want to maximize sum(r).
    # Try to increase radius of a random circle.
    # If it violates, try to move it or neighbors.
    
    for _ in range(5000):
        if time.time() - start_time > time_limit:
            break
            
        # Pick a circle to expand, preferably smaller ones to balance
        # Or just random
        i = rng.randint(N)
        
        # Try to expand radius slightly
        old_r = radii[i]
        # Adaptive step size
        step = 0.005 * (1.0 + old_r) 
        new_r = old_r + step
        
        # Check if new radius is valid with current positions
        # Boundary check
        max_r_wall = min(pos[i,0], 1-pos[i,0], pos[i,1], 1-pos[i,1])
        if new_r > max_r_wall:
            new_r = max_r_wall
            if new_r <= old_r:
                continue
        
        # Overlap check
        valid = True
        for j in range(N):
            if i == j: continue
            dx = pos[i,0] - pos[j,0]
            dy = pos[i,1] - pos[j,1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < new_r + radii[j]:
                valid = False
                break
        
        if valid:
            radii[i] = new_r
        else:
            # Attempt to move circle i to create space
            # Calculate repulsion forces from overlapping circles
            fx, fy = 0.0, 0.0
            for j in range(N):
                if i == j: continue
                dx = pos[i,0] - pos[j,0]
                dy = pos[i,1] - pos[j,1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < 1e-6:
                    dist = 1e-6
                    dx = rng.uniform(-0.01, 0.01)
                    dy = rng.uniform(-0.01, 0.01)
                
                overlap = (new_r + radii[j]) - dist
                if overlap > 0:
                    # Push away
                    fx += (dx / dist) * overlap
                    fy += (dy / dist) * overlap
            
            # Also push away from walls
            if pos[i,0] - new_r < 0:
                fx += (new_r - pos[i,0])
            if pos[i,0] + new_r > 1:
                fx -= (pos[i,0] + new_r - 1)
            if pos[i,1] - new_r < 0:
                fy += (new_r - pos[i,1])
            if pos[i,1] + new_r > 1:
                fy -= (pos[i,1] + new_r - 1)
            
            # Move
            move_step = 0.02
            pos[i,0] += fx * move_step
            pos[i,1] += fy * move_step
            
            # Clamp to bounds
            pos[i,0] = np.clip(pos[i,0], 0.0, 1.0)
            pos[i,1] = np.clip(pos[i,1], 0.0, 1.0)
            
            # After moving, try to set radius to max possible
            # This is a greedy approach: expand as much as possible
            max_r = min(pos[i,0], 1-pos[i,0], pos[i,1], 1-pos[i,1])
            for j in range(N):
                if i == j: continue
                dx = pos[i,0] - pos[j,0]
                dy = pos[i,1] - pos[j,1]
                dist = np.sqrt(dx*dx + dy*dy)
                max_r = min(max_r, dist - radii[j])
            
            if max_r > radii[i]:
                radii[i] = max_r
            else:
                # If we can't even fit current radius, shrink it
                radii[i] = max(0.0, max_r)

    # Final strict validity check and shrink
    pos, radii = _enforce_validity(pos, radii)
    
    return pos, radii

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin for 120s limit
    
    best_pos = None
    best_radii = None
    best_score = 0.0
    
    rng = np.random.RandomState(42)
    
    # Strategy: Multi-start Simulated Annealing
    # 1. Generate initial configurations
    # 2. Run SA on each
    # 3. Polish the best result
    
    n_starts = 10
    time_per_start = (deadline - start_time) / (n_starts * 2) # Leave time for polishing
    
    for s in range(n_starts):
        if time.time() > deadline - 10:
            break
            
        # Initialize
        # Use a grid-like perturbation or random
        if s == 0:
            # Grid init
            pos = np.zeros((N, 2))
            radii = np.zeros(N)
            k = 0
            for i in range(5):
                for j in range(5):
                    if k < N:
                        pos[k] = [(i + 0.5) / 5.0, (j + 0.5) / 5.0]
                        radii[k] = 0.09
                        k += 1
            if k < N:
                pos[k] = [0.5, 0.5]
                radii[k] = 0.01
        else:
            # Random init, ensure no initial overlaps by simple rejection or scaling
            pos = rng.uniform(0.1, 0.9, (N, 2))
            radii = rng.uniform(0.05, 0.15, N)
            # Enforce validity immediately to get a valid start
            pos, radii = _enforce_validity(pos, radii)
            
        # Run SA
        sa_time = min(time_per_start, 5.0)
        pos_sa, rad_sa = _simulated_annealing(pos, radii, sa_time, rng)
        
        # Polish
        polish_time = min(2.0, deadline - time.time() - 5)
        if polish_time <= 0:
            continue
            
        pos_pol, rad_pol = _polish_configuration(pos_sa, rad_sa, polish_time)
        
        # Validate
        if _is_valid(pos_pol, rad_pol):
            score = np.sum(rad_pol)
            if score > best_score:
                best_score = score
                best_pos = pos_pol
                best_radii = rad_pol
        
        # If we have a very good score, we can stop early?
        # Sum of radii for 26 circles in unit square.
        # Max possible sum is bounded by area? No, sum of radii.
        # If we get > 2.5, it's quite good.
        if best_score > 2.5:
            break

    # Fallback if SA failed or produced nothing
    if best_pos is None:
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        k = 0
        for i in range(5):
            for j in range(5):
                if k < N:
                    pos[k] = [(i + 0.5) / 5.0, (j + 0.5) / 5.0]
                    radii[k] = 0.09
                    k += 1
        if k < N:
            pos[k] = [0.5, 0.5]
            radii[k] = 0.01
        best_pos, best_radii = _enforce_validity(pos, radii)
        best_pos, best_radii = _polish_configuration(best_pos, best_radii, time_limit=5.0)

    # Final strict validation and shrink
    best_pos, best_radii = _enforce_validity(best_pos, best_radii)
    
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
