import numpy as np
import time
from scipy.optimize import minimize

N = 50

def compute_radius(positions):
    """
    Compute the maximum radius for the given positions.
    Returns (radius, is_valid).
    If invalid (points outside), radius is -1.
    """
    # Boundary constraint
    max_abs = np.max(np.abs(positions), axis=1)
    bound_dists = 0.5 - max_abs
    
    # If any point is outside, radius is invalid (negative)
    if np.any(bound_dists < -1e-9):
        return -1.0, False
        
    min_bound = np.min(bound_dists)
    
    # Pairwise distance constraint
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=-1)
    
    # Ignore self-distances
    np.fill_diagonal(dists_sq, np.inf)
    
    min_dist_sq = np.min(dists_sq)
    min_dist = np.sqrt(min_dist_sq)
    
    r = min(min_bound, min_dist / 2.0)
    
    return r, True

def compute_radius_fast(positions, moved_idx, new_pos, min_dist_sq_cache):
    """
    Fast radius computation when only one point has moved.
    min_dist_sq_cache: array of shape (N,) where min_dist_sq_cache[i] is the 
    minimum squared distance from point i to any other point (excluding itself).
    Returns (radius, is_valid, updated_min_dist_sq_cache).
    """
    # Boundary constraint for the moved point
    moved_max_abs = np.max(np.abs(new_pos))
    moved_bound_dist = 0.5 - moved_max_abs
    
    if moved_bound_dist < -1e-9:
        return -1.0, False, min_dist_sq_cache
    
    # Update distances from moved point to all others
    diffs = positions - new_pos  # shape (N, 2)
    diffs[moved_idx] = 0  # exclude self
    dists_sq = np.sum(diffs**2, axis=1)
    
    # Minimum distance from moved point to any other
    new_min_dist_sq_moved = np.min(dists_sq)
    
    # For all other points, their min distance to any other point might have changed
    # if the moved point was their closest neighbor
    # We need to recompute min_dist_sq for all points except moved_idx
    # Actually, we can be smarter: only points that had moved_idx as their closest need updating
    # But for simplicity and correctness, let's just recompute the full min_dist_sq cache
    # This is still faster than full pairwise if we do it incrementally
    
    # Recompute min_dist_sq_cache for all points
    # For each point j != moved_idx, min_dist_sq_cache[j] = min(min_dist_sq_cache[j], new distances)
    # But we also need to handle the case where moved_idx WAS the closest to some point
    
    # Simple approach: recompute full min_dist_sq_cache
    # This is O(N^2) but with smaller constant than full distance matrix
    
    new_min_dist_sq_cache = np.full(N, np.inf)
    for i in range(N):
        if i == moved_idx:
            new_min_dist_sq_cache[i] = new_min_dist_sq_moved
        else:
            # Distance from i to moved_idx
            d_sq = np.sum((positions[i] - new_pos)**2)
            # Minimum of old cache (excluding moved_idx) and d_sq
            # We need the min distance from i to any point j != i, j != moved_idx, plus d_sq
            # This requires knowing the second minimum, which we don't have cached
            # So we just recompute: min of all distances from i to others
            # But that's O(N) per point, so O(N^2) total
            diffs_i = positions - positions[i]
            diffs_i[i] = 0
            diffs_i[moved_idx] = 0  # exclude moved_idx temporarily
            dists_sq_i = np.sum(diffs_i**2, axis=1)
            min_excl_moved = np.min(dists_sq_i) if N > 2 else np.inf
            new_min_dist_sq_cache[i] = min(min_excl_moved, d_sq)
    
    # Now compute the global min
    min_dist_sq = np.min(new_min_dist_sq_cache)
    min_dist = np.sqrt(min_dist_sq)
    
    # Boundary: we need min over all points
    # Only the moved point's boundary dist changed
    # So min_bound = min(old_min_bound_except_moved, moved_bound_dist)
    # We don't have old_min_bound cached, so recompute
    max_abs_all = np.max(np.abs(positions), axis=1)
    max_abs_all[moved_idx] = np.max(np.abs(new_pos))
    bound_dists_all = 0.5 - max_abs_all
    min_bound = np.min(bound_dists_all)
    
    r = min(min_bound, min_dist / 2.0)
    
    return r, True, new_min_dist_sq_cache

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin
    
    # 1. Initialization: Hexagonal Grid
    s = 1.0
    points = []
    for j in range(-15, 16):
        for i in range(-15, 16):
            x = i * s + (j % 2) * s / 2.0
            y = j * s * np.sqrt(3.0) / 2.0
            points.append([x, y])
            
    points = np.array(points)
    
    # Select N points closest to the origin
    dists_to_center = np.sum(points**2, axis=1)
    indices = np.argsort(dists_to_center)[:N]
    positions = points[indices].copy()
    
    # Scale to fit in [-0.5, 0.5]^2
    max_abs = np.max(np.abs(positions))
    if max_abs > 0:
        scale = 0.5 / max_abs
        positions = positions * scale
    
    # Add small noise to break symmetry
    r_init, _ = compute_radius(positions)
    noise_std = max(0.001, r_init * 0.01)
    positions += np.random.normal(0, noise_std, size=positions.shape)
    
    # Ensure inside after noise (clamp)
    positions = np.clip(positions, -0.4999, 0.4999)
    
    current_radius, _ = compute_radius(positions)
    best_radius = current_radius
    best_positions = positions.copy()
    
    # 2. Simulated Annealing
    T = 0.001
    T_end = 1e-7
    alpha = 0.999
    step_size = 0.01
    min_step_size = 0.0001
    max_step_size = 0.05
    
    accept_count = 0
    total_count = 0
    
    while time.time() < deadline - 20.0:  # Leave 20s for polish
        for _ in range(100):
            if time.time() >= deadline - 20.0:
                break
                
            total_count += 1
            idx = np.random.randint(N)
            
            if total_count % 100 == 0:
                accept_rate = accept_count / max(1, total_count)
                if accept_rate < 0.2:
                    step_size *= 1.2
                elif accept_rate > 0.8:
                    step_size *= 0.8
                step_size = np.clip(step_size, min_step_size, max_step_size)
                accept_count = 0
                total_count = 0
                
            delta = np.random.normal(0, step_size, size=2)
            new_pos = positions[idx].copy() + delta
            
            old_pos = positions[idx].copy()
            positions[idx] = new_pos
            new_radius, is_valid = compute_radius(positions)
            
            if not is_valid:
                new_radius = -1.0
                
            if new_radius > current_radius:
                current_radius = new_radius
                accept_count += 1
                if new_radius > best_radius:
                    best_radius = new_radius
                    best_positions = positions.copy()
            else:
                delta_r = new_radius - current_radius
                if delta_r > -T * 10:
                    prob = np.exp(delta_r / T)
                    if np.random.random() < prob:
                        current_radius = new_radius
                        accept_count += 1
                    else:
                        positions[idx] = old_pos
                else:
                    positions[idx] = old_pos
                    
        T *= alpha
        if T < T_end:
            T = T_end
    
    # 3. Final polish with Nelder-Mead on the best solution
    def objective(x):
        """Negative radius to minimize."""
        pos = x.reshape(N, 2)
        r, valid = compute_radius(pos)
        if not valid:
            return 1.0  # Large penalty
        return -r
    
    # Try multiple restarts of Nelder-Mead from slightly perturbed best positions
    polish_deadline = deadline - 1.0  # Leave 1s margin
    
    for trial in range(5):
        if time.time() >= polish_deadline:
            break
            
        # Perturb best positions slightly
        if trial == 0:
            polish_start = best_positions.copy()
        else:
            polish_start = best_positions + np.random.normal(0, best_radius * 0.005, size=best_positions.shape)
            polish_start = np.clip(polish_start, -0.4999, 0.4999)
        
        # Flatten for Nelder-Mead
        x0 = polish_start.flatten()
        
        try:
            result = minimize(
                objective,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            
            # Check if result improved
            polish_pos = result.x.reshape(N, 2)
            polish_radius, polish_valid = compute_radius(polish_pos)
            
            if polish_valid and polish_radius > best_radius:
                best_radius = polish_radius
                best_positions = polish_pos.copy()
                
        except Exception:
            pass
    
    # Final validation
    final_radius, final_valid = compute_radius(best_positions)
    if not final_valid or final_radius < 0:
        # Fallback: return a valid simple configuration
        # Use the hex grid initialization
        positions_hex = points[indices].copy()
        max_abs_hex = np.max(np.abs(positions_hex))
        if max_abs_hex > 0:
            positions_hex = positions_hex * (0.4999 / max_abs_hex)
        return positions_hex
    
    return best_positions
