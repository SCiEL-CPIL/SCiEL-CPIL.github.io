"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time

N = 26

# EVOLVE-BLOCK-START

def _compute_forces(pos, radii, dt=0.1):
    """
    Compute forces on each circle to resolve overlaps and boundary violations.
    Returns forces array (N, 2).
    """
    forces = np.zeros((N, 2))
    
    # 1. Pairwise repulsion
    # Vectorized distance calculation
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    dist = np.sqrt(dist_sq + 1e-12) # Avoid div by zero
    
    # Minimum required distance
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    
    # Overlap amount: positive if overlapping
    overlap = min_dist - dist
    overlap = np.maximum(0, overlap)
    
    # Force magnitude: proportional to overlap
    # We want to push i away from j if i overlaps j.
    # Force on i from j is in direction (pos_i - pos_j) / dist * overlap
    # F_ij = (dx, dy) / dist * overlap
    # Sum over j for each i.
    
    # Unit vectors
    unit_x = dx / dist
    unit_y = dy / dist
    
    # Forces from overlaps
    # force_x[i] = sum_j (unit_x[i,j] * overlap[i,j])
    # Note: overlap is symmetric, unit vectors are antisymmetric.
    # We sum over all j != i.
    
    fx = np.sum(unit_x * overlap, axis=1)
    fy = np.sum(unit_y * overlap, axis=1)
    
    # Subtract self terms (should be 0 anyway due to dist>0, but overlap[i,i]=0)
    forces[:, 0] = fx
    forces[:, 1] = fy
    
    # 2. Boundary forces
    # Push inward if touching or crossing boundary
    # Left wall: x - r < 0 -> push right (positive x)
    # Right wall: x + r > 1 -> push left (negative x)
    # Bottom wall: y - r < 0 -> push up (positive y)
    # Top wall: y + r > 1 -> push down (negative y)
    
    # Strength of boundary force can be stronger to keep them inside
    # Let's use a strong penalty force
    boundary_strength = 2.0
    
    # Left
    viol_left = np.maximum(0, radii - pos[:, 0])
    forces[:, 0] += boundary_strength * viol_left
    
    # Right
    viol_right = np.maximum(0, pos[:, 0] + radii - 1.0)
    forces[:, 0] -= boundary_strength * viol_right
    
    # Bottom
    viol_bottom = np.maximum(0, radii - pos[:, 1])
    forces[:, 1] += boundary_strength * viol_bottom
    
    # Top
    viol_top = np.maximum(0, pos[:, 1] + radii - 1.0)
    forces[:, 1] -= boundary_strength * viol_top
    
    return forces

def _force_directed_packing(pos, radii, time_limit=10.0, rng=None):
    """
    Optimize packing using force-directed relaxation and radius growth.
    """
    if rng is None:
        rng = np.random.RandomState(42)
        
    start_time = time.time()
    pos = pos.copy()
    radii = radii.copy()
    
    # Parameters
    dt = 0.05 # Step size for position update
    growth_rate = 0.001 # How much to grow radii each "macro" step
    cooling = 0.999 # Cool down dt over time
    min_dt = 0.001
    
    # Initial validation: ensure we start inside and non-overlapping if possible
    # If starting from a bad config, the forces will fix it.
    
    best_score = np.sum(radii)
    best_pos = pos.copy()
    best_radii = radii.copy()
    
    while time.time() - start_time < time_limit:
        # 1. Calculate forces
        f = _compute_forces(pos, radii)
        
        # 2. Update positions
        # Simple Euler integration with damping
        # pos += dt * f
        # To prevent oscillations, we can use a smaller effective dt or damping
        # Let's just use the force directly as a displacement, scaled by dt
        pos += dt * f
        
        # 3. Clamp positions to [0, 1] roughly to prevent flying out
        # The boundary forces should keep them in, but let's be safe
        pos[:, 0] = np.clip(pos[:, 0], 0.0, 1.0)
        pos[:, 1] = np.clip(pos[:, 1], 0.0, 1.0)
        
        # 4. Grow radii
        # We grow all radii uniformly. This is a greedy strategy.
        # We might need to check if growth causes massive overlap, 
        # but the next force step will correct positions.
        radii += growth_rate
        
        # 5. Check validity and update best
        # For efficiency, we don't do a full O(N^2) sqrt check every single micro-step
        # but we can do a quick check.
        # Let's do a periodic full check.
        
        # Simple heuristic: if total overlap is low, try to update best
        # Actually, let's just update best if the configuration looks valid.
        # We can approximate validity by checking max overlap.
        
        # Periodic full validation
        if int((time.time() - start_time) * 10) % 5 == 0: # Every ~0.5s
            # Check boundaries
            if np.all(pos[:, 0] - radii >= -1e-8) and np.all(pos[:, 0] + radii <= 1.0 + 1e-8) and \
               np.all(pos[:, 1] - radii >= -1e-8) and np.all(pos[:, 1] + radii <= 1.0 + 1e-8):
                # Check overlaps
                dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
                dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
                dist = np.sqrt(dx*dx + dy*dy + 1e-12)
                min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
                mask = np.triu(np.ones((N, N), dtype=bool), k=1)
                if np.all(dist[mask] >= min_dist[mask] - 1e-8):
                    score = np.sum(radii)
                    if score > best_score:
                        best_score = score
                        best_pos = pos.copy()
                        best_radii = radii.copy()
                        
        # Cool down step size to allow convergence
        dt *= cooling
        if dt < min_dt:
            dt = min_dt
            # If dt is too small, we might be stuck. 
            # We could add a small random jitter to escape local minima
            if rng.random() < 0.1:
                pos += rng.normal(0, 0.01, pos.shape)
                pos[:, 0] = np.clip(pos[:, 0], 0.0, 1.0)
                pos[:, 1] = np.clip(pos[:, 1], 0.0, 1.0)

    return best_pos, best_radii

def _enforce_validity(pos, radii):
    """
    Greedily shrink radii to ensure validity.
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

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_pos = None
    best_radii = None
    best_score = 0.0
    
    rng = np.random.RandomState(42)
    
    n_starts = 5
    time_per_start = (deadline - start_time) / (n_starts * 1.2) # Leave buffer
    
    for s in range(n_starts):
        if time.time() > deadline - 5:
            break
            
        # Initialize
        if s == 0:
            # Grid init
            pos = np.zeros((N, 2))
            radii = np.zeros(N)
            k = 0
            for i in range(5):
                for j in range(5):
                    if k < N:
                        pos[k] = [(i + 0.5) / 5.0, (j + 0.5) / 5.0]
                        radii[k] = 0.01 # Start small
                        k += 1
            if k < N:
                pos[k] = [0.5, 0.5]
                radii[k] = 0.01
        else:
            # Random init
            pos = rng.uniform(0.1, 0.9, (N, 2))
            radii = np.full(N, 0.01)
            
        # Run force-directed packing
        pos_fd, rad_fd = _force_directed_packing(pos, radii, time_limit=time_per_start, rng=rng)
        
        # Enforce validity to ensure we return a valid config
        pos_valid, rad_valid = _enforce_validity(pos_fd, rad_fd)
        
        score = np.sum(rad_valid)
        if score > best_score:
            best_score = score
            best_pos = pos_valid
            best_radii = rad_valid
            
        # Early stop if we hit a very high score
        if best_score > 2.5:
            break
            
    # Fallback
    if best_pos is None:
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        k = 0
        for i in range(5):
            for j in range(5):
                if k < N:
                    pos[k] = [(i + 0.5) / 5.0, (j + 0.5) / 5.0]
                    radii[k] = 0.01
                    k += 1
        if k < N:
            pos[k] = [0.5, 0.5]
            radii[k] = 0.01
        best_pos, best_radii = _force_directed_packing(pos, radii, time_limit=10.0, rng=rng)
        best_pos, best_radii = _enforce_validity(best_pos, best_radii)
        
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
