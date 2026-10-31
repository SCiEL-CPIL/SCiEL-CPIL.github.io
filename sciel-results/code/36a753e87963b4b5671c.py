"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _check_validity(pos, radii, tol=1e-8):
    """
    Check if the configuration is valid.
    Returns True if all circles are within [0,1]^2 and disjoint.
    """
    n = len(radii)
    # Check bounds
    if np.any(pos[:, 0] < -tol) or np.any(pos[:, 0] > 1.0 + tol):
        return False
    if np.any(pos[:, 1] < -tol) or np.any(pos[:, 1] > 1.0 + tol):
        return False
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
    
    # dist < min_dist => dist_sq < min_dist^2
    if np.any(dist_sq[mask] < (min_dist[mask] - tol)**2):
        return False
        
    return True

def _shrink_to_valid(pos, radii, tol=1e-6):
    """
    Shrink radii and adjust positions slightly to ensure validity.
    This is a safety net.
    """
    pos = pos.copy()
    radii = radii.copy()
    
    # Iterative shrinking
    for _ in range(50):
        valid = True
        
        # Check bounds
        for i in range(N):
            x, y = pos[i]
            r = radii[i]
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            if r > max_r + tol:
                radii[i] = max(0.0, max_r - tol)
                valid = False
        
        # Check overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r - tol:
                    # Shrink both proportionally to fit
                    # dist >= r_i + r_j
                    # We want new_r_i + new_r_j <= dist
                    # Let new_r_i = k * r_i, new_r_j = k * r_j
                    # k * (r_i + r_j) <= dist => k <= dist / (r_i + r_j)
                    k = max(0.0, dist / (sum_r + 1e-12))
                    k = min(k, 1.0) # Only shrink
                    radii[i] = radii[i] * k
                    radii[j] = radii[j] * k
                    valid = False
        
        if valid:
            break
            
    return pos, radii

def _md_packing(init_pos, init_radii, steps=5000, seed=0):
    """
    Molecular Dynamics simulation to pack circles.
    """
    rng = np.random.RandomState(seed)
    pos = init_pos.copy().astype(float)
    radii = init_radii.copy().astype(float)
    
    # Parameters
    dt = 0.01
    damping = 0.9
    force_strength = 10.0
    
    best_score = np.sum(radii)
    best_pos = pos.copy()
    best_radii = radii.copy()
    
    for step in range(steps):
        # Calculate forces
        forces = np.zeros_like(pos)
        
        # Pairwise repulsion
        dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
        dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
        dist_sq = dx * dx + dy * dy
        # Avoid division by zero
        dist_sq[dist_sq == 0] = 1e-12
        dist = np.sqrt(dist_sq)
        
        min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
        
        # Overlap amount: min_dist - dist
        overlap = min_dist - dist
        overlap[overlap < 0] = 0.0
        
        # Force magnitude is proportional to overlap
        # F = k * overlap
        # Direction is normalized (dx, dy)
        norm_dist = dist
        norm_dist[norm_dist == 0] = 1.0
        
        fx = force_strength * overlap * (dx / norm_dist)
        fy = force_strength * overlap * (dy / norm_dist)
        
        # Sum forces for each circle
        # We need to sum over the other dimension
        # fx[i, j] is force on i due to j? 
        # dx = x_i - x_j. If overlap > 0, force on i should be in direction (x_i - x_j) to push away from j.
        # So force on i is sum of fx[i, :].
        forces[:, 0] += np.sum(fx, axis=1)
        forces[:, 1] += np.sum(fy, axis=1)
        
        # Wall repulsion
        # Left wall: x < r => force +x
        overlap_left = radii - pos[:, 0]
        overlap_left[overlap_left < 0] = 0
        forces[:, 0] += force_strength * overlap_left
        
        # Right wall: x > 1-r => force -x
        overlap_right = pos[:, 0] - (1.0 - radii)
        overlap_right[overlap_right < 0] = 0
        forces[:, 0] -= force_strength * overlap_right
        
        # Bottom wall: y < r => force +y
        overlap_bottom = radii - pos[:, 1]
        overlap_bottom[overlap_bottom < 0] = 0
        forces[:, 1] += force_strength * overlap_bottom
        
        # Top wall: y > 1-r => force -y
        overlap_top = pos[:, 1] - (1.0 - radii)
        overlap_top[overlap_top < 0] = 0
        forces[:, 1] -= force_strength * overlap_top
        
        # Update velocities/positions
        # Simple Euler integration with damping
        # v = v * damping + F * dt
        # We don't store velocity explicitly, we use forces directly as velocity proxy
        # pos += forces * dt
        # To prevent explosion, we can clamp forces or use adaptive dt
        
        # Adaptive step size: if total force is large, reduce dt
        total_force = np.sum(np.abs(forces))
        current_dt = dt / (1.0 + total_force * 0.1)
        
        pos += forces * current_dt
        
        # Clamp positions to [0, 1] roughly to prevent flying away
        pos[:, 0] = np.clip(pos[:, 0], -0.5, 1.5)
        pos[:, 1] = np.clip(pos[:, 1], -0.5, 1.5)
        
        # Try to grow radii if possible
        # Calculate max possible radius for each circle given current positions and other radii
        # This is a local greedy growth
        for i in range(N):
            max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            for j in range(N):
                if i == j:
                    continue
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                max_r = min(max_r, dist - radii[j])
            
            # Grow towards max_r, but slowly
            target_r = max_r
            if target_r > radii[i]:
                # Grow by a fraction of the gap
                radii[i] += (target_r - radii[i]) * 0.1
            else:
                # If it shrank due to overlap, keep it
                pass
            
            radii[i] = max(radii[i], 1e-4)
        
        # Check validity and update best
        if step % 10 == 0:
            # A quick validity check is expensive, so we do a "soft" check or just sum radii
            # If we use the forces, overlaps are being pushed out.
            # We can check if the "energy" (sum of overlaps) is low.
            # For now, let's just track sum of radii, but ensure we don't return invalid.
            # We will do a strict check at the end.
            score = np.sum(radii)
            if score > best_score:
                # Quick sanity check: are we exploding?
                if np.all(pos[:, 0] > -0.1) and np.all(pos[:, 0] < 1.1) and \
                   np.all(pos[:, 1] > -0.1) and np.all(pos[:, 1] < 1.1):
                    best_score = score
                    best_pos = pos.copy()
                    best_radii = radii.copy()
                    
    return best_pos, best_radii

def _objective(x, penalty_weight=1000.0):
    """
    Objective function for final polishing.
    x = [x1, y1, ..., xN, yN, r1, ..., rN]
    """
    pos = x[:2*N].reshape(N, 2)
    radii = np.maximum(x[2*N:], 1e-6)
    
    # Boundary violations
    bv = np.maximum(0.0, radii - pos[:, 0]) + np.maximum(0.0, pos[:, 0] - (1.0 - radii)) + \
         np.maximum(0.0, radii - pos[:, 1]) + np.maximum(0.0, pos[:, 1] - (1.0 - radii))
    boundary_pen = np.sum(bv * bv)
    
    # Overlap violations
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    min_dist_sq = min_dist * min_dist
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    viol = np.maximum(0.0, min_dist_sq - dist_sq)
    overlap_pen = np.sum(viol[mask] * viol[mask])
    
    return -np.sum(radii) + penalty_weight * (boundary_pen + overlap_pen)

def _polish(pos, radii, deadline):
    """
    Final high-precision polishing using L-BFGS-B.
    """
    x = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((1e-4, 0.5))
        
    try:
        result = minimize(_objective, x, method='L-BFGS-B', bounds=bounds,
                          args=(1000.0,),
                          options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12})
        if result.fun < _objective(x, 1000.0) - 1e-12:
            x = result.x
    except Exception:
        pass
    
    pos_out = x[:2*N].reshape(N, 2)
    radii_out = np.maximum(x[2*N:], 1e-4)
    
    return pos_out, radii_out

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 20s safety margin
    
    best_pos = None
    best_radii = None
    best_score = 0.0
    
    # Strategy: Run MD from multiple initializations
    # 1. Grid initialization
    # 2. Random initialization
    
    inits = []
    
    # Grid Init
    pos_grid = np.zeros((N, 2))
    rad_grid = np.zeros(N)
    # 5x5 grid is 25, need 26. 
    # Let's do 5x5 grid and one in the center? Or just 5x5 and add one small?
    # Better: 5x5 grid of larger circles, then optimize.
    spacing = 1.0 / 5.0
    for i in range(5):
        for j in range(5):
            idx = i * 5 + j
            if idx < N:
                pos_grid[idx] = [(i + 0.5) * spacing, (j + 0.5) * spacing]
                rad_grid[idx] = spacing * 0.49 # Slightly smaller to allow movement
    if N > 25:
        # Add 26th circle in the middle of a cell? No, that overlaps.
        # Just place it randomly for now, MD will sort it out.
        pos_grid[25] = [0.5, 0.5]
        rad_grid[25] = 0.05
    inits.append((pos_grid, rad_grid))
    
    # Random Init
    rng = np.random.RandomState(42)
    pos_rand = rng.uniform(0.1, 0.9, (N, 2))
    rad_rand = rng.uniform(0.05, 0.1, N)
    inits.append((pos_rand, rad_rand))
    
    # Hexagonal Init
    pos_hex = np.zeros((N, 2))
    rad_hex = np.zeros(N)
    r = 0.12
    dx = r * np.sqrt(3)
    dy = r * 1.5
    k = 0
    for row in range(6):
        offset = dx / 2 if row % 2 == 1 else 0
        y = row * dy + r
        for col in range(5):
            if k >= N:
                break
            x = r + offset + col * dx
            if 0 <= x <= 1 and 0 <= y <= 1:
                pos_hex[k] = [x, y]
                rad_hex[k] = r
                k += 1
        if k >= N:
            break
    while k < N:
        pos_hex[k] = [0.5, 0.5]
        rad_hex[k] = 0.01
        k += 1
    inits.append((pos_hex, rad_hex))
    
    for idx, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 30:
            break
            
        # Run MD
        pos_md, rad_md = _md_packing(pos_init, rad_init, steps=3000, seed=idx*100)
        
        # Polish
        if time.time() < deadline - 10:
            pos_pol, rad_pol = _polish(pos_md, rad_md, deadline)
            
            # Validate and shrink
            pos_val, rad_val = _shrink_to_valid(pos_pol, rad_pol)
            
            if _check_validity(pos_val, rad_val):
                score = np.sum(rad_val)
                if score > best_score:
                    best_score = score
                    best_pos = pos_val.copy()
                    best_radii = rad_val.copy()
                    
    # If no valid solution found, fallback to a simple grid
    if best_pos is None:
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        for i in range(5):
            for j in range(5):
                if i*5+j < N-1:
                    pos[i*5+j] = [(i+0.5)/5.0, (j+0.5)/5.0]
                    radii[i*5+j] = 0.09
        pos[N-1] = [0.5, 0.02]
        radii[N-1] = 0.01
        best_pos, best_radii = _shrink_to_valid(pos, radii)
        
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
