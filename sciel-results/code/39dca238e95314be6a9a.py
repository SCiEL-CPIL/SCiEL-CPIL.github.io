"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time

def _pbd_step(pos, radii, dt, iterations=5):
    """
    Position-Based Dynamics step to resolve collisions and boundaries.
    """
    n = len(pos)
    pos = pos.copy()
    
    # Precompute indices for pairs
    idx_i, idx_j = np.triu_indices(n, k=1)
    
    for _ in range(iterations):
        # 1. Boundary constraints
        # Project points into [r, 1-r]
        # Note: if 2*r > 1, this is impossible, but we assume r < 0.5
        # We clip x, y to be at least r and at most 1-r
        # However, simple clipping can cause issues if circles are large.
        # A better way: 
        # If x < r: x = r
        # If x > 1-r: x = 1-r
        # Same for y.
        
        pos[:, 0] = np.clip(pos[:, 0], radii, 1 - radii)
        pos[:, 1] = np.clip(pos[:, 1], radii, 1 - radii)
        
        # 2. Circle-circle constraints
        # For each pair, if dist < r_i + r_j, push apart.
        diff = pos[idx_i] - pos[idx_j]
        dist = np.linalg.norm(diff, axis=1)
        
        # Avoid division by zero
        safe_dist = np.where(dist > 1e-8, dist, 1e-8)
        
        r_sum = radii[idx_i] + radii[idx_j]
        
        # Violation: amount they overlap
        viol = r_sum - dist
        
        # Only correct if overlapping
        active = viol > 0
        
        if np.any(active):
            # Unit vector from j to i
            # diff is pos[i] - pos[j]
            # We want to move i away from j and j away from i
            # Correction vector for i: (diff / dist) * (viol / 2) * dt?
            # In PBD, we usually apply full correction or weighted by mass.
            # Assuming equal mass, split correction 50/50.
            
            # Correction magnitude
            corr_mag = viol[active] / 2.0
            
            # Unit vectors
            unit_diff = diff[active] / safe_dist[active, None]
            
            # Displacements
            disp = unit_diff * corr_mag[:, None]
            
            # Apply to i and j
            # i moves away from j (along diff)
            # j moves away from i (along -diff)
            
            # Accumulate updates
            # Using np.add.at for efficiency
            np.add.at(pos, idx_i[active], disp)
            np.add.at(pos, idx_j[active], -disp)
            
            # Re-clip boundaries after circle correction, as they might push out
            pos[:, 0] = np.clip(pos[:, 0], radii, 1 - radii)
            pos[:, 1] = np.clip(pos[:, 1], radii, 1 - radii)

    return pos

def _grow_radii(pos, radii, target_radius, step=0.001, max_iters=2000):
    """
    Gradually increase radii to target_radius while maintaining validity via PBD.
    """
    n = len(pos)
    current_r = radii.copy()
    
    # We want to increase all radii proportionally or uniformly?
    # Uniform increase is simpler.
    # Start from current_r, increase to target_radius.
    
    # If current_r is already >= target, return
    if current_r[0] >= target_radius:
        return pos, current_r
        
    # Determine number of steps
    # total_increase = target_radius - current_r[0]
    # steps = int(total_increase / step)
    # But we have a time limit. Let's just iterate until we hit target or time runs out.
    
    # Actually, let's just try to increase radius by a small amount, relax, and check.
    # If relaxation fails (e.g., circles pushed out of bounds significantly or overlap persists), 
    # we might need to backtrack or accept the current state.
    # For simplicity, we assume PBD handles it well enough.
    
    # Let's define a schedule of target radii.
    # r starts at current_r[0], ends at target_radius.
    
    # To be safe, let's do incremental steps.
    # Step size in radius
    dr = step
    
    while current_r[0] < target_radius - 1e-6:
        # Increase radius
        new_r = np.minimum(current_r + dr, target_radius)
        
        # Relax positions with new radii
        pos = _pbd_step(pos, new_r, dt=1.0, iterations=10)
        
        # Check if solution is still "valid" enough (no massive violations)
        # If PBD fails to resolve, the circles might be stuck.
        # We can check the max overlap.
        idx_i, idx_j = np.triu_indices(n, k=1)
        diff = pos[idx_i] - pos[idx_j]
        dist = np.linalg.norm(diff, axis=1)
        r_sum = new_r[idx_i] + new_r[idx_j]
        max_overlap = np.max(r_sum - dist)
        
        # Also check boundary
        min_boundary_dist = np.min(np.minimum(pos[:, 0], pos[:, 1]))
        max_boundary_dist = np.max(np.maximum(pos[:, 0], pos[:, 1]))
        # Boundary violation: min(r - x, r - y, x - (1-r), y - (1-r))
        # If any is negative, it's a violation.
        # Since we clip, boundary should be satisfied.
        
        if max_overlap > 1e-3:
            # If overlap is too large, maybe we grew too fast or got stuck.
            # Backtrack? Or just accept and stop growing?
            # Let's accept the previous valid state if this one is bad.
            # But for now, let's just break and return the last good one.
            # Actually, PBD should fix it. If it doesn't, the packing is likely full.
            break
        
        current_r = new_r

    return pos, current_r

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = N
    best_score = -1.0
    best_circles = None
    
    # Multi-start strategies
    starts = []
    
    # 1. Hexagonal grid initialization
    # Approximate hex grid for 26 circles
    # 5 rows: 5, 5, 5, 5, 6?
    rows = [5, 5, 5, 5, 6]
    spacing = 1.0 # Will scale
    pts = []
    for i, count in enumerate(rows):
        offset = (i % 2) * 0.5
        for j in range(count):
            x = j + offset
            y = i * np.sqrt(3)/2
            pts.append([x, y])
    pts = np.array(pts[:n])
    
    # Normalize to fit in [0, 1] with some margin
    min_pt = pts.min(axis=0)
    max_pt = pts.max(axis=0)
    extent = max(max_pt[0] - min_pt[0], max_pt[1] - min_pt[1])
    
    # Scale to fit in square of side 0.9 (leaving 0.05 margin)
    scale = 0.9 / extent
    pts = (pts - min_pt) * scale
    pts += 0.05
    
    # Initial radius: based on min distance between points / 2
    # Calculate min distance
    idx_i, idx_j = np.triu_indices(n, k=1)
    diff = pts[idx_i] - pts[idx_j]
    dists = np.linalg.norm(diff, axis=1)
    min_dist = np.min(dists)
    init_r_hex = min_dist / 2 * 0.9 # 90% of max possible
    
    starts.append((pts, init_r_hex))
    
    # 2. Square grid
    side = int(np.ceil(np.sqrt(n)))
    pts_sq = np.array([[(i + 0.5)/side, (j + 0.5)/side] for i in range(side) for j in range(side)])[:n]
    # Calculate min dist
    idx_i, idx_j = np.triu_indices(n, k=1)
    diff = pts_sq[idx_i] - pts_sq[idx_j]
    dists = np.linalg.norm(diff, axis=1)
    min_dist = np.min(dists)
    init_r_sq = min_dist / 2 * 0.9
    starts.append((pts_sq, init_r_sq))
    
    # 3. Random
    np.random.seed(42)
    pts_rand = np.random.rand(n, 2) * 0.8 + 0.1
    idx_i, idx_j = np.triu_indices(n, k=1)
    diff = pts_rand[idx_i] - pts_rand[idx_j]
    dists = np.linalg.norm(diff, axis=1)
    min_dist = np.min(dists)
    init_r_rand = min_dist / 2 * 0.9
    starts.append((pts_rand, init_r_rand))
    
    for idx, (init_pos, init_r) in enumerate(starts):
        if time.time() > deadline:
            break
            
        pos = init_pos.copy()
        radii = np.full(n, init_r)
        
        # Grow radii
        # Target radius: ideally 0.5/sqrt(n) ~ 0.1. But packing density limits this.
        # Hex packing density ~ 0.9069.
        # Area of n circles = n * pi * r^2.
        # Area of square = 1.
        # n * pi * r^2 <= 0.9069 * 1
        # r^2 <= 0.9069 / (n * pi)
        # r <= sqrt(0.9069 / (26 * pi)) ~ sqrt(0.01107) ~ 0.105
        # So max possible sum of radii ~ 26 * 0.105 ~ 2.73.
        # Let's try to grow up to 0.15, but PBD will stop if it can't fit.
        
        # We will try to grow in steps.
        # If time allows, try multiple growth targets.
        
        # Let's just grow as much as possible.
        # We'll run _grow_radii with a high target, and it will stop when it can't fit.
        
        # To be efficient, we can use a binary search on the radius or just linear growth.
        # Linear growth with small steps is robust.
        
        pos, radii = _grow_radii(pos, radii, target_radius=0.2, step=0.002, max_iters=1000)
        
        # Check validity and score
        # Validity check:
        valid = True
        for i in range(n):
            x, y, r = pos[i][0], pos[i][1], radii[i]
            if r <= 0:
                valid = False
                break
            if x < r - 1e-9 or x > 1 - r + 1e-9 or y < r - 1e-9 or y > 1 - r + 1e-9:
                valid = False
                break
        
        if valid:
            for i in range(n):
                for j in range(i + 1, n):
                    dx = pos[i][0] - pos[j][0]
                    dy = pos[i][1] - pos[j][1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < radii[i] + radii[j] - 1e-9:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            score = np.sum(radii)
            if score > best_score:
                best_score = score
                best_circles = np.column_stack([pos, radii]).copy()
                
    # If no valid circles found, fallback
    if best_circles is None:
        side = int(np.ceil(np.sqrt(n)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 0.01) for i in range(side) for j in range(side)]
        best_circles = np.array(cells[:n])
        
    return best_circles
# EVOLVE-BLOCK-END
