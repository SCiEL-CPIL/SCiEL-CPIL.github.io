"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist, squareform

def _eval_packing(xy, r):
    """
    Evaluate a packing: returns (total_radius, is_valid).
    Valid means all circles inside [0,1]^2 and disjoint interiors.
    """
    n = len(xy)
    if n == 0:
        return 0.0, True
        
    # Check walls
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if x - ri < -1e-9 or 1.0 - x - ri < -1e-9 or y - ri < -1e-9 or 1.0 - y - ri < -1e-9:
            return 0.0, False
            
    # Check pairwise distances
    # Vectorized check for speed
    dists = squareform(pdist(xy))
    min_dists = r[:, np.newaxis] + r[np.newaxis, :]
    # If any distance is less than sum of radii, they overlap
    if np.any(dists < min_dists - 1e-9):
        return 0.0, False
        
    return np.sum(r), True

def _force_directed_relax(xy, r, deadline, max_iter=2000, learning_rate=0.01):
    """
    Physics-based relaxation.
    Forces:
    1. Repulsion between overlapping circles (push apart).
    2. Repulsion from walls (push in).
    3. Attraction to center (prevent drift, keep in bounds).
    
    We optimize radii by expanding them as much as possible while maintaining separation.
    Strategy: 
    - Iterate: 
      a. Compute forces on positions.
      b. Update positions.
      c. Try to increase radii slightly if no overlap.
    """
    n = len(xy)
    best_total = 0.0
    best_valid = False
    best_xy = xy.copy()
    best_r = r.copy()
    
    # Initial check
    total, valid = _eval_packing(xy, r)
    if valid:
        best_total = total
        best_valid = True

    lr = learning_rate
    for i in range(max_iter):
        if time.time() > deadline:
            break
            
        # 1. Compute Forces
        forces = np.zeros_like(xy)
        
        # Wall forces
        # Left wall: if x < r, push right
        forces[:, 0] += np.maximum(0, r - xy[:, 0])
        # Right wall: if x > 1-r, push left
        forces[:, 0] -= np.maximum(0, xy[:, 0] - (1.0 - r))
        # Bottom wall: if y < r, push up
        forces[:, 1] += np.maximum(0, r - xy[:, 1])
        # Top wall: if y > 1-r, push down
        forces[:, 1] -= np.maximum(0, xy[:, 1] - (1.0 - r))
        
        # Pairwise repulsion
        dists = squareform(pdist(xy))
        min_dists = r[:, np.newaxis] + r[np.newaxis, :]
        
        # Only consider pairs that are overlapping or very close
        # To save time, we can just compute forces for all pairs, it's O(N^2) which is fine for N=26
        for a in range(n):
            for b in range(a + 1, n):
                dx = xy[a, 0] - xy[b, 0]
                dy = xy[a, 1] - xy[b, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < 1e-6:
                    # Random nudge
                    dx = np.random.uniform(-0.01, 0.01)
                    dy = np.random.uniform(-0.01, 0.01)
                    dist = np.sqrt(dx*dx + dy*dy)
                
                overlap = min_dists[a, b] - dist
                if overlap > 0:
                    # Force is proportional to overlap
                    fx = (dx / dist) * overlap
                    fy = (dy / dist) * overlap
                    forces[a, 0] += fx
                    forces[a, 1] += fy
                    forces[b, 0] -= fx
                    forces[b, 1] -= fy
        
        # 2. Update Positions
        xy += lr * forces
        # Clip to valid region roughly to prevent explosion
        # We don't clip to r yet because r might be large, but keep in [0,1]
        xy = np.clip(xy, 0.0, 1.0)
        
        # 3. Try to grow radii
        # Check if current config is valid
        # We can try to increase all radii by a small epsilon
        # Or, smarter: for each circle, find max possible radius given current positions
        # But that's expensive. Let's just try a global growth step.
        
        # Calculate max possible growth
        # A circle i can grow by delta_i if:
        # delta_i <= min( x_i, 1-x_i, y_i, 1-y_i ) - r_i  (walls)
        # delta_i <= min_j ( dist_ij - r_i - r_j )      (neighbors)
        
        # Let's compute the slack for each circle
        slacks = np.zeros(n)
        for i in range(n):
            # Wall slack
            wall_slack = np.min([xy[i,0] - r[i], 1.0 - xy[i,0] - r[i], xy[i,1] - r[i], 1.0 - xy[i,1] - r[i]])
            slacks[i] = wall_slack
            
            # Neighbor slack
            for j in range(n):
                if i == j: continue
                dist = np.sqrt((xy[i,0]-xy[j,0])**2 + (xy[i,1]-xy[j,1])**2)
                neighbor_slack = dist - r[i] - r[j]
                slacks[i] = min(slacks[i], neighbor_slack)
        
        # If any slack is positive, we can grow.
        # To be stable, we take the min positive slack and grow all by a fraction of it?
        # Or grow each by its own slack? Growing each by its own slack might cause new overlaps if they grow towards each other.
        # Safer: find global min slack. If min_slack > 0, grow all radii by min_slack * 0.5
        min_slack = np.min(slacks)
        if min_slack > 1e-6:
            # Grow
            growth = min_slack * 0.5
            r += growth
        else:
            # No global growth possible. Try individual growth?
            # Or just stop growing if stuck.
            # For now, just keep radii constant and let positions adjust.
            pass
            
        # Validate and update best
        total, valid = _eval_packing(xy, r)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = xy.copy()
            best_r = r.copy()
            
        # Decay learning rate
        lr *= 0.995
        
    return best_xy, best_r, best_total, best_valid

def _random_init(seed=None):
    """
    Generate a random valid initial packing.
    Start with small circles and grow them?
    Or just place random positions with small radii.
    """
    if seed is not None:
        np.random.seed(seed)
        
    xy = np.random.rand(N, 2)
    # Small initial radii to ensure validity
    r = np.full(N, 0.01)
    return xy, r

def _grid_init(side):
    """Create a grid of up to N circles."""
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                # Initial radius: half the cell size minus a small margin
                r = 0.45 / side
                positions.append([x, y, r])
    # Pad if needed
    while len(positions) < N:
        positions.append([0.5, 0.5, 0.01])
    return np.array(positions[:N])

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_xy = None
    best_r = None
    best_total = 0.0
    best_valid = False
    
    # Strategy: Multi-start with Force-Directed Relaxation
    # We will try several random starts and one grid start.
    
    num_starts = 5
    # Reserve time for each start
    time_per_start = (deadline - start_time - 10.0) / num_starts
    
    for k in range(num_starts):
        if time.time() > deadline - 5.0:
            break
            
        current_deadline = min(deadline - 5.0, start_time + (k + 1) * time_per_start)
        
        if k == 0:
            # Grid init
            side = 5
            init = _grid_init(side)
            xy = init[:, :2].copy()
            r = init[:, 2].copy()
        else:
            # Random init
            xy, r = _random_init(seed=42 + k)
            
        # Run force-directed relaxation
        out_xy, out_r, total, valid = _force_directed_relax(xy, r, current_deadline, max_iter=1500, learning_rate=0.05)
        
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = out_xy
            best_r = out_r

    # If we still have time, do a final polish on the best found
    if best_valid and time.time() < deadline - 5.0:
        # Run one more relaxation with very small learning rate to settle
        p_xy, p_r, p_total, p_valid = _force_directed_relax(best_xy, best_r, deadline - 2.0, max_iter=500, learning_rate=0.01)
        if p_valid and p_total > best_total:
            best_total = p_total
            best_valid = True
            best_xy = p_xy
            best_r = p_r

    # Fallback if nothing worked
    if best_xy is None or not best_valid:
        side = int(np.ceil(np.sqrt(N)))
        init = _grid_init(side)
        best_xy = init[:, :2].copy()
        best_r = init[:, 2].copy()
        # Force validity
        for i in range(N):
            best_r[i] = min(best_r[i], best_xy[i,0], 1-best_xy[i,0], best_xy[i,1], 1-best_xy[i,1])
        
    # Final validation and clipping
    n = len(best_xy)
    result = np.zeros((n, 3))
    for i in range(n):
        x, y, ri = best_xy[i, 0], best_xy[i, 1], best_r[i]
        # Ensure circle is inside unit square
        ri = min(ri, x, 1.0 - x, y, 1.0 - y)
        ri = max(ri, 0.001)
        result[i, 2] = ri
        result[i, 0] = np.clip(x, ri, 1.0 - ri)
        result[i, 1] = np.clip(y, ri, 1.0 - ri)
    
    # Final check
    final_xy = result[:, :2]
    final_r = result[:, 2]
    total, valid = _eval_packing(final_xy, final_r)
    if not valid:
        # Fallback to simple grid
        side = int(np.ceil(np.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side

    return result
# EVOLVE-BLOCK-END
