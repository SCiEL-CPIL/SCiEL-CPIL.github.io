"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

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
    dists = np.sqrt(np.sum((xy[:, np.newaxis, :] - xy[np.newaxis, :, :]) ** 2, axis=-1))
    min_dists = r[:, np.newaxis] + r[np.newaxis, :]
    # If any distance is less than sum of radii, they overlap
    if np.any(dists < min_dists - 1e-9):
        return 0.0, False
        
    return np.sum(r), True

def _sa_packing(deadline):
    """
    Simulated Annealing for circle packing.
    State: positions (xy) and radii (r).
    Objective: Maximize sum(r) subject to constraints.
    Penalty method: Minimize E = -sum(r) + penalty * (violations).
    
    Moves:
    1. Move a circle (x, y).
    2. Change a radius (r).
    3. Swap two circles (if we were tracking labels, but here labels don't matter for sum, so swap is identity).
    4. "Grow all" move: Try to increase all radii by a small epsilon if possible.
    """
    # Initialize
    # Start with a grid to ensure a good starting point
    side = int(np.ceil(np.sqrt(N)))
    xy = np.zeros((N, 2))
    r = np.zeros(N)
    
    # Place in grid
    idx = 0
    for i in range(side):
        for j in range(side):
            if idx >= N: break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            xy[idx] = [x, y]
            # Initial radius: fit in cell
            r[idx] = 0.45 / side
            idx += 1
    
    # Pad if N is not a perfect square
    while idx < N:
        xy[idx] = [0.5, 0.5]
        r[idx] = 0.01
        idx += 1

    current_xy = xy.copy()
    current_r = r.copy()
    
    # Calculate initial energy
    # We use a penalty function for invalid states to allow the SA to navigate through invalid states
    # Energy = -sum(r) + lambda * (sum of overlaps + sum of wall violations)
    
    def get_energy(xy, r):
        total_r = np.sum(r)
        penalty = 0.0
        
        # Wall violations
        # x < r -> violation r - x
        penalty += np.sum(np.maximum(0, r - xy[:, 0]))
        penalty += np.sum(np.maximum(0, xy[:, 0] - (1.0 - r)))
        penalty += np.sum(np.maximum(0, r - xy[:, 1]))
        penalty += np.sum(np.maximum(0, xy[:, 1] - (1.0 - r)))
        
        # Pairwise overlaps
        # dists < r_i + r_j -> violation (r_i + r_j - dist)
        dists = np.sqrt(np.sum((xy[:, np.newaxis, :] - xy[np.newaxis, :, :]) ** 2, axis=-1))
        min_dists = r[:, np.newaxis] + r[np.newaxis, :]
        overlaps = np.maximum(0, min_dists - dists)
        # Avoid double counting
        penalty += np.sum(overlaps) * 0.5
        
        return -total_r + 10.0 * penalty

    current_energy = get_energy(current_xy, current_r)
    
    best_xy = current_xy.copy()
    best_r = current_r.copy()
    best_total = 0.0
    best_valid = False
    
    # Check initial validity
    total, valid = _eval_packing(current_xy, current_r)
    if valid:
        best_total = total
        best_valid = True
        best_xy = current_xy.copy()
        best_r = current_r.copy()

    T = 0.1
    T_min = 0.0001
    alpha = 0.995
    steps = 0
    
    while time.time() < deadline and T > T_min:
        # Generate a neighbor state
        new_xy = current_xy.copy()
        new_r = current_r.copy()
        
        move_type = random.random()
        
        if move_type < 0.6:
            # Move a circle
            i = random.randint(0, N - 1)
            step = 0.05 * T + 0.001
            new_xy[i, 0] += random.uniform(-step, step)
            new_xy[i, 1] += random.uniform(-step, step)
            # Keep in bounds roughly
            new_xy[i, 0] = np.clip(new_xy[i, 0], 0.01, 0.99)
            new_xy[i, 1] = np.clip(new_xy[i, 1], 0.01, 0.99)
            
        elif move_type < 0.8:
            # Change radius
            i = random.randint(0, N - 1)
            step = 0.02 * T + 0.001
            new_r[i] += random.uniform(-step, step)
            new_r[i] = np.clip(new_r[i], 0.001, 0.5)
            
        else:
            # Global growth attempt (if valid, try to grow all slightly)
            # This helps escape local minima where individual growth is blocked but global is not
            # Or just a random perturbation of all radii
            for i in range(N):
                new_r[i] += random.uniform(-0.005, 0.005)
                new_r[i] = np.clip(new_r[i], 0.001, 0.5)

        new_energy = get_energy(new_xy, new_r)
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or random.random() < np.exp(-delta_e / T):
            current_xy = new_xy
            current_r = new_r
            current_energy = new_energy
            
            # Check if this is a valid and better solution
            # We only check validity every few steps to save time, but for SA it's safer to check often
            # Given N=26, _eval_packing is fast enough.
            if steps % 10 == 0:
                total, valid = _eval_packing(current_xy, current_r)
                if valid and total > best_total:
                    best_total = total
                    best_valid = True
                    best_xy = current_xy.copy()
                    best_r = current_r.copy()
        
        T *= alpha
        steps += 1
        
    # Final polish: if the best found is valid, try to grow it a bit more using local relaxation
    if best_valid:
        # Simple local growth
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(N):
                # Try to grow circle i
                # Find max possible radius for i given others
                max_r = min(best_xy[i, 0], 1.0 - best_xy[i, 0], best_xy[i, 1], 1.0 - best_xy[i, 1])
                for j in range(N):
                    if i == j: continue
                    dist = np.sqrt((best_xy[i,0]-best_xy[j,0])**2 + (best_xy[i,1]-best_xy[j,1])**2)
                    max_r = min(max_r, dist - best_r[j])
                
                if max_r > best_r[i] + 1e-5:
                    best_r[i] = max_r
                    improved = True
            
            # Check validity after growth
            total, valid = _eval_packing(best_xy, best_r)
            if not valid:
                # If invalid, revert to previous best? 
                # Actually, the growth logic above ensures validity IF the other circles are fixed.
                # But growing one circle might push it into another if my dist calc was slightly off or floating point.
                # Let's just re-evaluate. If invalid, we stop polishing.
                break
            else:
                best_total = total

    return best_xy, best_r, best_total, best_valid

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_xy = None
    best_r = None
    best_total = 0.0
    best_valid = False
    
    # Run SA multiple times with different seeds/temps
    # Since SA is stochastic, multiple runs help.
    # We have ~140s. Let's do 3 runs of ~40s each.
    
    num_runs = 3
    time_per_run = (deadline - start_time) / num_runs
    
    for k in range(num_runs):
        if time.time() > deadline - 5.0:
            break
        
        current_deadline = min(deadline - 5.0, start_time + (k + 1) * time_per_run)
        
        # Seed random for reproducibility of the run itself? No, let it be random.
        # But we can vary the initial temperature or start.
        
        out_xy, out_r, total, valid = _sa_packing(current_deadline)
        
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = out_xy
            best_r = out_r

    # Fallback
    if best_xy is None or not best_valid:
        side = int(np.ceil(np.sqrt(N)))
        best_xy = np.zeros((N, 2))
        best_r = np.zeros(N)
        idx = 0
        for i in range(side):
            for j in range(side):
                if idx >= N: break
                best_xy[idx] = [(i + 0.5) / side, (j + 0.5) / side]
                best_r[idx] = 0.45 / side
                idx += 1
        while idx < N:
            best_xy[idx] = [0.5, 0.5]
            best_r[idx] = 0.01
            idx += 1
            
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
