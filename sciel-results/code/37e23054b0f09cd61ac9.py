"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _score(pts):
    """Compute the maximum radius for the given points."""
    # Wall clearance
    abs_pts = np.abs(pts)
    max_abs = np.max(abs_pts, axis=1)
    wall_clearances = 0.5 - max_abs
    min_wall = np.min(wall_clearances)
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, np.inf)
    min_dist_sq = np.min(dists_sq)
    min_dist = np.sqrt(min_dist_sq) / 2.0
    
    return min(min_wall, min_dist)

def _energy(pts, r_target):
    """
    Compute a smooth energy function to minimize.
    We want to maximize min(dist/2, wall_dist).
    Equivalent to minimizing a function that penalizes violations of dist >= 2*r and wall_dist >= r.
    A common approach is to minimize sum of (1/d)^p for pairs and (1/wall)^p for walls,
    but this can be noisy.
    
    Alternative: Minimize the "inverse" of the minimum distance? No, non-differentiable.
    
    Let's use a potential energy approach:
    E = sum_{i<j} (1 / d_ij)^k + sum_i (1 / w_i)^k
    where d_ij is distance between centers, w_i is distance to wall.
    Minimizing this pushes points apart and away from walls.
    """
    # To avoid division by zero, add a small epsilon
    eps = 1e-6
    
    # Pairwise
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, np.inf)
    
    # We only care about pairs that are "close" or potentially violating?
    # For SA, we need a global energy.
    # Let's use a power law. k=4 is a stiff repulsion.
    # E_pair = sum (1/d)^4
    # But summing over all pairs is O(N^2). N=50 is fine.
    
    # We want to avoid 1/0.
    # If d is very small, energy is huge.
    
    # Actually, a better energy for packing is often related to the minimum distance.
    # But SA works well with smooth potentials.
    
    # Let's define energy as:
    # E = sum_{i<j} (1/d_ij^2) + sum_i (1/w_i^2)
    # This favors large minimum distances.
    
    # Wait, if we minimize sum of 1/d^2, we maximize the harmonic mean of distances?
    # Not exactly, but it's a valid landscape.
    
    # Let's try: E = sum_{i<j} max(0, 1/d_ij - 1/r_target)^2 ? No, non-smooth.
    
    # Standard Lennard-Jones-like repulsion only:
    # E = sum (sigma/d)^12
    # Let's just use 1/d^2 for simplicity and stability.
    
    # Pairwise energy
    # dists_sq is (N, N)
    # We want sum of 1/dists_sq for i<j
    # Since dists_sq is symmetric and diagonal is inf, we can sum upper triangle
    # But 1/inf = 0.
    # So E_pair = 0.5 * sum(1/dists_sq) - 0.5 * sum(diag(1/dists_sq))
    # Diagonal is inf, so 1/inf = 0.
    # So E_pair = 0.5 * np.sum(1.0 / dists_sq)
    
    # Wall energy
    abs_pts = np.abs(pts)
    max_abs = np.max(abs_pts, axis=1)
    w = 0.5 - max_abs
    # If w is close to 0, 1/w^2 is huge.
    # Add epsilon to w to prevent inf
    w_safe = np.maximum(w, eps)
    E_wall = np.sum(1.0 / (w_safe * w_safe))
    
    E_pair = 0.5 * np.sum(1.0 / dists_sq)
    
    return E_pair + E_wall

def _sa_optimize(pts, time_budget=10.0):
    """
    Simulated Annealing to minimize energy.
    """
    start_time = time.time()
    current_pts = pts.copy()
    current_energy = _energy(current_pts)
    
    best_pts = current_pts.copy()
    best_energy = current_energy
    best_score = _score(best_pts)
    
    T = 0.01  # Initial temperature
    T_min = 1e-5
    cooling_rate = 0.999
    
    # We will perform steps until time runs out or T < T_min
    # Step size should be adaptive or fixed small?
    # Fixed small step with random direction.
    step_size = 0.01
    
    while time.time() - start_time < time_budget:
        # Pick a random point
        i = random.randint(0, N - 1)
        
        # Generate random perturbation
        delta = np.random.normal(0, step_size, 3)
        new_pts = current_pts.copy()
        new_pts[i] += delta
        
        # Check bounds
        if np.any(np.abs(new_pts[i]) > 0.5):
            # Reflect or reject? Reject is simpler.
            continue
            
        new_energy = _energy(new_pts)
        
        # Acceptance criterion
        if new_energy < current_energy:
            current_pts = new_pts
            current_energy = new_energy
        else:
            # Metropolis criterion
            delta_E = new_energy - current_energy
            if T > 0:
                prob = np.exp(-delta_E / T)
                if random.random() < prob:
                    current_pts = new_pts
                    current_energy = new_energy
                    # Even if accepted, it's worse, so don't update best
                # else reject
            
        # Update best
        # Checking score every iteration is expensive (O(N^2) for dists).
        # Energy is also O(N^2).
        # We can check score less frequently or use energy as proxy?
        # Energy is a proxy for "goodness" but not exactly score.
        # However, low energy usually correlates with high score.
        # Let's check score every 100 iterations or if energy improves significantly?
        # Actually, _score is also O(N^2) with sqrt. _energy is O(N^2) without sqrt.
        # _energy is faster.
        # But we need to return the best *score*.
        # Let's evaluate score every 50 iterations.
        
        # Simple counter
        if int((time.time() - start_time) * 100) % 50 == 0:
             # This is a hacky way to throttle. Better to use a counter.
             pass
             
        # Let's use a proper counter
        # Actually, let's just check score if energy improves?
        # No, we might find a better score with slightly higher energy (unlikely but possible).
        # Let's check score every 20 iterations.
        
        # To keep code simple and fast, let's just check score every 100 iterations.
        # N=50, 100 iterations is fast.
        
        # We need a counter variable outside the loop.
        # Let's restructure slightly.
        
        # Cooling
        T *= cooling_rate
        if T < T_min:
            # Reheat? Or stop?
            # If we stop, we are done.
            # But we have time budget.
            # Let's just stop if T is too low, as steps become tiny.
            # Or we can reset T occasionally?
            # For now, stop.
            break
            
    # Final score check
    final_score = _score(current_pts)
    if final_score > best_score:
        best_pts = current_pts
        best_score = final_score
        
    return best_pts, best_score

def _generate_bcc_initial():
    """
    Generate an initial configuration based on BCC lattice.
    BCC has 2 atoms per unit cell.
    For N=50, we need 25 unit cells.
    25 = 5x5x1? Or 3x3x3=27 (close).
    Let's use a 3x3x3 grid of BCC cells? 27*2 = 54 points.
    We can take 50 of them.
    
    BCC unit cell: corners (0,0,0) to (1,1,1) and center (0.5,0.5,0.5).
    In a cube of side L, if we have n cells per side, cell size is L/n.
    Let n=3. Cell size = 1/3.
    Points:
    For ix in 0..2:
      For iy in 0..2:
        For iz in 0..2:
          Corner: (ix/3, iy/3, iz/3) -> shift to [-0.5, 0.5]
          Center: ((ix+0.5)/3, (iy+0.5)/3, (iz+0.5)/3)
          
    Total 27*2 = 54 points.
    We pick 50.
    """
    n = 3
    cell_size = 1.0 / n
    points = []
    for ix in range(n):
        for iy in range(n):
            for iz in range(n):
                # Corner
                x = (ix + 0.5) * cell_size - 0.5
                y = (iy + 0.5) * cell_size - 0.5
                z = (iz + 0.5) * cell_size - 0.5
                points.append([x, y, z])
                # Center
                xc = (ix + 0.5) * cell_size - 0.5 + 0.5 * cell_size
                yc = (iy + 0.5) * cell_size - 0.5 + 0.5 * cell_size
                zc = (iz + 0.5) * cell_size - 0.5 + 0.5 * cell_size
                # Wait, corner is at (ix*cs, iy*cs, iz*cs) in local cell coords?
                # If cell is [ix*cs, (ix+1)*cs], corner is ix*cs.
                # Center is (ix+0.5)*cs.
                # Let's re-calculate.
                # Global coord range [-0.5, 0.5].
                # Cell ix covers [ -0.5 + ix*cs, -0.5 + (ix+1)*cs ]
                # Corner at start of cell: -0.5 + ix*cs
                # Center: -0.5 + (ix+0.5)*cs
                
                x_corner = -0.5 + ix * cell_size
                y_corner = -0.5 + iy * cell_size
                z_corner = -0.5 + iz * cell_size
                points.append([x_corner, y_corner, z_corner])
                
                x_center = -0.5 + (ix + 0.5) * cell_size
                y_center = -0.5 + (iy + 0.5) * cell_size
                z_center = -0.5 + (iz + 0.5) * cell_size
                points.append([x_center, y_center, z_center])
                
    points = np.array(points)
    # We have 54 points. Pick 50.
    # To be fair, maybe shuffle and take first 50?
    # Or just take first 50.
    # Taking first 50 might bias towards one corner.
    # Let's shuffle.
    idx = np.random.choice(len(points), N, replace=False)
    return points[idx]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin
    
    # Strategy:
    # 1. Generate several diverse initial configurations (BCC, Grid, Random).
    # 2. Run SA on each for a portion of the time.
    # 3. Keep the best result.
    
    best_pts = None
    best_score = -1.0
    
    # Time allocation
    # Total 115s.
    # Let's try 3 different starts.
    # Start 1: BCC
    # Start 2: Grid
    # Start 3: Random
    
    configs = []
    
    # BCC
    configs.append(_generate_bcc_initial())
    
    # Grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    configs.append(grid[:N].copy())
    
    # Random
    configs.append(np.random.uniform(-0.4, 0.4, size=(N, 3)))
    
    # We can run SA on each.
    # To maximize chance, we should run the most promising one longer?
    # Or just split time evenly?
    # BCC is likely the best start. Let's give it more time.
    
    time_per_sa = 20.0
    # 3 * 20 = 60s. Leaves 55s for... what?
    # We can run more restarts or longer SA.
    # Let's do:
    # 1. BCC SA for 30s
    # 2. Grid SA for 20s
    # 3. Random SA for 20s
    # 4. If time left, run SA on the best result found so far for 20s (polish)
    
    # 1. BCC
    if time.time() < deadline - 35:
        pts, score = _sa_optimize(configs[0], time_budget=30.0)
        if score > best_score:
            best_score = score
            best_pts = pts
    
    # 2. Grid
    if time.time() < deadline - 15:
        pts, score = _sa_optimize(configs[1], time_budget=20.0)
        if score > best_score:
            best_score = score
            best_pts = pts
    
    # 3. Random
    if time.time() < deadline - 5:
        pts, score = _sa_optimize(configs[2], time_budget=20.0)
        if score > best_score:
            best_score = score
            best_pts = pts
            
    # 4. Polish best
    if best_pts is not None and time.time() < deadline - 5:
        # Run SA on the best found so far to see if we can improve further
        # Use a lower initial T? Or just run again.
        # Since SA is stochastic, running again might find a better local min.
        pts, score = _sa_optimize(best_pts, time_budget=15.0)
        if score > best_score:
            best_score = score
            best_pts = pts
    
    if best_pts is None:
        # Fallback
        best_pts = configs[1]
        
    # Ensure validity
    best_pts = np.clip(best_pts, -0.5, 0.5)
    
    return best_pts
# EVOLVE-BLOCK-END
