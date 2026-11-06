"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time


def _radius(centers):
    """Compute the maximum common radius for given centers (vectorized)."""
    n = len(centers)
    # Pairwise distances (upper triangle)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, np.inf)
    min_pair_dist = np.sqrt(np.min(d2))
    # Distance to walls
    wall = 0.5 - np.max(np.abs(centers), axis=1)
    min_wall = np.min(wall)
    return min(min_pair_dist / 2.0, min_wall)


def _sa_optimize(centers, time_budget):
    """Simulated Annealing optimization."""
    start = time.time()
    centers = centers.copy()
    n = len(centers)
    current_r = _radius(centers)
    best_r = current_r
    best_centers = centers.copy()
    
    T = 0.005
    T_min = 1e-5
    step_size = 0.01
    decay = 0.9995
    
    while time.time() - start < time_budget:
        # Pick a random point
        i = np.random.randint(n)
        
        # Perturb
        delta = np.random.normal(0, step_size, 3)
        new_center = centers[i].copy()
        new_center += delta
        new_center = np.clip(new_center, -0.5, 0.5)
        
        # Calculate new radius
        # To speed up, we can approximate, but for 50 points, full calc is fine
        # We only need to check distances involving point i and walls
        # However, moving point i might not change the global min if the min is elsewhere.
        # But _radius is vectorized and fast enough for 50 points.
        
        new_centers = centers.copy()
        new_centers[i] = new_center
        new_r = _radius(new_centers)
        
        # Acceptance criterion
        if new_r >= current_r:
            centers = new_centers
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_centers = centers.copy()
        else:
            # Accept with probability exp((new_r - current_r) / T)
            # Since r is small, (new_r - current_r) is negative.
            # If T is very small, prob is ~0.
            if np.random.rand() < np.exp((new_r - current_r) / T):
                centers = new_centers
                current_r = new_r
                if current_r > best_r:
                    best_r = current_r
                    best_centers = centers.copy()
        
        T *= decay
        step_size *= decay
        if T < T_min:
            T = T_min
            step_size = max(step_size, 0.001)
            # Restart with slightly higher T if stuck? 
            # For simplicity, just keep going with min T.
            
    return best_centers, best_r


def _coordinate_descent(centers, time_budget):
    """Coordinate Descent polish."""
    start = time.time()
    centers = centers.copy()
    n = len(centers)
    current_r = _radius(centers)
    best_r = current_r
    best_centers = centers.copy()
    
    while time.time() - start < time_budget:
        improved = False
        for i in range(n):
            for d in range(3):
                # Optimize coordinate d of point i
                # Current value
                val = centers[i, d]
                
                # Define search range
                # We can search in a small window around val or the whole cube.
                # Whole cube is safer.
                # Use a small grid search
                candidates = np.linspace(-0.5, 0.5, 20)
                best_val = val
                best_r_local = current_r
                
                # To speed up, we can pre-calculate distances to other points
                # But for 50 points, just evaluating _radius is fast.
                
                for c_val in candidates:
                    new_centers = centers.copy()
                    new_centers[i, d] = c_val
                    r = _radius(new_centers)
                    if r > best_r_local:
                        best_r_local = r
                        best_val = c_val
                
                if best_val != val:
                    centers[i, d] = best_val
                    current_r = best_r_local
                    improved = True
                    if current_r > best_r:
                        best_r = current_r
                        best_centers = centers.copy()
        
        if not improved:
            break
            
    return best_centers, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # Initial configuration: 4x4x4 grid (64 points), take 50
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    best_centers = centers.copy()
    best_r = _radius(best_centers)

    # Phase 1: Simulated Annealing
    # Give it a significant chunk of time
    if time.time() - start_time < deadline - 20:
        budget = min(60.0, deadline - 20 - (time.time() - start_time))
        sa_centers, sa_r = _sa_optimize(centers, budget)
        if sa_r > best_r:
            best_r = sa_r
            best_centers = sa_centers.copy()

    # Phase 2: Coordinate Descent
    # Polish the result
    if time.time() - start_time < deadline - 5:
        budget = min(30.0, deadline - 5 - (time.time() - start_time))
        cd_centers, cd_r = _coordinate_descent(best_centers, budget)
        if cd_r > best_r:
            best_r = cd_r
            best_centers = cd_centers.copy()

    # Final validity check
    if not np.all(np.isfinite(best_centers)) or np.any(np.abs(best_centers) > 0.5):
        best_centers = grid[:N].copy()

    return best_centers
# EVOLVE-BLOCK-END
