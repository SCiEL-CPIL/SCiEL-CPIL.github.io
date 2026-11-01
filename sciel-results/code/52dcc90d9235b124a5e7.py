import numpy as np
import time
from scipy.spatial.distance import cdist

N = 100
_DEADLINE = None

def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 145.0
    return _DEADLINE

def _time_left():
    return _get_deadline() - time.time()

def _compute_radius(centers):
    """
    Compute the largest common radius for the given centers.
    Centers are in [-0.5, 0.5]^3.
    """
    # Distance to walls
    dists_to_walls = 0.5 - np.abs(centers)
    min_wall_dist = np.min(dists_to_walls)
    
    # Pairwise distances
    dists = cdist(centers, centers, metric='euclidean')
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    return min(min_wall_dist, min_pair_dist / 2.0)

def _build_grid_init(nx, ny, nz, noise_scale=0.0, seed=42):
    """Build a grid initialization."""
    if nx == 1: xs = np.array([0.0])
    else: xs = np.linspace(-0.5, 0.5, nx)
    
    if ny == 1: ys = np.array([0.0])
    else: ys = np.linspace(-0.5, 0.5, ny)
    
    if nz == 1: zs = np.array([0.0])
    else: zs = np.linspace(-0.5, 0.5, nz)
    
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    centers = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    if len(centers) != N:
        if len(centers) < N:
            rng = np.random.default_rng(seed)
            extra = rng.uniform(-0.45, 0.45, (N - len(centers), 3))
            centers = np.vstack([centers, extra])
        else:
            centers = centers[:N]

    if noise_scale > 0:
        rng = np.random.default_rng(seed)
        centers += rng.normal(0, noise_scale, centers.shape)
    
    centers = np.clip(centers, -0.499, 0.499)
    return centers

def _build_halton_init(seed=42):
    """
    Generate N points using a Halton sequence (low discrepancy).
    """
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r

    bases = [2, 3, 5]
    points = []
    for i in range(1, N + 1):
        pt = [halton(i, b) for b in bases]
        pt_mapped = [p - 0.5 for p in pt]
        points.append(pt_mapped)
        
    centers = np.array(points)
    
    rng = np.random.default_rng(seed)
    centers += rng.normal(0, 0.01, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers

def _anneal(centers, deadline):
    """
    Simulated annealing to maximize the minimum clearance.
    Objective: maximize r = min(min_wall_dist, min_pair_dist/2)
    We minimize energy = -r + penalty for out-of-bounds.
    """
    centers = centers.copy()
    best_centers = centers.copy()
    best_r = _compute_radius(centers)
    
    # Initial temperature
    # Estimate typical distance
    d = cdist(centers, centers)
    np.fill_diagonal(d, np.inf)
    avg_dist = np.mean(d)
    T = 0.1 * avg_dist
    
    T_min = 1e-6
    # Cooling schedule: exponential
    # We want to run until deadline
    start_time = time.time()
    
    # Estimate number of iterations we can do
    # Each iteration involves cdist (100x100) which is fast, but let's be conservative.
    # Let's try to do as many as possible.
    
    while time.time() < deadline - 1.0:
        if T < T_min:
            # Reheat slightly or stop
            # If we haven't improved in a long time, maybe reheat?
            # For now, just stop if T is too low
            # Or continue with small moves
            T = T_min
            
        # Pick a random sphere
        idx = np.random.randint(N)
        
        # Random move
        # Step size proportional to T
        step_size = T * 0.5
        delta = np.random.normal(0, step_size, 3)
        
        new_center = centers[idx] + delta
        # Clip to box
        new_center = np.clip(new_center, -0.5, 0.5)
        
        # Calculate energy change
        # Old r
        old_r = best_r # Approximation: use current best? No, we need current state r.
        # Actually, we should track current r, but recomputing is expensive.
        # Let's compute r for the modified state.
        # We only changed one point.
        
        # Compute new wall distance for this point
        new_wall_dist = 0.5 - np.abs(new_center)
        min_new_wall = np.min(new_wall_dist)
        
        # Compute new pairwise distances involving this point
        # Distances to all other points
        diffs = centers - new_center
        diffs[idx] = np.inf # Exclude self
        dists_to_others = np.linalg.norm(diffs, axis=1)
        min_new_pair = np.min(dists_to_others) / 2.0
        
        # The new global minimum radius depends on:
        # 1. The minimum wall distance of ALL points (min of old others, new point)
        # 2. The minimum pairwise distance of ALL pairs (min of old others, new pairs)
        
        # This is tricky without storing the full distance matrix or sorted mins.
        # For simplicity and correctness, let's just recompute _compute_radius for the new state.
        # It's O(N^2). With N=100, it's 10k ops.
        # In Python, this might be slow if we do it every step.
        # Let's try to optimize by only checking if the new point is the limiting factor.
        
        # Alternative: Store the current limiting pair/wall.
        # This is complex. Let's just recompute.
        # If it's too slow, we reduce iterations.
        
        centers[idx] = new_center
        new_r = _compute_radius(centers)
        
        # Energy change: we want to maximize r.
        # So delta_E = old_r - new_r. If new_r > old_r, delta_E < 0 (good).
        # Let's use old_r as the r of the state BEFORE this move.
        # We need to keep track of current_r.
        
        # Wait, I didn't track current_r. Let's fix that.
        # I'll use best_r as a proxy for "current good state" but that's wrong for annealing.
        # I need current_r.
        
        # Let's restart the loop logic properly.
        
        # This implementation is getting messy. Let's switch to a simpler "hill climbing with random restarts" or "shaking" approach which is easier to implement correctly.
        
        # Actually, let's just do a simple random perturbation and accept if better (greedy hill climbing with noise).
        # Or better: Accept with probability exp((new_r - old_r)/T).
        
        # I need old_r.
        # Let's restructure.
        
        # Break out and write a cleaner version.
        break
        
    return best_centers, best_r

def _shaking_optimize(centers, deadline):
    """
    Greedy hill climbing with random shaking.
    1. Try to move a random point slightly to increase r.
    2. If stuck, shake all points.
    """
    centers = centers.copy()
    current_r = _compute_radius(centers)
    best_centers = centers.copy()
    best_r = current_r
    
    stuck_count = 0
    max_stuck = 1000
    
    while time.time() < deadline - 1.0:
        if stuck_count > max_stuck:
            # Shake
            centers += np.random.normal(0, 0.01, centers.shape)
            centers = np.clip(centers, -0.5, 0.5)
            current_r = _compute_radius(centers)
            stuck_count = 0
            continue
            
        # Pick random point
        idx = np.random.randint(N)
        
        # Try small random moves
        improved = False
        for _ in range(10): # Try 10 random directions
            delta = np.random.normal(0, 0.005, 3)
            new_center = centers[idx] + delta
            new_center = np.clip(new_center, -0.5, 0.5)
            
            old_center = centers[idx]
            centers[idx] = new_center
            new_r = _compute_radius(centers)
            
            if new_r > current_r:
                current_r = new_r
                improved = True
                if new_r > best_r:
                    best_r = new_r
                    best_centers = centers.copy()
                break
            else:
                centers[idx] = old_center
        
        if not improved:
            stuck_count += 1
        else:
            stuck_count = 0
            
    return best_centers, best_r

def _local_optimize(centers, deadline):
    """
    More structured local optimization.
    1. Identify the limiting constraint (pair or wall).
    2. Move the involved points away from each other or wall.
    """
    centers = centers.copy()
    best_centers = centers.copy()
    best_r = _compute_radius(centers)
    
    while time.time() < deadline - 1.0:
        r = _compute_radius(centers)
        if r > best_r:
            best_r = r
            best_centers = centers.copy()
            
        # Find limiting pair
        dists = cdist(centers, centers)
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        i, j = np.unravel_index(np.argmin(dists), dists.shape)
        
        # Check wall constraint
        wall_dists = 0.5 - np.abs(centers)
        min_wall = np.min(wall_dists)
        i_wall, j_wall = np.unravel_index(np.argmin(wall_dists), wall_dists.shape)
        
        if min_dist / 2.0 < min_wall:
            # Pair is limiting
            # Move i and j apart
            diff = centers[i] - centers[j]
            norm = np.linalg.norm(diff)
            if norm > 1e-9:
                dir_i = diff / norm
                dir_j = -diff / norm
            else:
                dir_i = np.random.normal(0, 1, 3)
                dir_i /= np.linalg.norm(dir_i)
                dir_j = -dir_i
                
            step = 0.01
            centers[i] += step * dir_i
            centers[j] += step * dir_j
        else:
            # Wall is limiting
            # Move i_wall away from wall
            k = j_wall
            # Direction is towards center (0,0,0)
            dir = -np.sign(centers[i_wall, k])
            if dir == 0:
                dir = 1.0
            step = 0.01
            centers[i_wall, k] += step * dir
            
        centers = np.clip(centers, -0.5, 0.5)
        
    return best_centers, best_r

def construct():
    start_time = time.time()
    deadline = _get_deadline()
    
    best_centers = None
    best_radius = 0.0
    
    # Initializations
    initializations = []
    
    # Grids
    grid_configs = [
        (4, 5, 5),
        (5, 4, 5),
        (5, 5, 4),
        (10, 5, 2),
        (5, 10, 2),
        (5, 2, 10),
    ]
    
    for nx, ny, nz in grid_configs:
        for noise in [0.0, 0.01, 0.02]:
            centers = _build_grid_init(nx, ny, nz, noise, seed=nx*100+ny*10+nz)
            initializations.append(centers)
            
    # Halton
    initializations.append(_build_halton_init())
    
    # Random
    rng = np.random.default_rng(123)
    for _ in range(5):
        centers = rng.uniform(-0.45, 0.45, (N, 3))
        initializations.append(centers)
        
    # Optimize each initialization
    # We have limited time. Let's allocate time to each.
    n_inits = len(initializations)
    time_per_init = (_get_deadline() - time.time() - 5.0) / n_inits
    
    for centers in initializations:
        if time.time() > deadline - 5.0:
            break
            
        # Quick local optimization
        opt_centers, opt_radius = _local_optimize(centers, time.time() + time_per_init * 0.5)
        
        if opt_radius > best_radius:
            best_radius = opt_radius
            best_centers = opt_centers
            
        # Shaking optimization with remaining time for this init
        if time.time() < deadline - 5.0:
            shake_deadline = min(time.time() + time_per_init * 0.5, deadline - 5.0)
            shake_centers, shake_radius = _shaking_optimize(opt_centers, shake_deadline)
            
            if shake_radius > best_radius:
                best_radius = shake_radius
                best_centers = shake_centers

    if best_centers is None:
        best_centers = _build_grid_init(5, 5, 4, 0.0)
        
    return best_centers
