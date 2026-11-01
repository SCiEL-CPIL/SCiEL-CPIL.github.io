"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def _score(positions, override_idx=None, override_pos=None):
    """
    Compute the radius for a given set of positions.
    r = min(min_{i<j} |c_i - c_j| / 2, min_i min_k (1/2 - |c_ik|))
    If override_idx and override_pos are provided, they are used in place of positions[override_idx].
    """
    n = len(positions)
    if n == 0:
        return 0.0
    
    # Prepare positions for calculation
    if override_idx is not None:
        p = positions.copy()
        p[override_idx] = override_pos
    else:
        p = positions

    # Wall distances
    wall_dists = 0.5 - np.abs(p)
    min_wall = np.min(wall_dists)
    
    if min_wall <= 0:
        return 0.0
        
    # Pairwise distances
    # Vectorized calculation
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    
    # Ignore diagonal
    np.fill_diagonal(dists_sq, np.inf)
    
    min_dist_sq = np.min(dists_sq)
    min_pair_dist = np.sqrt(min_dist_sq) / 2.0
    
    return min(min_wall, min_pair_dist)

def _init_grid(n):
    """
    Initialize with a 5x5x4 grid.
    """
    nx, ny, nz = 5, 5, 4
    xs = np.linspace(-0.5, 0.5, nx)
    ys = np.linspace(-0.5, 0.5, ny)
    zs = np.linspace(-0.5, 0.5, nz)
    
    points = []
    for z in zs:
        for y in ys:
            for x in xs:
                points.append([x, y, z])
                
    points = np.array(points[:n])
    # Add small random jitter
    points += np.random.normal(0, 0.01, size=points.shape)
    return points

def _anneal(positions, time_budget=100.0):
    """
    Metropolis-Hastings Annealing.
    """
    n = len(positions)
    current = positions.copy()
    current_r = _score(current)
    best = current.copy()
    best_r = current_r
    
    # Cooling schedule
    T = 0.05
    T_min = 1e-5
    alpha = 0.99995
    
    # Step size
    step = 0.05
    
    rng = np.random.default_rng(42)
    start_time = time.time()
    
    while time.time() - start_time < time_budget:
        # Generate move: pick a random sphere
        i = rng.integers(0, n)
        
        # Perturb
        delta = rng.normal(0, step, size=3)
        new_pos = current[i] + delta
        new_pos = np.clip(new_pos, -0.5, 0.5)
        
        # Calculate score with override
        proposed_r = _score(current, i, new_pos)
        
        # Acceptance
        if proposed_r >= current_r:
            current[i] = new_pos
            current_r = proposed_r
            if current_r > best_r:
                best = current.copy()
                best_r = current_r
        else:
            delta_r = proposed_r - current_r
            if delta_r < 0:
                prob = np.exp(delta_r / T)
                if rng.random() < prob:
                    current[i] = new_pos
                    current_r = proposed_r
        
        # Update temperature and step
        T *= alpha
        if T < T_min:
            T = T_min
            
        step = 0.05 * (T / 0.05) ** 0.5
        if step < 1e-4:
            step = 1e-4
            
    return best, best_r

def construct():
    """Construct positions for 100 equal spheres in a unit cube."""
    n = N
    
    # Try multiple initializations
    best_positions = None
    best_score = 0.0
    
    # Init 1: Grid
    pos1 = _init_grid(n)
    p1, s1 = _anneal(pos1, time_budget=30.0)
    if s1 > best_score:
        best_positions = p1
        best_score = s1
        
    # Init 2: Random
    pos2 = np.random.uniform(-0.45, 0.45, size=(n, 3))
    p2, s2 = _anneal(pos2, time_budget=30.0)
    if s2 > best_score:
        best_positions = p2
        best_score = s2
        
    # Init 3: Grid with different aspect ratio (4x5x5)
    nx, ny, nz = 4, 5, 5
    xs = np.linspace(-0.5, 0.5, nx)
    ys = np.linspace(-0.5, 0.5, ny)
    zs = np.linspace(-0.5, 0.5, nz)
    points = []
    for z in zs:
        for y in ys:
            for x in xs:
                points.append([x, y, z])
    pos3 = np.array(points[:n])
    pos3 += np.random.normal(0, 0.01, size=pos3.shape)
    p3, s3 = _anneal(pos3, time_budget=30.0)
    if s3 > best_score:
        best_positions = p3
        best_score = s3

    return best_positions
# EVOLVE-BLOCK-END
