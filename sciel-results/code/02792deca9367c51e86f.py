# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.spatial.distance as spd

def calculate_score(points):
    """Calculate the metric R = (d_min / d_max)^2"""
    if len(points) < 2:
        return 0.0
    dists = spd.pdist(points)
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0:
        return 0.0
    ratio = d_min / d_max
    return ratio * ratio

def optimize_points(points, time_limit, seed=0):
    """
    Optimize points using a combination of simulated annealing and gradient-free methods.
    """
    n = points.shape[0]
    current = points.copy()
    current_score = calculate_score(current)
    best = current.copy()
    best_score = current_score
    
    rng = np.random.RandomState(seed)
    start_time = time.time()
    
    # Simulated Annealing parameters
    temp = 1.0
    cooling = 0.9999
    min_temp = 1e-8
    
    while time.time() - start_time < time_limit:
        # Pick a random point to move
        idx = rng.randint(n)
        
        # Generate perturbation
        step_size = 0.1 * temp + 0.01
        perturbation = rng.randn(2) * step_size
        
        new_point = current[idx].copy() + perturbation
        trial = current.copy()
        trial[idx] = new_point
        
        trial_score = calculate_score(trial)
        
        # Acceptance criterion
        if trial_score > current_score or rng.rand() < np.exp((trial_score - current_score) / max(temp, 1e-10)):
            current = trial
            current_score = trial_score
            
            if current_score > best_score:
                best_score = current_score
                best = current.copy()
        
        temp *= cooling
        if temp < min_temp:
            temp = min_temp
            
    return best, best_score

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    """
    n = 16
    d = 2
    
    global_deadline = time.time() + 620  # Safety margin
    
    # Initial candidates
    candidates = []
    
    # 1. Hexagonal lattice (best known packing structure)
    # Generate hex grid points
    hex_points = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            x = q + 0.5 * r
            y = np.sqrt(3) / 2 * r
            hex_points.append([x, y])
    hex_points = np.array(hex_points)
    # Select 16 points closest to center
    dists_to_origin = np.linalg.norm(hex_points, axis=1)
    idx = np.argsort(dists_to_origin)[:16]
    candidates.append(hex_points[idx])
    
    # 2. 4x4 grid
    x = np.linspace(-1.5, 1.5, 4)
    y = np.linspace(-1.5, 1.5, 4)
    xx, yy = np.meshgrid(x, y)
    candidates.append(np.column_stack((xx.ravel(), yy.ravel())))
    
    # 3. Circle with two rings
    angles1 = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    angles2 = np.linspace(np.pi / 8, 2 * np.pi, 8, endpoint=False)
    r1, r2 = 1.0, 0.5
    x1 = r1 * np.cos(angles1)
    y1 = r1 * np.sin(angles1)
    x2 = r2 * np.cos(angles2)
    y2 = r2 * np.sin(angles2)
    candidates.append(np.column_stack([np.concatenate([x1, x2]), np.concatenate([y1, y2])]))
    
    # 4. Random points in a disk
    rng_init = np.random.RandomState(42)
    angles = rng_init.rand(n) * 2 * np.pi
    radii = np.sqrt(rng_init.rand(n)) * 1.5
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    candidates.append(np.column_stack((x, y)))
    
    # 5. Random points in a square
    rng_init = np.random.RandomState(123)
    candidates.append(rng_init.rand(n, 2) * 3 - 1.5)
    
    # Optimize each candidate
    best_points = None
    best_score = -1
    
    time_per_candidate = 100
    
    for i, cand in enumerate(candidates):
        if time.time() > global_deadline - 10:
            break
        
        cand = cand.astype(float)
        res_points, res_score = optimize_points(cand, time_per_candidate, seed=1000 + i)
        
        if res_score > best_score:
            best_score = res_score
            best_points = res_points
    
    # If time remains, do a final refinement on the best candidate
    if best_points is not None and time.time() < global_deadline - 5:
        remaining = global_deadline - 5 - time.time()
        if remaining > 10:
            res_points, res_score = optimize_points(best_points, remaining, seed=9999)
            if res_score > best_score:
                best_points = res_points
                best_score = res_score
    
    if best_points is None:
        np.random.seed(42)
        best_points = np.random.randn(n, d)
    
    if not np.all(np.isfinite(best_points)):
        best_points = np.zeros((n, d))
    
    return best_points

# EVOLVE-BLOCK-END
