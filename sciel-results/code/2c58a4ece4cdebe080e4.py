# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, cdist
import time
import random

N = 100
CUBE_HALF = 0.5

def _score(c):
    """
    Compute the maximum radius for a given set of centers.
    r = min(min_pairwise_dist / 2, min_boundary_dist)
    """
    # Pairwise distances
    if N > 1:
        dists = pdist(c, metric='euclidean')
        min_pair_dist = np.min(dists)
    else:
        min_pair_dist = np.inf
        
    # Boundary distances
    # Distance from each point to the nearest face
    # For a point (x, y, z) in [-0.5, 0.5]^3
    # dist to x-face = 0.5 - |x|
    abs_c = np.abs(c)
    boundary_dists = CUBE_HALF - np.max(abs_c, axis=1)
    min_boundary_dist = np.min(boundary_dists)
    
    # The radius is limited by half the minimum distance between centers
    # and the minimum distance to the boundary.
    r = min(min_pair_dist / 2.0, min_boundary_dist)
    return r

def _score_delta(c, c_new):
    """
    Fast delta scoring: only compute what changed.
    Returns (new_score, delta).
    """
    # Compute new pairwise distances
    new_dists = pdist(c_new, metric='euclidean')
    new_min_pair = np.min(new_dists)
    
    # Boundary
    abs_new = np.abs(c_new)
    new_boundary = CUBE_HALF - np.max(abs_new, axis=1)
    new_min_boundary = np.min(new_boundary)
    
    new_score = min(new_min_pair / 2.0, new_min_boundary)
    return new_score

def _initial_grid():
    """Create a dense grid initial placement."""
    m = 5
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    np.random.shuffle(grid)
    return grid[:N].copy()

def _initial_fcc():
    """Create FCC-like lattice initial placement."""
    # FCC lattice has higher density than simple cubic
    # Generate FCC points in a box and select N
    a = 0.3  # lattice constant
    points = []
    # FCC basis: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)
    basis = np.array([[0,0,0], [0,0.5,0.5], [0.5,0,0.5], [0.5,0.5,0]])
    
    # Generate grid of FCC unit cells
    for ix in range(-1, 3):
        for iy in range(-1, 3):
            for iz in range(-1, 3):
                origin = np.array([ix*a, iy*a, iz*a])
                for b in basis:
                    p = origin + b * a
                    # Check if in cube
                    if np.all(np.abs(p) <= CUBE_HALF):
                        points.append(p)
    
    points = np.array(points)
    # Shuffle and take N
    np.random.shuffle(points)
    if len(points) >= N:
        return points[:N].copy()
    else:
        # Fall back to random
        return np.random.uniform(-CUBE_HALF, CUBE_HALF, size=(N, 3))

def _initial_random():
    """Create a random initial placement."""
    return np.random.uniform(-CUBE_HALF, CUBE_HALF, size=(N, 3))

def _stochastic_local_search(c, time_limit=50, T_start=0.01, T_end=1e-7, alpha=0.998):
    """
    Stochastic Local Search with simulated annealing.
    Uses vectorized moves and fast scoring.
    """
    start_time = time.time()
    c = c.copy()
    current_score = _score(c)
    best_c = c.copy()
    best_score = current_score
    
    T = T_start
    step_size = 0.03
    
    # Move type probabilities
    p_single = 0.7
    p_double = 0.2
    p_jump = 0.1
    
    while time.time() - start_time < time_limit:
        r = random.random()
        c_new = c.copy()
        
        if r < p_single:
            # Move one point
            idx = random.randint(0, N - 1)
            delta = np.random.randn(3) * step_size
            c_new[idx] = c_new[idx] + delta
            c_new[idx] = np.clip(c_new[idx], -CUBE_HALF, CUBE_HALF)
        elif r < p_single + p_double:
            # Move two points
            i1, i2 = random.sample(range(N), 2)
            c_new[i1] = c_new[i1] + np.random.randn(3) * step_size
            c_new[i2] = c_new[i2] + np.random.randn(3) * step_size
            c_new[i1] = np.clip(c_new[i1], -CUBE_HALF, CUBE_HALF)
            c_new[i2] = np.clip(c_new[i2], -CUBE_HALF, CUBE_HALF)
        else:
            # Large jump for one point
            idx = random.randint(0, N - 1)
            c_new[idx] = np.random.uniform(-CUBE_HALF, CUBE_HALF, size=3)
        
        new_score = _score_delta(c, c_new)
        
        # Acceptance criterion
        if new_score >= current_score:
            c = c_new
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_c = c.copy()
        else:
            delta = new_score - current_score
            if delta > -100 * T:  # Avoid underflow
                prob = np.exp(delta / T)
                if random.random() < prob:
                    c = c_new
                    current_score = new_score
        
        # Cool down
        T *= alpha
        if T < T_end:
            T = T_end
        
        # Adaptive step size
        step_size = max(0.005, 0.03 * (T / T_start) ** 0.3)
    
    return best_c, best_score

def _polish(c, time_limit=10):
    """
    Final polishing: hill climbing with small steps.
    """
    start_time = time.time()
    c = c.copy()
    current_score = _score(c)
    best_c = c.copy()
    best_score = current_score
    
    step_sizes = [0.01, 0.005, 0.002, 0.001]
    
    for step_size in step_sizes:
        while time.time() - start_time < time_limit:
            improved = False
            for i in range(N):
                # Try 6 directions + 3 axis-aligned
                for dx, dy, dz in [
                    (step_size, 0, 0), (-step_size, 0, 0),
                    (0, step_size, 0), (0, -step_size, 0),
                    (0, 0, step_size), (0, 0, -step_size),
                    (step_size/np.sqrt(3), step_size/np.sqrt(3), step_size/np.sqrt(3)),
                    (-step_size/np.sqrt(3), -step_size/np.sqrt(3), -step_size/np.sqrt(3)),
                ]:
                    c_test = c.copy()
                    c_test[i] = c[i] + np.array([dx, dy, dz])
                    c_test[i] = np.clip(c_test[i], -CUBE_HALF, CUBE_HALF)
                    s_test = _score(c_test)
                    if s_test > current_score + 1e-12:
                        c = c_test
                        current_score = s_test
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_c = c.copy()
                        break
                if improved:
                    break
            if not improved:
                break
    
    return best_c, best_score

def construct():
    """
    Construct the optimal arrangement of 100 equal spheres in a unit cube.
    """
    best_c = None
    best_score = -np.inf
    
    # Time budget: 120s total. Leave 5s for overhead.
    total_budget = 115
    
    # 1. Grid start
    c_grid = _initial_grid()
    c_best, s_best = _stochastic_local_search(c_grid, time_limit=25)
    if s_best > best_score:
        best_score = s_best
        best_c = c_best
    
    # 2. FCC start
    c_fcc = _initial_fcc()
    c_best, s_best = _stochastic_local_search(c_fcc, time_limit=25)
    if s_best > best_score:
        best_score = s_best
        best_c = c_best
    
    # 3. Random starts
    for i in range(3):
        c_rand = _initial_random()
        c_curr, s_curr = _stochastic_local_search(c_rand, time_limit=15)
        if s_curr > best_score:
            best_score = s_curr
            best_c = c_curr
    
    # 4. Final refinement on the best solution
    if best_c is not None:
        c_polish, s_polish = _polish(best_c, time_limit=15)
        if s_polish > best_score:
            best_score = s_polish
            best_c = c_polish
    
    return best_c
# EVOLVE-BLOCK-END
