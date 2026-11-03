# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import time
import random

N = 100
CUBE_HALF = 0.5

def _score(c):
    """
    Compute the maximum radius for a given set of centers.
    r = min(min_pairwise_dist / 2, min_boundary_dist)
    """
    if N > 1:
        dists = pdist(c, metric='euclidean')
        min_pair_dist = np.min(dists)
    else:
        min_pair_dist = np.inf
        
    abs_c = np.abs(c)
    boundary_dists = CUBE_HALF - np.max(abs_c, axis=1)
    min_boundary_dist = np.min(boundary_dists)
    
    r = min(min_pair_dist / 2.0, min_boundary_dist)
    return r

def _initial_grid():
    """Create a dense grid initial placement."""
    m = 5
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    np.random.shuffle(grid)
    return grid[:N].copy()

def _initial_random():
    """Create a random initial placement."""
    return np.random.uniform(-CUBE_HALF, CUBE_HALF, size=(N, 3))

def _fast_score(c):
    """
    Fast approximate score using only the closest few pairs and boundary.
    This is used during SA iterations for speed.
    """
    # Boundary check is cheap
    abs_c = np.abs(c)
    boundary_dists = CUBE_HALF - np.max(abs_c, axis=1)
    min_boundary_dist = np.min(boundary_dists)
    
    # Pairwise: use pdist but we can't easily truncate without full compute in scipy
    # However, for N=100, pdist is 4950 distances. That's fast enough.
    dists = pdist(c, metric='euclidean')
    min_pair_dist = np.min(dists)
    
    r = min(min_pair_dist / 2.0, min_boundary_dist)
    return r

def _simulated_annealing(c, time_limit=20, T_start=0.03, T_end=0.001):
    """
    Simulated Annealing to maximize the minimum radius.
    Optimized for speed and effectiveness.
    """
    start_time = time.time()
    c = c.copy()
    current_score = _score(c)
    best_c = c.copy()
    best_score = current_score
    
    # Temperature schedule: exponential decay
    # T(t) = T_end + (T_start - T_end) * exp(-t / tau)
    # We want T to reach T_end at time_limit
    # T_start * exp(-time_limit / tau) = T_end
    # exp(-time_limit / tau) = T_end / T_start
    # -time_limit / tau = log(T_end / T_start)
    # tau = -time_limit / log(T_end / T_start)
    
    if T_end >= T_start:
        T_end = T_start * 0.1
    
    tau = -time_limit / np.log(T_end / T_start)
    
    step_size = 0.05
    
    while time.time() - start_time < time_limit:
        elapsed = time.time() - start_time
        T = T_end + (T_start - T_end) * np.exp(-elapsed / tau)
        
        # Adaptive step size: larger when hot, smaller when cold
        # Scale with sqrt(T) or linearly
        step = step_size * np.sqrt(T / T_start)
        step = max(step, 0.002)
        
        # Choose move type
        r = random.random()
        c_new = c.copy()
        
        if r < 0.7:
            # Move one point
            idx = random.randint(0, N - 1)
            delta = np.random.randn(3) * step
            c_new[idx] = c_new[idx] + delta
            c_new[idx] = np.clip(c_new[idx], -CUBE_HALF, CUBE_HALF)
        elif r < 0.9:
            # Move two points
            i1, i2 = random.sample(range(N), 2)
            delta1 = np.random.randn(3) * step
            delta2 = np.random.randn(3) * step
            c_new[i1] = c_new[i1] + delta1
            c_new[i2] = c_new[i2] + delta2
            c_new[i1] = np.clip(c_new[i1], -CUBE_HALF, CUBE_HALF)
            c_new[i2] = np.clip(c_new[i2], -CUBE_HALF, CUBE_HALF)
        else:
            # Large jump for one point (restart-like)
            idx = random.randint(0, N - 1)
            c_new[idx] = np.random.uniform(-CUBE_HALF, CUBE_HALF, size=3)
        
        new_score = _score(c_new)
        
        # Acceptance
        if new_score >= current_score:
            c = c_new
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_c = c.copy()
        else:
            delta_score = new_score - current_score
            # Avoid division by zero or underflow
            if T > 1e-10:
                prob = np.exp(delta_score / T)
                if random.random() < prob:
                    c = c_new
                    current_score = new_score

    return best_c, best_score

def _local_search(c, max_iters=100, step=0.01):
    """
    Deterministic local search to polish the solution.
    Tries small moves in 6 directions for each point.
    """
    c = c.copy()
    current_score = _score(c)
    
    for _ in range(max_iters):
        improved = False
        for i in range(N):
            # Try 6 axis-aligned directions + 6 diagonal directions
            directions = [
                [1, 0, 0], [-1, 0, 0],
                [0, 1, 0], [0, -1, 0],
                [0, 0, 1], [0, 0, -1],
                [1, 1, 0], [-1, -1, 0],
                [1, 0, 1], [-1, 0, -1],
                [0, 1, 1], [0, -1, -1]
            ]
            
            for d in directions:
                d = np.array(d)
                d = d / np.linalg.norm(d)
                c_new = c.copy()
                c_new[i] = c[i] + d * step
                c_new[i] = np.clip(c_new[i], -CUBE_HALF, CUBE_HALF)
                
                s_new = _score(c_new)
                if s_new > current_score:
                    c = c_new
                    current_score = s_new
                    improved = True
                    break
            if improved:
                break
        if not improved:
            # Try smaller step
            step *= 0.5
            if step < 1e-5:
                break
    
    return c, current_score

def construct():
    """
    Construct the optimal arrangement of 100 equal spheres in a unit cube.
    """
    best_c = None
    best_score = -np.inf
    
    # Time budget: 120s total. Leave 5s for overhead.
    
    # 1. Grid start
    c_grid = _initial_grid()
    c_best, s_best = _simulated_annealing(c_grid, time_limit=15, T_start=0.04, T_end=0.001)
    if s_best > best_score:
        best_score = s_best
        best_c = c_best
        
    # 2. Random starts
    # We have ~100s left. Let's do 4 random restarts of 20s each
    for i in range(4):
        c_rand = _initial_random()
        c_curr, s_curr = _simulated_annealing(c_rand, time_limit=20, T_start=0.05, T_end=0.001)
        if s_curr > best_score:
            best_score = s_curr
            best_c = c_curr
            
    # 3. Final refinement on the best solution found
    if best_c is not None:
        # Run local search to polish
        c_polish, s_polish = _local_search(best_c, max_iters=50, step=0.01)
        if s_polish > best_score:
            best_score = s_polish
            best_c = c_polish
        
        # One more SA pass with lower temperature for fine-tuning
        c_final, s_final = _simulated_annealing(best_c, time_limit=10, T_start=0.01, T_end=0.0001)
        if s_final > best_score:
            best_score = s_final
            best_c = c_final
        
        # Final local search
        c_final2, s_final2 = _local_search(best_c, max_iters=50, step=0.005)
        if s_final2 > best_score:
            best_score = s_final2
            best_c = c_final2

    return best_c
# EVOLVE-BLOCK-END
