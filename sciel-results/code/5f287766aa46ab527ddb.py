import numpy as np
import time
from scipy.spatial.distance import cdist

N = 100

# EVOLVE-BLOCK-START

def _compute_radius(centers):
    """Compute the largest common radius for the given centers in [-0.5, 0.5]^3."""
    # Distance to walls
    dists_to_walls = 0.5 - np.abs(centers)
    min_wall_dist = np.min(dists_to_walls)
    
    # Pairwise distances
    dists = cdist(centers, centers, metric='euclidean')
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    return min(min_wall_dist, min_pair_dist / 2.0)

def _local_search(centers, deadline, rng):
    """
    Greedy Local Search with Random Kicks.
    """
    c = centers.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    # Parameters
    step_size = 0.01
    kick_size = 0.05
    no_improve_count = 0
    max_no_improve = 2000
    
    while time.time() < deadline:
        # Pick a random particle
        idx = rng.integers(0, N)
        
        # Try to move it in a random direction
        c_new = c.copy()
        c_new[idx] += rng.normal(0, step_size, 3)
        c_new = np.clip(c_new, -0.5, 0.5)
        
        r_new = _compute_radius(c_new)
        
        if r_new > best_r + 1e-12:
            c = c_new
            best_c = c_new.copy()
            best_r = r_new
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # If stuck, do a random kick to escape local optimum
        if no_improve_count > max_no_improve:
            # Kick a random subset of particles
            kick_indices = rng.choice(N, size=10, replace=False)
            c[kick_indices] += rng.normal(0, kick_size, (10, 3))
            c = np.clip(c, -0.5, 0.5)
            r_curr = _compute_radius(c)
            if r_curr > best_r:
                best_c = c.copy()
                best_r = r_curr
            no_improve_count = 0
            # Reset step size to a moderate value after kick
            step_size = 0.01

    return best_c, best_r

def construct():
    start_time = time.time()
    # Safety margin: 110s is well within 120s.
    deadline = start_time + 110.0
    
    rng = np.random.default_rng(42)
    
    best_centers = None
    best_radius = -np.inf
    
    # Multiple restarts
    # We will run local search from different initializations.
    # Allocation: 50% to first, 30% to second, 20% to third.
    
    # Initialization 1: Grid
    nx, ny, nz = 5, 5, 4
    xs = np.linspace(-0.4, 0.4, nx)
    ys = np.linspace(-0.4, 0.4, ny)
    zs = np.linspace(-0.4, 0.4, nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    init1 = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))[:N]
    
    # Initialization 2: Random
    init2 = rng.uniform(-0.45, 0.45, size=(N, 3))
    
    # Initialization 3: Perturbed Grid
    init3 = init1 + rng.normal(0, 0.02, init1.shape)
    init3 = np.clip(init3, -0.49, 0.49)
    
    inits = [init1, init2, init3]
    
    # Sort by initial radius to prioritize good starts
    init_radii = [_compute_radius(i) for i in inits]
    sorted_idx = np.argsort(init_radii)[::-1]
    inits = [inits[i] for i in sorted_idx]
    
    # Time allocation
    # 60% to best, 25% to second, 15% to third
    weights = [0.6, 0.25, 0.15]
    
    for i, init in enumerate(inits):
        if time.time() > deadline - 5.0:
            break
        
        slice_duration = (deadline - start_time) * weights[i]
        slice_deadline = time.time() + slice_duration
        slice_deadline = min(slice_deadline, deadline - 1.0)
        
        c_opt, r_opt = _local_search(init, slice_deadline, rng)
        
        if r_opt > best_radius:
            best_radius = r_opt
            best_centers = c_opt.copy()
            
    # Final polish on best found
    if best_centers is not None and time.time() < deadline - 2.0:
        c_final, r_final = _local_search(best_centers, deadline - 0.5, rng)
        if r_final > best_radius:
            best_radius = r_final
            best_centers = c_final.copy()
            
    if best_centers is None:
        best_centers = inits[0]
        
    return best_centers

# EVOLVE-BLOCK-END
