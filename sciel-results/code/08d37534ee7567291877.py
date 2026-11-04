"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial.distance import pdist

N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import time

def _compute_radius(centers):
    """
    Compute the maximum radius for the given centers.
    centers: shape (N, 3)
    Returns: float, the minimum radius that fits all spheres.
    """
    # 1. Pairwise distances between centers
    # pdist returns condensed distance matrix, shape (N*(N-1)/2,)
    if N > 1:
        d = pdist(centers)
        min_dist_centers = np.min(d)
    else:
        min_dist_centers = np.inf
    
    # 2. Distances to walls
    # Cube is [-0.5, 0.5]^3
    # Distance from center c_i to wall x=0.5 is 0.5 - c_i[0]
    # Distance from center c_i to wall x=-0.5 is c_i[0] + 0.5
    # We need the minimum distance to any wall for each sphere.
    # dist_to_wall_i = min(0.5 - c_i[x], c_i[x] + 0.5) for x in {0,1,2}
    # This is equivalent to 0.5 - max(|c_i[x]|)
    abs_coords = np.abs(centers)
    max_abs_coord = np.max(abs_coords, axis=1)
    min_dist_walls = 0.5 - max_abs_coord
    
    # The radius is limited by both center-center distance and wall distance
    # r = min(min_dist_centers / 2, min_dist_walls)
    # Note: The task defines r = min(min_{i<j} |c_i - c_j| / 2, min_i min_k (1/2 - |c_ik|))
    
    r = np.min([min_dist_centers / 2.0, np.min(min_dist_walls)])
    return r

def construct():
    start_time = time.time()
    timeout = 115.0  # Safety margin for 120s limit
    
    # Initial random configuration
    # Uniform in [-0.5, 0.5]
    centers = np.random.uniform(-0.5, 0.5, size=(N, 3))
    
    current_r = _compute_radius(centers)
    best_r = current_r
    best_centers = centers.copy()
    
    # Annealing parameters
    # Start temperature high enough to allow overlaps, end low for precision
    T_start = 0.1
    T_end = 1e-6
    # Number of iterations depends on speed. 
    # pdist for N=100 is very fast. We can afford many iterations.
    # Let's estimate: pdist(100,3) takes ~microseconds. 
    # 100,000 iterations should take < 10 seconds.
    # We will run until timeout or max iterations.
    max_iters = 200000
    
    # Adaptive step size
    # Initial step size should be comparable to the desired radius
    step_size = 0.05
    min_step_size = 1e-5
    
    for i in range(max_iters):
        if time.time() - start_time > timeout:
            break
            
        # Annealing schedule: Exponential cooling
        # T = T_start * (T_end / T_start) ** (i / max_iters)
        # Or simpler: T = T_start * (1 - i/max_iters)
        # Exponential is better for SA
        T = T_start * (T_end / T_start) ** (i / max_iters)
        
        # Generate a random move
        # Move one random sphere by a random vector
        idx = np.random.randint(N)
        # Gaussian noise
        noise = np.random.normal(0, step_size, size=3)
        
        new_centers = centers.copy()
        new_centers[idx] += noise
        
        # Evaluate new radius
        new_r = _compute_radius(new_centers)
        
        # Acceptance probability
        if new_r >= current_r:
            # Always accept improvement
            centers = new_centers
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_centers = centers.copy()
        else:
            # Accept worse move with probability exp((new_r - current_r) / T)
            # Note: We are maximizing r.
            # Delta = new_r - current_r (negative)
            # Prob = exp(Delta / T)
            delta = new_r - current_r
            # Avoid overflow/underflow issues, though T is small so Delta/T is very negative
            if delta / T > -700: # exp(-700) is approx 1e-304
                prob = np.exp(delta / T)
            else:
                prob = 0.0
                
            if np.random.random() < prob:
                centers = new_centers
                current_r = new_r
        
        # Adapt step size
        # If we've been accepting many moves, maybe increase step? 
        # If we've been rejecting many, decrease step?
        # A simple heuristic: shrink step size slowly as T drops
        step_size = max(min_step_size, T * 0.5)
        
    # Final polish: try to fix any small boundary violations or overlaps if close
    # A simple local search with very small steps can help
    T_polish = 1e-8
    step_polish = 1e-4
    for i in range(1000):
        if time.time() - start_time > timeout:
            break
        idx = np.random.randint(N)
        noise = np.random.normal(0, step_polish, size=3)
        new_centers = best_centers.copy()
        new_centers[idx] += noise
        new_r = _compute_radius(new_centers)
        if new_r > best_r:
            best_r = new_r
            best_centers = new_centers
            
    return best_centers

# EVOLVE-BLOCK-END
