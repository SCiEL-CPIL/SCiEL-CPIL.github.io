import numpy as np
import time

N = 100

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 140.0  # 10s margin
    
    # 1. Initialization: 5x5x4 grid
    # x: 5 points, y: 5 points, z: 4 points
    xs = np.linspace(-0.5 + 0.5/5, 0.5 - 0.5/5, 5)
    ys = np.linspace(-0.5 + 0.5/5, 0.5 - 0.5/5, 5)
    zs = np.linspace(-0.5 + 0.5/4, 0.5 - 0.5/4, 4)
    
    # Create meshgrid and flatten
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    C = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    # Ensure we have exactly N points
    # 5*5*4 = 100 points exactly.
    
    # Calculate initial radius
    def calc_radius(C):
        # Pairwise distances
        diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        
        # Boundary distances
        # Distance to closest wall
        # Wall is at +/- 0.5
        # dist to wall = 0.5 - |coord|
        boundary_dists = 0.5 - np.abs(C)
        min_boundary = np.min(boundary_dists)
        
        return min(min_dist / 2.0, min_boundary)
    
    best_C = C.copy()
    best_r = calc_radius(best_C)
    curr_r = best_r
    
    # SA Parameters
    T_start = 0.005
    T_end = 0.00001
    steps = 150000
    step_size_start = 0.05
    step_size_end = 0.001
    
    # Pre-compute indices for random selection
    # We pick a random point i and random axis k
    
    for t in range(steps):
        # Check time
        if t % 1000 == 0:
            if time.time() > deadline:
                break
        
        # Annealing parameters
        alpha = t / steps
        T = T_start * (T_end / T_start) ** alpha
        step_size = step_size_start * (step_size_end / step_size_start) ** alpha
        
        # Propose move
        i = np.random.randint(N)
        k = np.random.randint(3)
        
        old_val = C[i, k]
        delta = np.random.uniform(-step_size, step_size)
        C[i, k] = old_val + delta
        
        # Calculate new radius
        new_r = calc_radius(C)
        
        # Acceptance
        if new_r > curr_r or np.random.random() < np.exp((new_r - curr_r) / T):
            curr_r = new_r
            if new_r > best_r:
                best_r = new_r
                best_C = C.copy()
        else:
            # Reject
            C[i, k] = old_val
    
    return best_C

# EVOLVE-BLOCK-END
