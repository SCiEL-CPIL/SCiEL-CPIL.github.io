import numpy as np
import time

N = 50

def construct():
    start_time = time.time()
    deadline = 110.0 # Leave 10s margin for overhead/return
    
    # 1. Initialization: Hexagonal Lattice
    # Estimate r
    # Area per circle in hex packing ~ 2.31 * r^2
    # Total area 1.0
    # 50 * 2.31 * r^2 = 1.0 => r = sqrt(1/115.5) ~ 0.093 (too high, boundary effects)
    # Best known ~ 0.071.
    # Let's start with r=0.07
    r_est = 0.07
    spacing = 2 * r_est
    
    # Generate hex grid points
    # Hex grid basis vectors
    v1 = np.array([spacing, 0])
    v2 = np.array([spacing/2, spacing * np.sqrt(3)/2])
    
    # Generate a large range of indices
    max_idx = 20
    points = []
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            p = i * v1 + j * v2
            # Center the grid roughly
            # The square is [-0.5, 0.5]
            # We want to fill it
            if -0.5 <= p[0] <= 0.5 and -0.5 <= p[1] <= 0.5:
                points.append(p)
                
    points = np.array(points)
    
    # If we have more than N, randomly select N
    if len(points) > N:
        idx = np.random.choice(len(points), N, replace=False)
        centers = points[idx]
    else:
        # If too few, add random points
        centers = np.vstack([points, np.random.uniform(-0.5, 0.5, size=(N - len(points), 2))])
        
    # 2. Optimization: Simulated Annealing
    # Objective: Maximize r
    # r = min(min_dist/2, min_wall_dist)
    
    def calc_r(centers):
        # min distance between pairs
        # Use scipy.spatial.distance or manual
        # For N=50, manual is fine if vectorized per circle
        min_d = float('inf')
        for i in range(N):
            dists = np.linalg.norm(centers[i] - centers, axis=1)
            dists[i] = float('inf') # ignore self
            min_d = min(min_d, np.min(dists))
        
        # min distance to walls
        # distance to 0.5
        wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
        min_wall = np.min(wall_dists)
        
        return min(min_d / 2.0, min_wall)
        
    current_centers = centers.copy()
    current_r = calc_r(current_centers)
    best_centers = current_centers.copy()
    best_r = current_r
    
    T = 0.01
    T_min = 1e-6
    alpha = 0.9999
    
    while time.time() < deadline:
        # Pick random circle
        i = np.random.randint(N)
        
        # Generate move
        # Gaussian perturbation
        step_size = 0.01
        delta = np.random.normal(0, step_size, 2)
        new_centers = current_centers.copy()
        new_centers[i] += delta
        
        # Check if inside square?
        # If not, the wall distance will be negative, reducing r.
        # We can just calculate new_r.
        
        new_r = calc_r(new_centers)
        
        # Acceptance criterion
        if new_r > current_r:
            current_centers = new_centers
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_centers = current_centers.copy()
        else:
            # Accept with probability exp((new_r - current_r) / T)
            # Note: new_r - current_r is negative
            prob = np.exp((new_r - current_r) / T)
            if np.random.random() < prob:
                current_centers = new_centers
                current_r = new_r
        
        T *= alpha
        if T < T_min:
            T = T_min

    return best_centers
