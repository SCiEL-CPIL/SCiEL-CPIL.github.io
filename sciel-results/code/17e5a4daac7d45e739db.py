import numpy as np
import time

N = 300

def _compute_radius(pos):
    """
    Computes the maximum radius of non-overlapping circles centered at pos
    inside a unit circle.
    """
    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    # Distance to boundary
    norms = np.sqrt(np.sum(pos**2, axis=1))
    min_boundary = 1.0 - np.max(norms)
    
    return min(min_dist / 2.0, min_boundary)

def _init_hex_lattice(N):
    """
    Initializes points in a hexagonal lattice pattern within the unit circle.
    """
    points = []
    # Generate a grid large enough to cover the circle
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    k = int(1.0 / spacing) + 2
    
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if np.hypot(x, y) <= 1.0:
                points.append([x, y])
    
    points = np.array(points)
    if len(points) < N:
        points = np.random.uniform(-1, 1, (N, 2))
        norms = np.hypot(points[:, 0], points[:, 1])
        points = points / norms[:, np.newaxis] * 0.9
    
    radii = np.hypot(points[:, 0], points[:, 1])
    indices = np.argsort(radii)
    points = points[indices[:N]]
    
    # Add small random noise to break symmetry
    points += np.random.normal(0, 0.01, points.shape)
    
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Initial position
    pos = _init_hex_lattice(N)
    r_curr = _compute_radius(pos)
    pos_best = pos.copy()
    r_best = r_curr
    
    # Simulated Annealing Parameters
    T = 0.005
    cooling_factor = 0.9995
    min_T = 1e-6
    
    steps_since_improvement = 0
    max_steps_no_improve = 5000
    
    while time.time() < deadline:
        # Generate move
        noise = np.random.normal(0, T, pos.shape)
        pos_new = pos + noise
        
        r_new = _compute_radius(pos_new)
        
        delta = r_new - r_curr
        
        # Metropolis acceptance
        if delta > 0:
            accept = True
        else:
            # Avoid overflow in exp
            exponent = delta / T
            if exponent < -100:
                accept = False
            else:
                accept = np.random.random() < np.exp(exponent)
                
        if accept:
            pos = pos_new
            r_curr = r_new
            
            if r_curr > r_best:
                r_best = r_curr
                pos_best = pos.copy()
                steps_since_improvement = 0
            else:
                steps_since_improvement += 1
        else:
            steps_since_improvement += 1
            
        # Cooling
        T *= cooling_factor
        
        # Restart if stuck or T too low
        if T < min_T or steps_since_improvement > max_steps_no_improve:
            # Reset T
            T = 0.005
            # Perturb the best position to escape local minimum
            pos = pos_best + np.random.normal(0, 0.05, pos_best.shape)
            r_curr = _compute_radius(pos)
            steps_since_improvement = 0
            max_steps_no_improve = 5000
            
    return pos_best
