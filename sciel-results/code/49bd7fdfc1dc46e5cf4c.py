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

def hexagonal_grid_init(n):
    """Initialize points on a hexagonal grid pattern for 16 points in 2D"""
    # Create a hexagonal packing arrangement
    # For 16 points, we can use a 4x4 grid with hexagonal offsets
    points = []
    rows = 4
    cols = 4
    dx = 1.0
    dy = dx * np.sqrt(3) / 2
    
    for i in range(rows):
        for j in range(cols):
            x = j * dx + (i % 2) * (dx / 2)
            y = i * dy
            points.append([x, y])
    
    return np.array(points[:n])

def simulated_annealing(points, time_limit, seed=42):
    """
    Perform simulated annealing to maximize the min/max distance ratio.
    """
    n = points.shape[0]
    current_points = points.copy()
    current_score = calculate_score(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    start_time = time.time()
    temp = 0.1  # Initial temperature
    min_temp = 1e-6
    cooling_rate = 0.9995
    
    np.random.seed(seed)
    
    while time.time() - start_time < time_limit:
        # Check if we should stop
        if temp < min_temp:
            break
            
        # Generate a random perturbation
        idx = np.random.randint(n)
        delta = np.random.randn(2) * temp
        new_points = current_points.copy()
        new_points[idx] += delta
        
        new_score = calculate_score(new_points)
        
        # Acceptance criterion
        if new_score > current_score:
            current_points = new_points
            current_score = new_score
            if current_score > best_score:
                best_points = current_points.copy()
                best_score = current_score
        else:
            # Accept with probability exp((new_score - current_score) / temp)
            if np.random.rand() < np.exp((new_score - current_score) / temp):
                current_points = new_points
                current_score = new_score
        
        temp *= cooling_rate
    
    return best_points, best_score

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    n = 16
    d = 2
    
    # Set a deadline with safety margin (630s limit, use 580s)
    deadline = time.time() + 580
    
    # Start with hexagonal grid initialization
    points = hexagonal_grid_init(n)
    best_points = points.copy()
    best_score = calculate_score(best_points)
    
    # Run simulated annealing with multiple restarts
    num_restarts = 5
    time_per_restart = (deadline - time.time()) / num_restarts
    
    for restart in range(num_restarts):
        # For subsequent restarts, add some randomization to the initial grid
        if restart > 0:
            np.random.seed(42 + restart)
            perturbed = points + np.random.randn(n, d) * 0.1
        else:
            perturbed = points.copy()
        
        result_points, result_score = simulated_annealing(perturbed, time_per_restart, seed=42 + restart)
        
        if result_score > best_score:
            best_points = result_points
            best_score = result_score
        
        # Check if we're running out of time
        if time.time() > deadline - 30:
            break
    
    return best_points


# EVOLVE-BLOCK-END
