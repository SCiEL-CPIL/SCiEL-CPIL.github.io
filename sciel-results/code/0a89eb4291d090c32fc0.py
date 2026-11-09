# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.spatial.distance as spd


def calculate_score(points):
    """Calculate R = (d_min / d_max)^2"""
    n = len(points)
    if n < 2:
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


def hexagonal_init(n, seed=0):
    """Initialize points on a hexagonal grid pattern with some randomness."""
    rng = np.random.default_rng(seed)
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
    
    points = np.array(points[:n])
    # Add small random perturbation
    points += rng.normal(0, 0.05, points.shape)
    return points


def circular_init(n, seed=0):
    """Initialize points on a circle with some randomness."""
    rng = np.random.default_rng(seed)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + rng.normal(0, 0.1, n)
    radii = 1.0 + rng.normal(0, 0.1, n)
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack([x, y])


def simulated_annealing(points, start_time, max_time, seed=0):
    """Run simulated annealing to maximize R."""
    rng = np.random.default_rng(seed)
    n = len(points)
    
    current_points = points.copy()
    current_score = calculate_score(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    # Temperature schedule
    t_start = 0.5
    t_end = 0.001
    total_steps = 200000
    
    for step in range(total_steps):
        # Check time
        if time.time() - start_time > max_time:
            break
        
        # Temperature
        progress = step / total_steps
        temperature = t_start * (t_end / t_start) ** progress
        
        # Generate a random move
        idx = rng.integers(n)
        delta = rng.normal(0, temperature, 2)
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
            # Accept with probability exp((new_score - current_score) / temperature)
            # Since we're maximizing, delta_score = new - current <= 0
            delta_score = new_score - current_score
            if temperature > 0:
                prob = np.exp(delta_score / temperature)
                if rng.random() < prob:
                    current_points = new_points
                    current_score = new_score
    
    return best_points, best_score


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    start_time = time.time()
    max_time = 580.0  # Safety margin before 630s limit
    
    n = 16
    best_points = None
    best_score = -1.0
    
    # Multiple restarts with different initializations
    n_restarts = 8
    
    for restart in range(n_restarts):
        remaining_time = max_time - (time.time() - start_time)
        if remaining_time < 30:
            break
        
        # Choose initialization
        if restart % 2 == 0:
            init_points = hexagonal_init(n, seed=restart * 100)
        else:
            init_points = circular_init(n, seed=restart * 100 + 50)
        
        # Run simulated annealing with time budget for this restart
        time_per_restart = remaining_time / (n_restarts - restart)
        points, score = simulated_annealing(
            init_points, start_time, start_time + time_per_restart, seed=restart * 1000
        )
        
        if score > best_score:
            best_score = score
            best_points = points.copy()
    
    # If no valid result, return hexagonal grid
    if best_points is None:
        best_points = hexagonal_init(n, seed=42)
    
    # Ensure shape is (16, 2)
    assert best_points.shape == (16, 2)
    
    return best_points


# EVOLVE-BLOCK-END
