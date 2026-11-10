"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_radius(centers):
    """Compute the maximum common radius for the given centers."""
    n = len(centers)
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    # Distance to walls: square is [-0.5, 0.5]^2
    min_x = 0.5 - np.max(centers[:, 0])
    min_y = 0.5 - np.max(centers[:, 1])
    min_x2 = np.min(centers[:, 0]) + 0.5
    min_y2 = np.min(centers[:, 1]) + 0.5
    min_wall = min(min_x, min_y, min_x2, min_y2)
    
    return min(min_pair, min_wall)


def _initial_hexagonal_lattice():
    """Generate a hexagonal lattice initial configuration."""
    points = []
    # Use rows with hexagonal offset
    # 50 circles: try 7 rows with ~7-8 per row
    rows = [7, 7, 7, 7, 7, 7, 8]  # 50 total
    
    # Calculate spacing to fit in [-0.5, 0.5]
    # Horizontal spacing based on max row length
    max_row = max(rows)
    dx = 0.95 / max_row  # Leave small margin
    
    # Vertical spacing for hexagonal: dy = dx * sqrt(3)/2
    dy = dx * np.sqrt(3) / 2
    
    # Center vertically
    total_height = (len(rows) - 1) * dy
    y_start = -total_height / 2
    
    for i, count in enumerate(rows):
        y = y_start + i * dy
        # Offset odd rows by dx/2
        offset = (dx / 2) if i % 2 == 1 else 0
        # Center the row horizontally
        row_width = (count - 1) * dx
        x_start = -row_width / 2
        
        for j in range(count):
            x = x_start + j * dx + offset
            points.append([x, y])
    
    return np.array(points[:N])


def _simulated_annealing(centers, start_time, deadline):
    """Run simulated annealing to improve circle packing."""
    current = centers.copy()
    current_radius = _compute_radius(current)
    best = current.copy()
    best_radius = current_radius
    
    # SA parameters
    T0 = 0.01  # Initial temperature
    T_end = 0.0001  # Final temperature
    cooling = 0.9995  # Cooling rate
    T = T0
    
    rng = np.random.RandomState(42)
    
    while T > T_end:
        if time.time() - start_time > deadline - 5:
            break
        
        # Generate a perturbation: move a random point
        idx = rng.randint(N)
        # Perturbation size proportional to temperature
        step_size = T * 0.1
        delta = rng.normal(0, step_size, 2)
        
        new_point = current[idx].copy() + delta
        # Keep within bounds
        new_point[0] = np.clip(new_point[0], -0.45, 0.45)
        new_point[1] = np.clip(new_point[1], -0.45, 0.45)
        
        new_centers = current.copy()
        new_centers[idx] = new_point
        new_radius = _compute_radius(new_centers)
        
        # Acceptance criterion
        if new_radius >= current_radius:
            current = new_centers
            current_radius = new_radius
            if new_radius > best_radius:
                best = new_centers.copy()
                best_radius = new_radius
        else:
            # Accept with probability exp((new - current) / T)
            delta_obj = new_radius - current_radius
            prob = np.exp(delta_obj / T)
            if rng.random() < prob:
                current = new_centers
                current_radius = new_radius
        
        T *= cooling
    
    return best, best_radius


def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    # Initial configuration
    centers = _initial_hexagonal_lattice()
    best_centers = centers.copy()
    best_radius = _compute_radius(centers)
    
    # Run simulated annealing
    best_centers, best_radius = _simulated_annealing(centers, start_time, deadline)
    
    # If time remains, try a few more SA runs with different random seeds
    if time.time() - start_time < deadline - 10:
        for seed in [123, 456, 789]:
            if time.time() - start_time > deadline - 5:
                break
            rng = np.random.RandomState(seed)
            # Jitter the initial configuration
            jittered = best_centers + rng.normal(0, 0.005, (N, 2))
            jittered[:, 0] = np.clip(jittered[:, 0], -0.45, 0.45)
            jittered[:, 1] = np.clip(jittered[:, 1], -0.45, 0.45)
            
            # Quick SA with fewer iterations
            current = jittered.copy()
            current_radius = _compute_radius(current)
            local_best = jittered.copy()
            local_best_radius = current_radius
            
            T0 = 0.005
            T_end = 0.0005
            cooling = 0.999
            T = T0
            
            while T > T_end and time.time() - start_time < deadline - 3:
                idx = rng.randint(N)
                step_size = T * 0.1
                delta = rng.normal(0, step_size, 2)
                new_point = current[idx].copy() + delta
                new_point[0] = np.clip(new_point[0], -0.45, 0.45)
                new_point[1] = np.clip(new_point[1], -0.45, 0.45)
                
                new_centers = current.copy()
                new_centers[idx] = new_point
                new_radius = _compute_radius(new_centers)
                
                if new_radius >= current_radius:
                    current = new_centers
                    current_radius = new_radius
                    if new_radius > local_best_radius:
                        local_best = new_centers.copy()
                        local_best_radius = new_radius
                else:
                    delta_obj = new_radius - current_radius
                    prob = np.exp(delta_obj / T)
                    if rng.random() < prob:
                        current = new_centers
                        current_radius = new_radius
                
                T *= cooling
            
            if local_best_radius > best_radius:
                best_centers = local_best
                best_radius = local_best_radius
    
    return best_centers
# EVOLVE-BLOCK-END
