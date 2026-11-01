"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def compute_radius(points):
    """
    Compute the largest common radius for the given centers.
    r = min(min_pairwise_dist / 2, min_dist_to_boundary)
    """
    if points.shape[0] != N:
        return 0.0
    # Check for finite values
    if not np.all(np.isfinite(points)):
        return 0.0
    
    # Distance to boundary
    dists_to_boundary = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(dists_to_boundary)
    
    if min_boundary < 0:
        return 0.0 # Invalid, circle outside
        
    # Pairwise distances
    # Vectorized computation
    # points is (N, 2)
    # dist_matrix[i,j] = ||p_i - p_j||
    # Using broadcasting
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Ignore diagonal (self distance)
    np.fill_diagonal(dists, np.inf)
    min_pairwise = np.min(dists)
    
    return min(min_pairwise / 2.0, min_boundary)

def initial_hexagonal_layout():
    """
    Generate a hexagonal lattice of points and select the N closest to the center.
    """
    # Estimate spacing for 50 points in unit circle
    # Area of unit circle = pi. Area per point ~ pi/50.
    # Hex area per point ~ s^2 * sqrt(3)/2.
    # s ~ sqrt(2*pi / (50*sqrt(3)/2)) = sqrt(4*pi / (50*sqrt(3)))
    s = np.sqrt(4.0 * np.pi / (N * np.sqrt(3.0)))
    
    # Generate grid
    points = []
    # Range large enough
    k = int(1.0 / s) + 3
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3.0) / 2.0
            if np.hypot(x, y) < 1.0:
                points.append([x, y])
                
    points = np.array(points)
    if len(points) < N:
        # Fallback to random if lattice is too sparse
        angles = np.random.uniform(0, 2*np.pi, N)
        radii = np.sqrt(np.random.uniform(0, 1, N))
        points = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
    else:
        # Select N closest to origin
        dists = np.linalg.norm(points, axis=1)
        indices = np.argsort(dists)[:N]
        points = points[indices]
        
    return points

def simulated_annealing(points, time_limit=110):
    """
    Optimize the packing using Simulated Annealing.
    """
    start_time = time.time()
    current_points = points.copy()
    current_score = compute_radius(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    # Initial temperature
    T = 0.01
    cooling_rate = 0.995
    max_steps = 1000000
    
    # Step size
    step_size = 0.05
    
    for step in range(max_steps):
        if time.time() - start_time > time_limit:
            break
            
        # Random move: pick a random point and move it
        i = np.random.randint(N)
        move_vec = np.random.normal(0, step_size, 2)
        new_points = current_points.copy()
        new_points[i] += move_vec
        
        # Check if new point is within container (allow slight overshoot for SA? No, better to penalize or reject)
        # For strict validity in final output, we must be inside.
        # During SA, we can allow slight violations if we penalize heavily, 
        # but simpler: reject if outside.
        if np.linalg.norm(new_points[i]) > 1.0:
            continue
            
        new_score = compute_radius(new_points)
        
        if new_score > current_score:
            current_points = new_points
            current_score = new_score
            if current_score > best_score:
                best_points = current_points.copy()
                best_score = current_score
        else:
            # Accept with probability exp((new_score - current_score) / T)
            # Since we maximize, delta = new - current is negative.
            delta = new_score - current_score
            if delta < 0:
                prob = np.exp(delta / T)
                if np.random.random() < prob:
                    current_points = new_points
                    current_score = new_score
        
        # Cool down
        T *= cooling_rate
        
        # Adapt step size? 
        # If T is very small, step size should also be small.
        step_size = max(0.001, 0.05 * T / 0.01)
        
    return best_points

def construct():
    # Deadline guard
    start_time = time.time()
    max_time = 115  # Leave 5s margin for 120s limit
    
    # 1. Initial Layout
    points = initial_hexagonal_layout()
    
    # 2. Optimization
    if time.time() - start_time < max_time:
        points = simulated_annealing(points, time_limit=max_time - (time.time() - start_time))
        
    # 3. Final Polish / Validation
    # Ensure the returned points are strictly valid
    # If any point is outside, move it in?
    # The SA rejects outside moves, so they should be inside.
    # But let's double check.
    norms = np.linalg.norm(points, axis=1)
    if np.any(norms > 1.0):
        # Scale down slightly if any are outside (should not happen)
        points = points / np.max(norms) * 0.99
        
    return points
# EVOLVE-BLOCK-END
