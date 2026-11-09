# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.spatial.distance as spd
from scipy.optimize import minimize

def calculate_score(points):
    """
    Calculate the metric R = (d_min / d_max)^2
    """
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

def objective_function(flat_points):
    """
    Objective function for minimization: minimize d_max / d_min
    which is equivalent to maximizing d_min / d_max
    """
    points = flat_points.reshape(-1, 2)
    dists = spd.pdist(points)
    if len(dists) == 0:
        return 1e10
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_min == 0:
        return 1e10
    return d_max / d_min

def generate_hex_lattice(n_points=16):
    """
    Generate a hexagonal lattice subset of n_points, centered.
    """
    # Generate a larger hex lattice and pick the n_points closest to center
    # Hex lattice basis vectors
    a = 1.0
    b = a * np.sqrt(3) / 2
    
    points_list = []
    # Generate points in a hexagonal grid
    for i in range(-4, 5):
        for j in range(-4, 5):
            x = i * a + (j % 2) * (a / 2)
            y = j * b
            points_list.append([x, y])
    
    points = np.array(points_list)
    # Center the lattice
    points -= np.mean(points, axis=0)
    
    # Sort by distance from center
    dists_from_center = np.linalg.norm(points, axis=1)
    sorted_indices = np.argsort(dists_from_center)
    
    # Take the n_points closest to center
    hex_points = points[sorted_indices[:n_points]]
    
    # Optional: slight random perturbation to break symmetry
    # hex_points += np.random.randn(n_points, 2) * 0.01
    
    return hex_points

def local_search(points, time_limit):
    """
    Perform local search with simulated annealing to maximize the score.
    """
    n = points.shape[0]
    current_points = points.copy()
    current_score = calculate_score(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    start_time = time.time()
    
    # Simulated Annealing parameters
    temp = 0.01
    min_temp = 1e-6
    cooling_rate = 0.999
    
    # Step size for perturbations
    step_size = 0.1
    
    while time.time() - start_time < time_limit:
        # Pick a random point to move
        idx = np.random.randint(n)
        
        # Generate a random perturbation
        # We scale the perturbation based on the current spread of points
        # to ensure we explore the relevant scale
        spread = np.std(current_points)
        if spread == 0:
            spread = 1.0
            
        # Perturbation magnitude
        delta = np.random.randn(2) * step_size * spread
        
        new_points = current_points.copy()
        new_points[idx] += delta
        
        new_score = calculate_score(new_points)
        
        # Acceptance criterion
        if new_score > current_score:
            current_points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
        else:
            # Simulated annealing acceptance
            if temp > min_temp:
                delta_score = new_score - current_score
                # Probability of accepting a worse solution
                # Since scores are small (e.g., 0.05), temp should be small
                prob = np.exp(delta_score / temp)
                if np.random.rand() < prob:
                    current_points = new_points
                    current_score = new_score
        
        # Cool down
        temp *= cooling_rate
        
        # Adaptive step size: if we've been improving, keep step size.
        # If stuck, maybe reduce step size to fine-tune.
        # For simplicity, we keep it constant or slowly reduce it.
        # Let's reduce step size slowly to converge.
        step_size *= 0.9999
        
    return best_points

def refine_points(points, time_limit):
    """
    Use Nelder-Mead to refine points for higher precision.
    """
    n = points.shape[0]
    start_time = time.time()
    
    # Flatten points for scipy.optimize.minimize
    x0 = points.flatten()
    
    # Use a tighter tolerance and more iterations
    # We want to minimize d_max / d_min
    res = minimize(
        objective_function, 
        x0, 
        method='Nelder-Mead', 
        options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-12}
    )
    
    refined_points = res.x.reshape(-1, 2)
    
    # If time allows, do a second pass with even tighter tolerance
    if time.time() - start_time < time_limit * 0.5:
        res2 = minimize(
            objective_function, 
            res.x, 
            method='Nelder-Mead', 
            options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-14}
        )
        refined_points = res2.x.reshape(-1, 2)
        
    return refined_points

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) coordinates.

    """
    n = 16
    d = 2
    
    # Deadline guard: 630s limit, use 600s for safety
    deadline = time.time() + 590
    
    # Initial guess: 4x4 grid
    # This provides a good baseline ratio
    x = np.linspace(-1.5, 1.5, 4)
    y = np.linspace(-1.5, 1.5, 4)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    
    # Run local search
    # We can run multiple restarts if time permits
    best_points = points
    best_score = calculate_score(points)
    
    # Try a few different initializations
    inits = []
    
    # 1. Standard Grid
    inits.append(points)
    
    # 2. Hexagonal lattice subset (better geometry)
    pts_hex = generate_hex_lattice(n)
    inits.append(pts_hex)
    
    # 3. Random
    np.random.seed(42)
    pts_rand = np.random.randn(n, d)
    pts_rand -= np.mean(pts_rand, axis=0)
    inits.append(pts_rand)
    
    # Allocate time for each init
    num_inits = len(inits)
    time_per_init = (deadline - time.time()) / (num_inits + 1)  # +1 for final polish
    
    for init_pts in inits:
        if time.time() > deadline - 30:  # Leave time for polish
            break
            
        # Run local search on this init
        # Give it a portion of the time
        end_time_for_this = time.time() + time_per_init
        res_points = local_search(init_pts, end_time_for_this - time.time())
        res_score = calculate_score(res_points)
        
        if res_score > best_score:
            best_score = res_score
            best_points = res_points
    
    # If time remains, do a final high-precision polish on the best result
    if time.time() < deadline - 5:
        remaining_time = deadline - 5 - time.time()
        best_points = refine_points(best_points, remaining_time)
        
    return best_points

# EVOLVE-BLOCK-END
