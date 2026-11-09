"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist

def _compute_radius(points):
    """Compute the feasible radius for a set of points."""
    if points is None or len(points) == 0:
        return 0.0
    dist_boundary = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(dist_boundary)
    
    d = pdist(points)
    min_dist = np.min(d)
    min_pair = min_dist / 2.0
    
    return min(min_boundary, min_pair)

def _objective(points):
    """
    Objective function to maximize: the minimum of (pairwise distances/2, boundary distances).
    We return the negative of this for minimization, but here we'll just return the value.
    """
    dist_boundary = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(dist_boundary)
    d = pdist(points)
    min_dist = np.min(d)
    min_pair = min_dist / 2.0
    return min(min_boundary, min_pair)

def _local_optimize(points, max_iter=500, step_size=0.01, cooling=0.995, temperature=0.05):
    """
    Local optimization using a simple gradient-free hill climbing / simulated annealing approach.
    We perturb points and accept if the objective improves or with some probability (annealing).
    """
    current_points = points.copy()
    current_obj = _objective(current_points)
    
    # Pre-compute nearest neighbors? No, just use pdist.
    # To speed up, we can approximate the gradient by checking only nearest neighbors.
    # But for 300 points, pdist is ~45k distances, which is fast enough.
    
    best_points = current_points.copy()
    best_obj = current_obj
    
    for i in range(max_iter):
        # Generate random perturbations
        # Perturb a random subset of points to reduce noise
        n_perturb = max(5, N // 10)
        idx = np.random.choice(N, n_perturb, replace=False)
        
        # Random directions
        angles = np.random.uniform(0, 2 * np.pi, n_perturb)
        magnitudes = np.random.uniform(0, step_size, n_perturb)
        
        new_points = current_points.copy()
        new_points[idx, 0] += magnitudes * np.cos(angles)
        new_points[idx, 1] += magnitudes * np.sin(angles)
        
        new_obj = _objective(new_points)
        
        # Accept if better, or with probability exp((new_obj - current_obj) / temperature)
        if new_obj > current_obj:
            current_points = new_points
            current_obj = new_obj
            if current_obj > best_obj:
                best_points = current_points.copy()
                best_obj = current_obj
        else:
            # Simulated annealing acceptance
            prob = np.exp((new_obj - current_obj) / max(temperature, 1e-9))
            if np.random.rand() < prob:
                current_points = new_points
                current_obj = new_obj
        
        # Cool down
        temperature *= cooling
        step_size *= 0.995
        
    return best_points, best_obj

def _initial_hexagonal():
    """Generate a good initial hexagonal lattice packing."""
    # Estimate spacing
    # Area of unit circle = pi. Area per circle ~ pi * r^2.
    # N * pi * r^2 <= pi => r <= 1/sqrt(N)
    # But hexagonal packing density is sqrt(3)/2.
    # Effective area per circle = (2r)^2 * sqrt(3)/2 = 2 * sqrt(3) * r^2
    # N * 2 * sqrt(3) * r^2 = pi => r = sqrt(pi / (2 * sqrt(3) * N))
    r_est = np.sqrt(np.pi / (2 * np.sqrt(3) * N))
    spacing = 2 * r_est
    
    # Generate hex lattice points
    points = []
    k = int(1.0 / spacing) + 5
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3) / 2.0
            points.append([x, y])
    points = np.array(points)
    
    # Filter points inside the circle (with margin)
    radii = np.linalg.norm(points, axis=1)
    mask = radii < 1.0 - r_est
    points = points[mask]
    
    # If we have more than N, keep the closest to center
    if len(points) > N:
        radii = np.linalg.norm(points, axis=1)
        idx = np.argsort(radii)[:N]
        points = points[idx]
    else:
        # If we have fewer, pad with random points? Or just return what we have.
        # But we need exactly N. Let's add random points if needed.
        if len(points) < N:
            n_needed = N - len(points)
            random_points = np.random.uniform(-0.9, 0.9, (n_needed, 2))
            points = np.vstack([points, random_points])
    
    return points

def _initial_random():
    """Generate a random initial packing."""
    points = np.random.uniform(-0.9, 0.9, (N, 2))
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    # Start with a good hexagonal initial guess
    best_points = _initial_hexagonal()
    best_r = _objective(best_points)
    
    # Perform multiple local optimizations from different starts
    n_starts = 5
    
    for i in range(n_starts):
        if time.time() > deadline - 20:
            break
        
        if i == 0:
            # First start is the hexagonal one
            current_points = best_points.copy()
        else:
            # Subsequent starts are perturbed versions of the best found so far
            # Or random? Let's try perturbed best to stay in a good basin.
            current_points = best_points.copy()
            # Add some noise
            noise = np.random.normal(0, 0.05, (N, 2))
            current_points += noise
            # Ensure points are inside the circle
            radii = np.linalg.norm(current_points, axis=1)
            mask = radii > 0.95
            current_points[mask] *= 0.95 / radii[mask]
        
        # Local optimization
        opt_points, opt_r = _local_optimize(
            current_points, 
            max_iter=200, 
            step_size=0.02, 
            cooling=0.99, 
            temperature=0.02
        )
        
        if opt_r > best_r:
            best_r = opt_r
            best_points = opt_points
    
    # Final check
    if time.time() > deadline - 5:
        return best_points
    
    return best_points
# EVOLVE-BLOCK-END
