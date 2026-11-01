"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree

def _fibonacci_sphere(n):
    """Generate n points on a unit sphere using the Fibonacci lattice method."""
    points = np.zeros((n, 3))
    for i in range(n):
        z = 1 - 2 * (i + 0.5) / n
        r = np.sqrt(1 - z*z)
        theta = 2 * np.pi * i / n
        points[i, 0] = r * np.cos(theta)
        points[i, 1] = r * np.sin(theta)
        points[i, 2] = z
    return points

def _covering_radius_proxy(points, test_points):
    """
    Estimate the covering radius by finding the maximum Euclidean distance from 
    a set of test points on the sphere to the nearest configuration point.
    Minimizing this is equivalent to minimizing the angular covering radius.
    """
    tree = cKDTree(points)
    dists, _ = tree.query(test_points)
    return np.max(dists)

def construct():
    rng = np.random.default_rng(42)
    
    # 1. Initialize with Fibonacci Sphere (structured, high-quality start)
    points = _fibonacci_sphere(N)
    
    # 2. Generate test points for covering radius estimation
    # 10,000 points provide a good approximation of the true covering radius
    n_test = 10000
    test_points = _fibonacci_sphere(n_test)
    
    # 3. Simulated Annealing
    current_energy = _covering_radius_proxy(points, test_points)
    best_points = points.copy()
    best_energy = current_energy
    
    T = 0.1  # Initial temperature
    T_min = 1e-6
    cooling_rate = 0.9995
    n_iter = 40000
    
    for i in range(n_iter):
        # Pick a random point to move
        idx = rng.integers(0, N)
        
        # Generate a random move scaled by temperature
        step_size = T * 0.1
        move = rng.standard_normal(3) * step_size
        
        new_point = points[idx] + move
        norm = np.linalg.norm(new_point)
        if norm < 1e-10:
            continue
        new_point = new_point / norm
        
        # Apply move
        old_point = points[idx].copy()
        points[idx] = new_point
        
        # Calculate new energy
        new_energy = _covering_radius_proxy(points, test_points)
        
        # Metropolis criterion
        delta_e = new_energy - current_energy
        if delta_e < 0 or rng.random() < np.exp(-delta_e / T):
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_points = points.copy()
        else:
            # Reject move
            points[idx] = old_point
            
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min

    return best_points
# EVOLVE-BLOCK-END
