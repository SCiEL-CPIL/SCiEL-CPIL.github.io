"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
from scipy.stats import qmc

def get_test_points(M):
    """Generate M points on the unit sphere using Fibonacci spiral for testing covering radius."""
    i = np.arange(M) + 0.5
    z = 1.0 - 2.0 * i / M
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def covering_radius(points, test_points):
    """
    Calculate the covering radius (approximated by max distance to nearest point in points 
    for all points in test_points).
    points: (N, 3)
    test_points: (M, 3)
    Returns: float (angular distance in radians)
    """
    # Euclidean distance between test points and config points
    # dist_sq[i, j] = ||test_points[i] - points[j]||^2
    # Using broadcasting
    # test_points[:, np.newaxis, :] shape (M, 1, 3)
    # points[np.newaxis, :, :] shape (1, N, 3)
    diff = test_points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    
    # Min distance from each test point to any config point
    min_dist_sq = np.min(dist_sq, axis=1)
    
    # The covering radius is the max of the min distances
    max_dist_sq = np.max(min_dist_sq)
    
    # Convert Euclidean distance to angular distance
    # ||a - b|| = 2 * sin(theta / 2) for unit vectors
    # theta = 2 * arcsin(||a - b|| / 2)
    # If max_dist_sq is 0, theta is 0.
    # Clamp to [0, 4] to avoid numerical issues in arcsin (max dist is 2)
    max_dist = np.sqrt(np.clip(max_dist_sq, 0, 4))
    theta = 2.0 * np.arcsin(max_dist / 2.0)
    return theta

def normalize(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms == 0] = 1.0
    return points / norms

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # 1. Generate high-quality initial points using Sobol sequence
    # Sobol sequence in [0, 1)^2 -> map to sphere
    sampler = qmc.Sobol(d=2, scramble=False)
    samples = sampler.random(N)
    
    # Map [0, 1)^2 to sphere
    # theta (azimuth) = 2*pi * u1
    # phi (polar) = arccos(2*u2 - 1)
    theta = 2.0 * np.pi * samples[:, 0]
    cos_phi = 2.0 * samples[:, 1] - 1.0
    phi = np.arccos(cos_phi)
    
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = cos_phi
    points = np.column_stack([x, y, z])
    
    # 2. Generate test points for covering radius evaluation
    M_test = 10000
    test_pts = get_test_points(M_test)
    
    best_points = points.copy()
    best_radius = covering_radius(points, test_pts)
    
    # 3. Optimization Loop
    # We will use Nelder-Mead for local optimization, potentially with restarts.
    
    def objective(flat_x):
        pts = flat_x.reshape(N, 3)
        pts = normalize(pts)
        return covering_radius(pts, test_pts)
    
    # Initial optimization
    current_points = points.copy()
    current_radius = best_radius
    
    # We can try a few different strategies or restarts if time allows
    # Strategy 1: Nelder-Mead on the current best
    
    if time.time() < deadline:
        res = minimize(objective, current_points.flatten(), method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10})
        
        if res.success and res.fun < current_radius:
            current_points = normalize(res.x.reshape(N, 3))
            current_radius = res.fun
            best_points = current_points.copy()
            best_radius = current_radius

    # Strategy 2: Try a different initial guess (Fibonacci) and optimize
    if time.time() < deadline:
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        fib_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
        res2 = minimize(objective, fib_points.flatten(), method='Nelder-Mead', 
                        options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10})
        
        if res2.success and res2.fun < current_radius:
            current_points = normalize(res2.x.reshape(N, 3))
            current_radius = res2.fun
            best_points = current_points.copy()
            best_radius = current_radius

    # Strategy 3: Random restarts with local polish if time remains
    # We can try random perturbations of the best points
    
    if time.time() < deadline:
        rng = np.random.default_rng(42)
        attempts = 0
        while time.time() < deadline:
            # Random perturbation
            noise = rng.normal(0, 0.5, size=(N, 3))
            perturbed = best_points + noise
            perturbed = normalize(perturbed)
            
            res3 = minimize(objective, perturbed.flatten(), method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-9})
            
            if res3.success and res3.fun < best_radius:
                best_points = normalize(res3.x.reshape(N, 3))
                best_radius = res3.fun
                attempts += 1
                if attempts > 20: break # Cap restarts

    return best_points
# EVOLVE-BLOCK-END
