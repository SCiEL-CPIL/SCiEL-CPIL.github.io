"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _hex_lattice_init(n):
    """Initialize with a hexagonal lattice pattern."""
    # Estimate spacing s such that n circles fit in unit disk area
    # Density of hex lattice: 2 / (s^2 * sqrt(3))
    # n * s^2 * sqrt(3) / 2 = pi  =>  s = sqrt(2*pi / (n*sqrt(3)))
    s = np.sqrt(2 * np.pi / (n * np.sqrt(3)))
    
    # Generate points in a grid larger than the unit circle
    k = int(1.0 / s) + 2
    points = []
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3.0) / 2.0
            if np.hypot(x, y) <= 1.0:
                points.append([x, y])
    
    points = np.array(points)
    if len(points) < n:
        # Fallback to random if lattice is too sparse
        return np.random.uniform(-0.9, 0.9, (n, 2))
    
    # Select the n points closest to the center to form a compact cluster
    radii = np.hypot(points[:, 0], points[:, 1])
    idx = np.argsort(radii)
    return points[idx[:n]]

def _potential(x_flat):
    """
    Objective function to minimize.
    Sum of pairwise repulsions (1/d) and boundary repulsions (1/(1-r)).
    Minimizing this pushes circles apart and towards the boundary.
    """
    x = x_flat.reshape(-1, 2)
    
    # Pairwise distances
    # Use broadcasting for efficiency
    dists = np.linalg.norm(x[:, np.newaxis, :] - x[np.newaxis, :, :], axis=2)
    # Avoid division by zero
    np.fill_diagonal(dists, 1.0)
    
    # Pairwise repulsion energy
    # 1/d is smoother than 1/d^2 and less prone to numerical explosion
    pair_energy = np.sum(1.0 / dists)
    
    # Boundary repulsion energy
    radii = np.linalg.norm(x, axis=1)
    # Push radii towards 1.0
    # If r > 1, we are outside, penalty should be huge.
    # Using max(1-r, eps) ensures we penalize being inside too, but 
    # we really want to push to the boundary.
    # Actually, if we minimize 1/(1-r)^2, we push r to 1 from below.
    # If r > 1, 1-r is negative. Square is positive.
    # We want to avoid r > 1.
    # Let's use a potential that is 0 at r=1 and increases as r moves away.
    # But we specifically want to maximize the minimum distance to boundary.
    # So we want all circles to be as close to r=1 as possible, 
    # and the "weakest link" (closest to center) to be as far as possible.
    # Minimizing sum of 1/(1-r)^2 achieves this: it penalizes circles 
    # that are far from the boundary.
    
    eps = 1e-6
    # If r is close to 1, 1-r is small, term is large.
    # We want to MINIMIZE this? No, we want to MAXIMIZE the distance to boundary.
    # Wait.
    # If we minimize 1/(1-r)^2, we make (1-r)^2 large, so (1-r) large, so r small.
    # That's wrong. We want r to be close to 1.
    
    # Correction:
    # We want to maximize r_i for all i.
    # So we should minimize (1 - r_i).
    # But (1-r_i) is 0 at boundary.
    # If we minimize sum(1-r_i), we push all to boundary.
    # But this is linear.
    # To make it smooth and avoid flat regions, we can use (1-r_i)^2.
    # Minimizing sum((1-r_i)^2) pushes all circles to the boundary.
    
    boundary_energy = np.sum((1.0 - radii)**2)
    
    return pair_energy + boundary_energy

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    best_x = None
    best_score = -np.inf  # We track the actual radius score for comparison
    
    # Helper to compute score
    def _score(x_flat):
        x = x_flat.reshape(-1, 2)
        dists = np.linalg.norm(x[:, np.newaxis, :] - x[np.newaxis, :, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        r_pair = np.min(dists) / 2.0
        radii = np.linalg.norm(x, axis=1)
        r_bound = 1.0 - np.max(radii)
        return min(r_pair, r_bound)
    
    # 1. Hex Lattice Initialization
    x0 = _hex_lattice_init(N)
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    # Optimize
    # L-BFGS-B is good for bounded problems
    res = minimize(_potential, x0.flatten(), method='L-BFGS-B', bounds=bounds, 
                   options={'maxiter': 2000, 'ftol': 1e-12})
    
    if res.fun < np.inf:
        current_score = _score(res.x)
        if current_score > best_score:
            best_x = res.x
            best_score = current_score
    
    # 2. Local Search / Random Restarts
    # We try to escape local minima by perturbing the best solution
    attempts = 0
    while time.time() < deadline and attempts < 20:
        attempts += 1
        
        # Perturb the best solution found so far
        # Small noise to escape local minima
        noise_scale = 0.05
        x0 = best_x + np.random.normal(0, noise_scale, (2 * N,))
        # Ensure bounds
        x0 = np.clip(x0, -1.0, 1.0)
        
        res = minimize(_potential, x0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 1000, 'ftol': 1e-12})
        
        if res.fun < np.inf:
            current_score = _score(res.x)
            if current_score > best_score:
                best_x = res.x
                best_score = current_score
    
    # Final validity check
    if best_x is None:
        return _hex_lattice_init(N)
        
    x = best_x.reshape(-1, 2)
    # Ensure no NaNs
    if not np.all(np.isfinite(x)):
        return _hex_lattice_init(N)
        
    return x
# EVOLVE-BLOCK-END
