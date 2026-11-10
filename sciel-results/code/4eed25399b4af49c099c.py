import numpy as np
from scipy.optimize import minimize
import time

N = 300

def repulsive_potential_and_grad(x_flat):
    """
    Computes the value and gradient of the repulsive potential.
    x_flat is a 1D array of length 2*N.
    """
    x = x_flat.reshape(-1, 2)
    # Compute pairwise distances
    # dist[i, j] = |x_i - x_j|
    # Using broadcasting: x[:, np.newaxis, :] - x[np.newaxis, :, :]
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    # Avoid division by zero on the diagonal (i=j)
    np.fill_diagonal(dist, 1.0)
    
    # Potential: sum_{i<j} 1/dist_ij
    # We sum over all i != j and divide by 2 to avoid double counting, 
    # or just sum all i != j. The constant factor doesn't matter for optimization.
    # Let's sum all i != j for simplicity in gradient.
    potential = np.sum(1.0 / dist) - N # Subtract diagonal terms (1/1 = 1)
    
    # Gradient: 
    # grad_i = - sum_{j != i} (x_i - x_j) / dist_ij^3
    # diff is x_i - x_j.
    # We need dist^3 for each pair.
    dist_cubed = dist ** 3
    # Set diagonal to 1 to avoid inf, but they will be multiplied by 0 vector anyway?
    # diff[i, i] = 0. So 0 / 1 = 0. Correct.
    np.fill_diagonal(dist_cubed, 1.0)
    
    forces = diff / dist_cubed[:, :, np.newaxis]
    
    # Sum forces over j for each i.
    # forces[i, j] is the vector contribution from j to i?
    # diff[i, j] = x_i - x_j.
    # Gradient of 1/|x_i - x_j| w.r.t x_i is -(x_i - x_j)/|x_i - x_j|^3.
    # So we sum -forces[i, :].
    grad_pairwise = -np.sum(forces, axis=1)
    
    # Boundary term: sum_i 1 / (1 - |x_i|)
    r = np.linalg.norm(x, axis=1)
    # Add epsilon to avoid division by zero if r=1 (should not happen if we stay inside)
    # If r >= 1, the penalty is huge.
    boundary_potential = np.sum(1.0 / (1.0 - r + 1e-12))
    
    # Gradient of boundary term:
    # d/dr (1/(1-r)) = 1/(1-r)^2
    # grad_r = (1/(1-r)^2) * (x/r)
    # If r == 0, grad is 0.
    grad_boundary = np.zeros_like(x)
    mask = r > 1e-6
    grad_boundary[mask] = (1.0 / (1.0 - r[mask]) ** 2) * (x[mask] / r[mask, np.newaxis])
    
    total_potential = potential + boundary_potential
    total_grad = (grad_pairwise + grad_boundary).flatten()
    
    return total_potential, total_grad

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s safety margin
    
    best_score = -np.inf
    best_points = None
    
    # Initial configurations
    initial_configs = []
    
    # 1. Sunflower curve
    i = np.arange(N)
    theta = i * np.pi * (3.0 - np.sqrt(5.0))
    r = np.sqrt((i + 0.5) / N)
    x_sun = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
    initial_configs.append(x_sun.flatten())
    
    # 2. Hexagonal lattice
    # Try to generate a hex lattice that fits 300 points
    spacing = 0.1 # Initial guess
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            # Take the first N or the most central N
            idx = np.argsort(radius)[:N]
            initial_configs.append(points[idx].flatten())
            break
        spacing *= 0.95
        
    # 3. Random uniform in disk
    rng = np.random.default_rng(42)
    r_rand = np.sqrt(rng.random(N))
    theta_rand = rng.random(N) * 2 * np.pi
    x_rand = np.column_stack((r_rand * np.cos(theta_rand), r_rand * np.sin(theta_rand)))
    initial_configs.append(x_rand.flatten())

    for init_flat in initial_configs:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(repulsive_potential_and_grad, init_flat, method='L-BFGS-B',
                           jac=True, options={'maxiter': 500, 'maxls': 50})
            
            if res.success or res.status == 0:
                x_opt = res.x.reshape(-1, 2)
                # Check validity
                dists = np.linalg.norm(x_opt[:, np.newaxis, :] - x_opt[np.newaxis, :, :], axis=2)
                np.fill_diagonal(dists, np.inf)
                min_pair_dist = np.min(dists)
                
                radii = np.linalg.norm(x_opt, axis=1)
                min_boundary_dist = np.min(1.0 - radii)
                
                r_candidate = min(min_pair_dist / 2.0, min_boundary_dist)
                
                if r_candidate > best_score:
                    best_score = r_candidate
                    best_points = x_opt
                    
        except Exception:
            continue

    if best_points is None:
        # Fallback to seed
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
            if len(points) >= N:
                return points[np.argsort(radius, kind="stable")[:N]]
            spacing *= 0.98
            
    return best_points
