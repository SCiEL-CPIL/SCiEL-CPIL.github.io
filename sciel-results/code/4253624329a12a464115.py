"""Rewritten program for circle packing in a circle."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 300

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    def calculate_radius(points):
        """Calculate the maximum common radius for a given set of points."""
        if len(points) < 2:
            return 1.0
        # Pairwise min distance using cdist for efficiency
        dists = cdist(points, points)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Boundary min distance
        r_i = np.linalg.norm(points, axis=1)
        min_bound_dist = np.min(1.0 - r_i)
        
        return min(min_pair_dist / 2.0, min_bound_dist)

    def gen_hex_lattice(spacing):
        """Generate hexagonal lattice points inside unit circle."""
        k = int(1.0 / spacing) + 3
        pts = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                r = np.hypot(x, y)
                if r <= 1.0 - spacing * 0.5:
                    pts.append((x, y))
        return np.array(pts)

    def gen_random_points(n, seed=42):
        """Generate random points inside unit circle."""
        rng = np.random.RandomState(seed)
        angles = rng.uniform(0, 2 * np.pi, size=n)
        radii = rng.uniform(0, 0.95, size=n)
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        return np.column_stack([x, y])

    # Generate initial hexagonal lattice
    spacing = 0.078
    points = gen_hex_lattice(spacing)
    
    # If we don't have enough points, reduce spacing
    while len(points) < N and spacing > 0.01:
        spacing *= 0.95
        points = gen_hex_lattice(spacing)
        
    # If we have more than N, take the ones closest to center
    if len(points) >= N:
        dists = np.linalg.norm(points, axis=1)
        sorted_idx = np.argsort(dists)
        init_points = points[sorted_idx[:N]]
    else:
        # If still not enough, pad with random points
        init_points = points
        while len(init_points) < N:
            init_points = np.vstack([init_points, gen_random_points(1)])
        init_points = init_points[:N]

    # Define energy function to minimize
    def energy_and_grad(flat_x):
        x = flat_x.reshape((N, 2))
        
        # Calculate pairwise distances
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        
        # Avoid division by zero for i==j
        inv_dist_sq = 1.0 / (dist_sq + 1e-9)
        np.fill_diagonal(inv_dist_sq, 0.0)
        
        E_pair = np.sum(inv_dist_sq)
        
        # Gradient of E_pair
        grad_factor = -2.0 / (dist_sq + 1e-9)**2
        np.fill_diagonal(grad_factor, 0.0)
        grad_pair = np.sum(grad_factor[:, :, np.newaxis] * diff, axis=1)
        
        # Boundary repulsion
        r_i = np.linalg.norm(x, axis=1)
        r_safe = np.where(r_i < 1e-9, 1e-9, r_i)
        d_b = 1.0 - r_i
        d_b_safe = np.where(d_b < 1e-9, 1e-9, d_b)
        
        E_bound = np.sum(1.0 / d_b_safe**2)
        grad_bound = 2.0 * (x / r_safe[:, np.newaxis]) / d_b_safe[:, np.newaxis]**3
        
        total_energy = E_pair + E_bound
        total_grad = grad_pair + grad_bound
        
        return total_energy, total_grad

    # Initial guess
    x0 = init_points.flatten()
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    best_points = init_points.copy()
    best_score = calculate_radius(init_points)

    # First optimization pass
    try:
        res = minimize(
            energy_and_grad, 
            x0, 
            jac=True, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 3000, 'disp': False, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
        )
        optimized_points = res.x.reshape((N, 2))
        
        # Check validity
        if np.all(np.isfinite(optimized_points)):
            r_i = np.linalg.norm(optimized_points, axis=1)
            if np.all(r_i <= 1.0):
                score = calculate_radius(optimized_points)
                if score > best_score:
                    best_points = optimized_points.copy()
                    best_score = score
    except Exception:
        pass

    # Check time
    if time.time() > deadline - 30:
        return best_points

    # Second optimization pass with perturbation
    perturbation_scale = 0.005
    perturbed = best_points + np.random.randn(N, 2) * perturbation_scale
    # Keep perturbed points inside unit circle
    r_i = np.linalg.norm(perturbed, axis=1)
    mask = r_i > 0.99
    perturbed[mask] = perturbed[mask] * (0.99 / r_i[mask])[:, np.newaxis]
    
    x0_pert = perturbed.flatten()
    try:
        res2 = minimize(
            energy_and_grad, 
            x0_pert, 
            jac=True, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 2000, 'disp': False, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
        )
        optimized_points2 = res2.x.reshape((N, 2))
        
        if np.all(np.isfinite(optimized_points2)):
            r_i2 = np.linalg.norm(optimized_points2, axis=1)
            if np.all(r_i2 <= 1.0):
                score2 = calculate_radius(optimized_points2)
                if score2 > best_score:
                    best_points = optimized_points2.copy()
                    best_score = score2
    except Exception:
        pass

    # Check time
    if time.time() > deadline - 10:
        return best_points

    # Third pass: try different initializations
    for seed in [1, 2, 3]:
        if time.time() > deadline - 15:
            break
        
        random_init = gen_random_points(N, seed=seed)
        x0_rand = random_init.flatten()
        
        try:
            res3 = minimize(
                energy_and_grad, 
                x0_rand, 
                jac=True, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 1500, 'disp': False, 'ftol': 1e-11, 'gtol': 1e-9, 'maxls': 40}
            )
            optimized_points3 = res3.x.reshape((N, 2))
            
            if np.all(np.isfinite(optimized_points3)):
                r_i3 = np.linalg.norm(optimized_points3, axis=1)
                if np.all(r_i3 <= 1.0):
                    score3 = calculate_radius(optimized_points3)
                    if score3 > best_score:
                        best_points = optimized_points3.copy()
                        best_score = score3
        except Exception:
            pass

    return best_points
# EVOLVE-BLOCK-END
