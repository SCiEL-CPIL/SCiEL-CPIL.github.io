"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_radius(points):
    """Compute the packing radius for a set of points."""
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_pair_dist = np.sqrt(np.min(dist_sq))
    
    # Boundary distances
    r_i = np.linalg.norm(points, axis=1)
    min_bound_dist = np.min(1.0 - r_i)
    
    return min(min_pair_dist / 2.0, min_bound_dist)


def _generate_hex_lattice(n_points, target_radius=None):
    """
    Generate a hexagonal lattice and select n_points closest to the center
    that fit within a radius (1 - spacing/2) to ensure validity.
    """
    if target_radius is None:
        # Estimate spacing based on area
        # Area per point ~ pi / N
        # Hex cell area ~ sqrt(3)/2 * s^2
        # s ~ sqrt(2 * pi / (N * sqrt(3)))
        s = np.sqrt(2.0 * np.pi / (N * np.sqrt(3.0)))
    else:
        s = 2.0 * target_radius * 1.01 # Slightly larger spacing to start safe

    # Generate points in a large enough grid
    k = int(1.0 / s) + 3
    points = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = s * (i + 0.5 * (j % 2))
            y = s * j * np.sqrt(3.0) / 2.0
            points.append([x, y])
    
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    
    # Ensure points are strictly inside the unit circle with margin for their radius
    # The circle of radius r centered at c is inside unit circle if |c| + r <= 1
    # Here r = s/2 (if we assume touching neighbors). 
    # So we need |c| <= 1 - s/2.
    margin = s / 2.0
    mask = dists <= 1.0 - margin
    
    if not np.any(mask):
        # If no points fit, return center
        return np.zeros((n_points, 2))

    points = points[mask]
    dists = dists[mask]
    
    if len(points) < n_points:
        # If we don't have enough points, we need a tighter lattice
        # This shouldn't happen with the initial estimate, but just in case
        # For now, just take what we have and duplicate? No, better to return fewer?
        # The function signature expects shape (N, 2).
        # If we can't fit N points with this spacing, the spacing is too large.
        # But the caller controls the initial spacing.
        # Let's just return the closest N points we found, even if they overlap?
        # No, that would be invalid.
        # However, the initial estimate should be safe.
        pass

    # Sort by distance to center and take N
    sorted_idx = np.argsort(dists, kind="stable")
    selected_idx = sorted_idx[:n_points]
    
    return points[selected_idx]


def construct():
    start_time = time.time()
    # Deadline: 140s out of 150s limit, leaving 10s buffer
    deadline = start_time + 140.0
    
    # Initial guess: Hexagonal lattice
    # Estimate optimal radius roughly
    # Density of hex packing is pi / (2 * sqrt(3)) ~ 0.9069
    # N * pi * r^2 ~ 0.9069 * pi * 1^2
    # r ~ sqrt(0.9069 / N)
    r_est = np.sqrt(0.9069 / N)
    
    # Start with a slightly smaller radius to ensure valid initial configuration
    # and allow room for optimization to push outwards.
    init_radius = r_est * 0.95
    
    # Generate initial points based on this radius
    # Spacing s = 2 * r
    points = _generate_hex_lattice(N, target_radius=init_radius)
    
    # If we couldn't generate enough points or they are invalid, fallback to simple grid
    if points.shape[0] < N:
        # Fallback: simple grid
        s = 2.0 / (np.sqrt(N) + 1.0)
        k = int(1.0 / s) + 2
        pts = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = s * (i + 0.5 * (j % 2))
                y = s * j * np.sqrt(3.0) / 2.0
                pts.append([x, y])
        pts = np.array(pts)
        dists = np.linalg.norm(pts, axis=1)
        pts = pts[dists <= 1.0 - s/2]
        dists = dists[dists <= 1.0 - s/2]
        sorted_idx = np.argsort(dists, kind="stable")
        points = pts[sorted_idx[:N]]

    best_points = points.copy()
    best_radius = _compute_radius(best_points)
    
    # Optimization using L-BFGS-B
    # Objective: Minimize -radius (maximize radius)
    # Since radius is min of many terms, it's non-smooth.
    # Alternative: Minimize a potential energy that encourages separation.
    # E = sum_{i<j} 1/d_ij^2 + sum_i 1/(1-|c_i|)^2
    # This is smooth and encourages points to spread out and stay away from boundary.
    
    def energy_and_grad(flat_x):
        x = flat_x.reshape((N, 2))
        
        # Pairwise repulsion
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        
        # Avoid division by zero
        dist_sq_safe = np.where(dist_sq < 1e-12, 1e-12, dist_sq)
        
        inv_dist_sq = 1.0 / dist_sq_safe
        np.fill_diagonal(inv_dist_sq, 0.0)
        
        E_pair = np.sum(inv_dist_sq)
        
        # Gradient of E_pair
        # d/dx_i (1/|x_i - x_j|^2) = -2 (x_i - x_j) / |x_i - x_j|^4
        grad_factor = -2.0 / (dist_sq_safe ** 2)
        np.fill_diagonal(grad_factor, 0.0)
        grad_pair = np.sum(grad_factor[:, :, np.newaxis] * diff, axis=1)
        
        # Boundary repulsion
        r_i = np.linalg.norm(x, axis=1)
        r_safe = np.where(r_i < 1e-9, 1e-9, r_i)
        
        d_b = 1.0 - r_i
        # If a point is outside, d_b is negative. We want to penalize this strongly.
        # Use max(d_b, epsilon) to keep gradient finite but large.
        d_b_safe = np.where(d_b < 1e-6, 1e-6, d_b)
        
        E_bound = np.sum(1.0 / d_b_safe**2)
        
        # Gradient of E_bound
        # d/dx_i (1/(1-r_i)^2) = 2 (x_i/r_i) / (1-r_i)^3
        grad_bound = 2.0 * (x / r_safe[:, np.newaxis]) / d_b_safe[:, np.newaxis]**3
        
        total_energy = E_pair + E_bound
        total_grad = grad_pair + grad_bound
        
        return total_energy, total_grad

    x0 = points.flatten()
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    try:
        # Use L-BFGS-B
        # maxiter limited by time
        res = minimize(
            energy_and_grad, 
            x0, 
            jac=True, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 1000, 'disp': False, 'ftol': 1e-8, 'gtol': 1e-6, 'maxfun': 5000}
        )
        
        optimized_points = res.x.reshape((N, 2))
        
        # Check if optimized points are valid
        # 1. All points inside unit circle (with some margin to avoid floating point errors)
        # 2. No coincident points
        
        # Calculate radius
        opt_radius = _compute_radius(optimized_points)
        
        # Check validity explicitly
        # If any point is outside (1 - |c| <= 0) or min dist <= 0, it's invalid
        # _compute_radius returns min(min_pair/2, min_bound).
        # If min_bound < 0, radius is negative.
        # If min_pair < 0, radius is negative.
        
        if opt_radius > best_radius:
            # Additional check: ensure no points are actually outside the circle
            # due to numerical issues in the optimization
            dists = np.linalg.norm(optimized_points, axis=1)
            if np.all(dists < 1.0):
                # Check for coincident points
                diff = optimized_points[:, np.newaxis, :] - optimized_points[np.newaxis, :, :]
                dist_sq = np.sum(diff**2, axis=2)
                np.fill_diagonal(dist_sq, np.inf)
                if np.min(dist_sq) > 1e-12:
                    best_points = optimized_points
                    best_radius = opt_radius
    except Exception:
        pass
    
    # If time permits, try a second round of optimization with a different initial guess
    # or refine the current best with a different method
    if time.time() < deadline - 10.0:
        # Try a simple local search / perturbation on the best points found
        # Or just run L-BFGS again with tighter tolerances
        try:
            res2 = minimize(
                energy_and_grad, 
                best_points.flatten(), 
                jac=True, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 2000, 'disp': False, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 10000}
            )
            
            refined_points = res2.x.reshape((N, 2))
            refined_radius = _compute_radius(refined_points)
            
            dists = np.linalg.norm(refined_points, axis=1)
            if np.all(dists < 1.0):
                diff = refined_points[:, np.newaxis, :] - refined_points[np.newaxis, :, :]
                dist_sq = np.sum(diff**2, axis=2)
                np.fill_diagonal(dist_sq, np.inf)
                if np.min(dist_sq) > 1e-12 and refined_radius > best_radius:
                    best_points = refined_points
                    best_radius = refined_radius
        except Exception:
            pass
            
    return best_points
# EVOLVE-BLOCK-END
