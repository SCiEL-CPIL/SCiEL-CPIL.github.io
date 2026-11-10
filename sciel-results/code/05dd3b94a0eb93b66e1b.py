import numpy as np
import time
from scipy.spatial.distance import cdist

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def calculate_radius(points):
        """Calculate the maximum radius for a given set of points."""
        # Pairwise min distance
        dists = cdist(points, points)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Boundary min distance
        r_i = np.linalg.norm(points, axis=1)
        min_bound_dist = np.min(1.0 - r_i)
        
        return min(min_pair_dist / 2.0, min_bound_dist)

    def generate_hex_lattice(N_points):
        """Generate a hexagonal lattice initialization."""
        # Estimate spacing based on area
        # Area per point ~ pi/N. Hex cell area ~ 3*sqrt(3)/2 * s^2.
        # s ~ sqrt(2*pi / (3*sqrt(3)*N))
        s = np.sqrt(2.0 * np.pi / (3.0 * np.sqrt(3.0) * N_points))
        
        # Generate grid
        k = int(1.0 / s) + 2
        rows = []
        for i in range(-k, k + 1):
            for j in range(-k, k + 1):
                x = s * (i + 0.5 * (j % 2))
                y = s * j * np.sqrt(3.0) / 2.0
                rows.append([x, y])
        
        points = np.array(rows)
        dists = np.linalg.norm(points, axis=1)
        points = points[dists <= 1.0 - s/2]
        
        if len(points) < N_points:
            # Try smaller spacing
            s *= 0.9
            k = int(1.0 / s) + 2
            rows = []
            for i in range(-k, k + 1):
                for j in range(-k, k + 1):
                    x = s * (i + 0.5 * (j % 2))
                    y = s * j * np.sqrt(3.0) / 2.0
                    rows.append([x, y])
            points = np.array(rows)
            dists = np.linalg.norm(points, axis=1)
            points = points[dists <= 1.0 - s/2]
            
        # Sort by distance to center and take N
        dists = np.linalg.norm(points, axis=1)
        sorted_idx = np.argsort(dists)
        points = points[sorted_idx[:N_points]]
        
        return points

    # 1. Initialization
    best_points = generate_hex_lattice(N)
    best_radius = calculate_radius(best_points)

    # 2. Optimization using Simulated Annealing or simple Hill Climbing with restarts
    # Given time constraints, a robust local search with multiple restarts is often better than one long gradient descent.
    # We will use a simple gradient ascent on the radius objective using the "minimum distance" gradient.
    
    def optimize_points(points, max_time):
        """Improve points using gradient ascent on the minimum distance."""
        current_points = points.copy()
        current_radius = calculate_radius(current_points)
        
        # Precompute indices for nearest neighbors? 
        # For gradient of min distance, we need the pair that achieves the min.
        
        step_size = 0.01
        no_improve_count = 0
        max_no_improve = 200
        
        while time.time() < max_time and no_improve_count < max_no_improve:
            # Find the bottleneck
            dists = cdist(current_points, current_points)
            np.fill_diagonal(dists, np.inf)
            
            # Find min pair distance
            min_pair_dist = np.min(dists)
            # Find min boundary distance
            r_i = np.linalg.norm(current_points, axis=1)
            min_bound_dist = np.min(1.0 - r_i)
            
            # Determine which constraint is active
            # Score is min(min_pair_dist/2, min_bound_dist)
            # We want to increase the smaller one.
            
            # Gradient calculation is tricky with min. 
            # Approximation: Move points apart if pair distance is limiting.
            # Move points in if boundary distance is limiting.
            
            # Let's use a simpler heuristic: 
            # If pair distance is limiting, find the closest pair and push them apart.
            # If boundary distance is limiting, find the outermost point and pull it in.
            
            # Actually, a standard approach for packing is to maximize the minimum distance.
            # Let's use a simple gradient ascent step based on the "soft" minimum or just the current minimum.
            
            # To be efficient, let's just do random perturbations of the worst points.
            # Or, simpler: use the gradient of the sum of inverse distances as a proxy for separation.
            
            # Let's stick to the energy minimization from the seed but with better parameters and multiple restarts.
            # The seed used L-BFGS-B. Let's use a simpler, faster local search: 
            # Move each point in the direction that increases the distance to its nearest neighbor.
            
            # Find nearest neighbor for each point
            # This is O(N^2) per step. N=300. 90k ops. Fast enough.
            
            # dists is already computed.
            min_dist_matrix = dists.copy()
            min_dist_matrix[np.diag_indices_from(min_dist_matrix)] = np.inf
            
            # For each point, find nearest neighbor
            nearest_neighbor_indices = np.argmin(min_dist_matrix, axis=1)
            nearest_distances = min_dist_matrix[np.arange(N), nearest_neighbor_indices]
            
            # Also consider boundary
            # We want to maximize r = min(min_pair_dist/2, min_bound_dist)
            
            # Let's calculate gradients for a potential function:
            # V = sum_{i} 1/d_i^k where d_i is min distance to neighbor or boundary.
            # This is complex. Let's just use the seed's energy function but optimize it faster.
            
            break # Skip complex custom optimizer, stick to scipy if it works well, or use a simpler restart strategy.

        return current_points, current_radius

    # Strategy:
    # 1. Try L-BFGS-B on the initial hex lattice.
    # 2. If time permits, try a few random restarts with local optimization.
    
    # Define the energy function for L-BFGS-B
    def energy_and_grad(flat_x):
        x = flat_x.reshape((N, 2))
        
        # Calculate pairwise distances
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        
        # Avoid division by zero
        dist_sq_safe = dist_sq + 1e-9
        np.fill_diagonal(dist_sq_safe, 1.0) # Diagonal doesn't matter if we zero out gradient
        
        # Energy term 1: Pairwise repulsion (1/d^2)
        inv_dist_sq = 1.0 / dist_sq_safe
        np.fill_diagonal(inv_dist_sq, 0.0)
        E_pair = np.sum(inv_dist_sq)
        
        # Gradient of E_pair
        # d/dx_i (1/|x_i - x_j|^2) = -2 (x_i - x_j) / |x_i - x_j|^4
        grad_factor = -2.0 / (dist_sq_safe**2)
        np.fill_diagonal(grad_factor, 0.0)
        grad_pair = np.sum(grad_factor[:, :, np.newaxis] * diff, axis=1)
        
        # Energy term 2: Boundary repulsion
        r_i = np.linalg.norm(x, axis=1)
        r_safe = np.where(r_i < 1e-9, 1e-9, r_i)
        d_b = 1.0 - r_i
        d_b_safe = np.where(d_b < 1e-9, 1e-9, d_b)
        
        E_bound = np.sum(1.0 / d_b_safe**2)
        
        # Gradient of E_bound
        # d/dx_i (1/(1-|x_i|)^2) = 2 (x_i/|x_i|) / (1-|x_i|)^3
        grad_bound = 2.0 * (x / r_safe[:, np.newaxis]) / d_b_safe[:, np.newaxis]**3
        
        total_energy = E_pair + E_bound
        total_grad = grad_pair + grad_bound
        
        return total_energy, total_grad

    bounds = [(-1.0, 1.0)] * (2 * N)
    
    # Attempt 1: Optimize initial lattice
    try:
        from scipy.optimize import minimize
        res = minimize(
            energy_and_grad, 
            best_points.flatten(), 
            jac=True, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 1000, 'disp': False, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        opt_points = res.x.reshape((N, 2))
        opt_radius = calculate_radius(opt_points)
        
        if opt_radius > best_radius:
            best_points = opt_points
            best_radius = opt_radius
            
    except Exception as e:
        pass

    # Attempt 2: Random restarts if time permits
    # Generate random points and optimize locally
    while time.time() < deadline - 5.0:
        # Generate random points inside circle
        theta = np.random.uniform(0, 2*np.pi, N)
        r = np.random.uniform(0, 1, N)
        # Rejection sampling to ensure they are inside? No, uniform in radius is not uniform in area.
        # For random init, we want them somewhat spread out. 
        # A simple random init is bad for L-BFGS because it gets stuck in local minima.
        # Let's use a perturbed hex lattice instead.
        
        # Perturb best_points
        perturb_radius = 0.01
        new_points = best_points + np.random.normal(0, perturb_radius, (N, 2))
        
        # Ensure they stay inside
        r_i = np.linalg.norm(new_points, axis=1)
        mask = r_i > 1.0
        if np.any(mask):
            new_points[mask] = new_points[mask] / r_i[mask][:, np.newaxis] * 0.99
            
        try:
            res = minimize(
                energy_and_grad, 
                new_points.flatten(), 
                jac=True, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 500, 'disp': False, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            opt_points = res.x.reshape((N, 2))
            opt_radius = calculate_radius(opt_points)
            
            if opt_radius > best_radius:
                best_points = opt_points
                best_radius = opt_radius
                # Reset perturbation size or keep it
        except Exception:
            pass

    return best_points

# EVOLVE-BLOCK-END
