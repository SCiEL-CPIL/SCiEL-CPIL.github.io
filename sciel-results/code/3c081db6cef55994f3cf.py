"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 300

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    def calculate_radius(points):
        """Calculate the maximum radius for a given set of points."""
        if len(points) < 2:
            return 1.0
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

    # 2. Define energy function for L-BFGS-B
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
        # d/dx_i (1/d_b^2) = 2 * (x_i / r_i) / d_b^3
        grad_bound = 2.0 * (x / r_safe[:, np.newaxis]) / d_b_safe[:, np.newaxis]**3
        
        # Handle case where d_b was clamped
        # If d_b < 1e-9, the gradient direction is ambiguous, but the force is huge.
        # The clamping in E_bound creates a discontinuity.
        # For optimization stability, this is usually fine.
        
        total_energy = E_pair + E_bound
        total_grad = grad_pair + grad_bound
        
        return total_energy, total_grad

    # 3. Optimization using L-BFGS-B
    x0 = best_points.flatten()
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    try:
        res = minimize(
            energy_and_grad, 
            x0, 
            jac=True, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 10000, 'disp': False, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
        )
        optimized_points = res.x.reshape((N, 2))
        
        # Check validity
        if np.all(np.isfinite(optimized_points)):
            # Ensure points are inside the circle
            r_i = np.linalg.norm(optimized_points, axis=1)
            if np.all(r_i <= 1.0):
                score = calculate_radius(optimized_points)
                if score > best_radius:
                    best_points = optimized_points
                    best_radius = score
    except Exception:
        pass

    # 4. Local refinement: push apart closest pairs and pull in outermost points
    if time.time() < deadline - 5.0:
        current_points = best_points.copy()
        current_radius = best_radius
        
        # Simple local search: repeatedly find the bottleneck and adjust
        for _ in range(300):
            if time.time() > deadline - 5.0:
                break
                
            # Find the current bottleneck
            dists = cdist(current_points, current_points)
            np.fill_diagonal(dists, np.inf)
            min_pair_dist = np.min(dists)
            
            r_i = np.linalg.norm(current_points, axis=1)
            min_bound_dist = np.min(1.0 - r_i)
            
            # Determine which constraint is active
            if min_pair_dist / 2.0 < min_bound_dist:
                # Pair distance is limiting
                # Find the pair
                idx = np.unravel_index(np.argmin(dists), dists.shape)
                i, j = idx
                
                # Push them apart
                mid = (current_points[i] + current_points[j]) / 2.0
                direction = current_points[i] - current_points[j]
                norm = np.linalg.norm(direction)
                if norm > 1e-9:
                    direction /= norm
                    step = 0.001
                    current_points[i] += direction * step
                    current_points[j] -= direction * step
            else:
                # Boundary distance is limiting
                # Find the outermost point
                idx = np.argmin(1.0 - r_i)
                point = current_points[idx]
                norm = np.linalg.norm(point)
                if norm > 1e-9:
                    direction = -point / norm
                    step = 0.001
                    current_points[idx] += direction * step
            
            # Check if the move improved the radius
            new_radius = calculate_radius(current_points)
            if new_radius > current_radius:
                current_radius = new_radius
            else:
                # Revert
                if min_pair_dist / 2.0 < min_bound_dist:
                    current_points[i] -= direction * step
                    current_points[j] += direction * step
                else:
                    current_points[idx] -= direction * step
        
        if current_radius > best_radius:
            best_points = current_points
            best_radius = current_radius

    return best_points
# EVOLVE-BLOCK-END
