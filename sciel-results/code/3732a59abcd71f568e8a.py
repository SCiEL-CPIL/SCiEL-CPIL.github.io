"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, LinearConstraint
import time

def construct():
    """
    Construct 200 circle centres in [-0.5, 0.5]^2 maximizing the minimum radius.
    
    Approach: 
    1. Start with a hexagonal lattice packing (warm start).
    2. Use a projected gradient method to maximize the minimum pairwise distance
       while keeping points inside the square.
    3. The objective is to maximize r such that:
       - For all i != j: ||c_i - c_j|| >= 2r
       - For all i, k: 0.5 - |c_ik| >= r
    
    We solve this via iterative improvement:
    - Start with hexagonal lattice
    - Repeatedly identify the closest pair and push them apart
    - Project back into the square
    - Use scipy.optimize.minimize with a barrier-like objective for refinement
    """
    # Step 1: Generate initial hexagonal lattice
    def hex_lattice(n):
        """Generate n points in a hexagonal pattern scaled to fit in unit square."""
        points = []
        # Hexagonal lattice spacing
        # We want to pack n points. For hexagonal, density is pi/(2*sqrt(3)) * (1/d^2)
        # Let's generate a grid and select the best n points
        
        # Generate a larger hexagonal grid
        max_i = 15
        max_j = 15
        hex_points = []
        
        for i in range(-max_i, max_i + 1):
            for j in range(-max_j, max_j + 1):
                x = i + 0.5 * (j % 2)
                y = j * np.sqrt(3) / 2
                hex_points.append((x, y))
        
        hex_points = np.array(hex_points)
        
        # Scale to fit in [-0.5, 0.5]^2
        # Find the bounding box
        x_min, y_min = hex_points.min(axis=0)
        x_max, y_max = hex_points.max(axis=0)
        
        # Scale so that the points fit in the unit square
        scale = min(1.0 / (x_max - x_min), 1.0 / (y_max - y_min))
        hex_points = (hex_points - (hex_points.min(axis=0) + hex_points.max(axis=0)) / 2) * scale
        
        # Center in the square
        hex_points = hex_points + np.array([0, 0])
        
        # Select the n points closest to the center (for better packing)
        dists = np.sum(hex_points**2, axis=1)
        idx = np.argsort(dists)[:n]
        points = hex_points[idx]
        
        return points
    
    # Step 2: Initial hexagonal lattice
    points = hex_lattice(N)
    
    # Ensure points are within bounds
    points = np.clip(points, -0.5, 0.5)
    
    # Step 3: Iterative improvement using projected gradient
    def compute_radius(pts):
        """Compute the maximum valid radius for given points."""
        if len(pts) < 2:
            return 0.5
        # Pairwise distances
        from scipy.spatial.distance import cdist
        D = cdist(pts, pts)
        np.fill_diagonal(D, np.inf)
        min_pair = D.min()
        # Distance to boundary
        min_bound = np.min(np.minimum(0.5 - np.abs(pts), 0.5))
        return min(min_pair / 2, min_bound)
    
    # Gradient ascent on the minimum distance
    # We want to maximize the minimum of:
    # - pairwise distances / 2
    # - distance to boundary
    
    # Use a soft-min approach for differentiability
    def soft_min_radius(pts, temperature=0.01):
        """Soft minimum of all constraints for differentiability."""
        from scipy.spatial.distance import cdist
        D = cdist(pts, pts)
        np.fill_diagonal(D, np.inf)
        pair_dists = D[D < np.inf] / 2
        
        # Distance to boundary
        bound_dists = np.minimum(0.5 - np.abs(pts), 0.5)
        
        all_dists = np.concatenate([pair_dists, bound_dists])
        
        # Soft min: -T * log(sum(exp(-x/T)))
        # For numerical stability
        min_val = all_dists.min()
        exp_vals = np.exp(-(all_dists - min_val) / temperature)
        soft_min = min_val - temperature * np.log(np.sum(exp_vals))
        
        return soft_min
    
    # Use scipy minimize to maximize soft_min_radius
    # Convert to minimization: minimize -soft_min_radius
    
    def objective_neg_radius(x_flat, temperature=0.01):
        pts = x_flat.reshape(-1, 2)
        # Add penalty for being outside bounds
        outside_penalty = np.sum(np.maximum(0, np.abs(pts) - 0.5)**2) * 100
        return -soft_min_radius(pts, temperature) + outside_penalty
    
    def objective_grad(x_flat, temperature=0.01):
        pts = x_flat.reshape(-1, 2)
        # Numerical gradient
        eps = 1e-6
        grad = np.zeros_like(x_flat)
        for i in range(len(x_flat)):
            x_plus = x_flat.copy()
            x_minus = x_flat.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            grad[i] = (objective_neg_radius(x_plus, temperature) - objective_neg_radius(x_minus, temperature)) / (2 * eps)
        return grad
    
    # Run optimization
    x0 = points.flatten()
    
    # Use L-BFGS-B with bounds
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    # Try a few optimization rounds with decreasing temperature
    for temp in [0.05, 0.02, 0.01, 0.005]:
        result = minimize(
            objective_neg_radius,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
        )
        x0 = result.x
        points = x0.reshape(-1, 2)
        # Project back to bounds
        points = np.clip(points, -0.5, 0.5)
    
    # Final refinement: local pairwise repulsion
    # Iteratively push apart the closest pairs
    for iteration in range(1000):
        from scipy.spatial.distance import cdist
        D = cdist(points, points)
        np.fill_diagonal(D, np.inf)
        
        # Find closest pair
        min_idx = np.unravel_index(D.argmin(), D.shape)
        i, j = min_idx
        min_dist = D[i, j]
        
        # Check if we can improve by moving i and j apart
        if min_dist < 0.07:  # Only if they're close
            # Direction from i to j
            diff = points[j] - points[i]
            norm = np.linalg.norm(diff)
            if norm > 1e-10:
                direction = diff / norm
            else:
                direction = np.array([1.0, 0.0])
            
            # Push apart
            step = 0.001
            points[i] -= direction * step
            points[j] += direction * step
            
            # Project back to bounds
            points = np.clip(points, -0.5, 0.5)
    
    # Final check and cleanup
    points = np.clip(points, -0.5, 0.5)
    
    # Verify no coincident points
    from scipy.spatial.distance import cdist
    D = cdist(points, points)
    np.fill_diagonal(D, np.inf)
    if D.min() < 1e-10:
        # Perturb coincident points
        for i in range(N):
            for j in range(i+1, N):
                if np.linalg.norm(points[i] - points[j]) < 1e-10:
                    points[j] += np.array([1e-8, 1e-8])
                    points[j] = np.clip(points[j], -0.5, 0.5)
    
    return points
# EVOLVE-BLOCK-END
