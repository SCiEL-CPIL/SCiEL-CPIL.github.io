"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt

def construct():
    """
    Constructs the centers of 300 equal circles inside a unit circle.
    Uses a collocation-inspired approach: treat the packing as a trajectory
    optimization where we evolve from a sparse configuration to a dense one,
    using sparse constraint handling (only nearest neighbors matter).
    """
    start_time = time.time()
    time_limit = 115.0  # Leave 5s buffer

    # Generate initial points using a well-distributed hexagonal-like pattern
    def generate_initial_points(n):
        """Generate initial points using a spiral/hexagonal pattern."""
        points = []
        # Use a phyllotaxis-like spiral for good distribution
        for i in range(n):
            r = 0.95 * np.sqrt(i / n)
            theta = i * 2.399963  # golden angle in radians
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            points.append([x, y])
        
        points = np.array(points)
        # Ensure all points are within unit circle
        norms = np.linalg.norm(points, axis=1)
        mask = norms > 0.99
        if np.any(mask):
            points[mask] = points[mask] / norms[mask, np.newaxis] * 0.99
        return points

    points = generate_initial_points(N)
    
    best_points = points.copy()
    best_r = 0.0

    # Helper to calculate the current valid radius for a set of points
    def calc_radius(pts):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        norms = np.sqrt(np.sum(pts ** 2, axis=1))
        min_bound_dist = np.min(1.0 - norms)
        
        return min(min_pair_dist / 2.0, min_bound_dist)

    # Helper to find nearest neighbors (sparse constraint handling)
    def find_nearest_neighbors(pts, k=10):
        """Find k nearest neighbors for each point using efficient method."""
        n = len(pts)
        # Compute pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # For each point, find k nearest neighbors
        nn_indices = np.argsort(dists, axis=1)[:, :k]
        return nn_indices

    # Helper to calculate energy with sparse constraints (only consider nearest neighbors)
    def energy_sparse(pts, target_r, nn_indices):
        """Calculate energy considering only nearest neighbor pairs."""
        n = len(pts)
        e = 0.0
        
        # Pairwise overlaps - only for nearest neighbors
        for i in range(n):
            for j in nn_indices[i]:
                if j > i:  # Avoid double counting
                    d = np.linalg.norm(pts[i] - pts[j])
                    overlap = max(0, 2.0 * target_r - d)
                    e += overlap ** 2
        
        # Boundary violations
        norms = np.sqrt(np.sum(pts ** 2, axis=1))
        bound_viol = np.maximum(0, norms - (1.0 - target_r))
        e += np.sum(bound_viol ** 2)
        
        return e

    # Helper to calculate gradient with sparse constraints
    def grad_sparse(x, target_r, nn_indices):
        """Calculate gradient considering only nearest neighbor pairs."""
        p = x.reshape(-1, 2)
        n = len(p)
        g = np.zeros_like(p)
        
        # Pairwise gradient - only for nearest neighbors
        for i in range(n):
            for j in nn_indices[i]:
                if j > i:  # Avoid double counting
                    diff = p[i] - p[j]
                    d = np.linalg.norm(diff)
                    d_safe = max(d, 1e-10)
                    overlap = max(0, 2.0 * target_r - d)
                    if overlap > 0:
                        unit_diff = diff / d_safe
                        grad_mag = -2.0 * overlap
                        g[i] += grad_mag * unit_diff
                        g[j] -= grad_mag * unit_diff
        
        # Boundary gradient
        norms = np.sqrt(np.sum(p ** 2, axis=1))
        bound_viol = np.maximum(0, norms - (1.0 - target_r))
        unit_inward = p / np.maximum(norms[:, np.newaxis], 1e-10)
        g += 2.0 * bound_viol[:, np.newaxis] * unit_inward
        
        return g.flatten()

    # Main collocation-inspired optimization: evolve from sparse to dense
    # We'll use a multi-stage approach where we gradually increase the target radius
    # and update nearest neighbors as the configuration evolves
    
    current_points = points.copy()
    target_r = 0.04
    step_size = 0.002
    
    # Initial nearest neighbors
    nn_indices = find_nearest_neighbors(current_points, k=12)
    
    consecutive_failures = 0
    max_consecutive_failures = 3
    stage = 0

    while time.time() - start_time < time_limit:
        # Update nearest neighbors periodically
        if stage % 10 == 0:
            nn_indices = find_nearest_neighbors(current_points, k=12)
        
        # Run optimization for current target_r
        def obj(x):
            p = x.reshape(-1, 2)
            return energy_sparse(p, target_r, nn_indices)
            
        def grad(x):
            return grad_sparse(x, target_r, nn_indices)
        
        x0 = current_points.flatten()
        res = opt.minimize(obj, x0, method='L-BFGS-B', jac=grad, 
                          options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
        
        optimized_points = res.x.reshape(-1, 2)
        energy_val = res.fun
        
        # Check if we achieved a valid packing
        if energy_val < 1e-8:
            actual_r = calc_radius(optimized_points)
            if actual_r > best_r:
                best_r = actual_r
                best_points = optimized_points.copy()
                current_points = optimized_points.copy()
                target_r += step_size
                consecutive_failures = 0
            else:
                target_r += step_size * 0.3
                consecutive_failures = 0
        else:
            consecutive_failures += 1
            
            if consecutive_failures >= max_consecutive_failures:
                # Restart with perturbation
                target_r = max(0.03, best_r * 0.98)
                perturbation = np.random.normal(0, 0.003, size=best_points.shape)
                current_points = best_points + perturbation
                # Ensure points stay within bounds
                norms = np.linalg.norm(current_points, axis=1)
                mask = norms > (1.0 - best_r)
                if np.any(mask):
                    current_points[mask] = best_points[mask]
                nn_indices = find_nearest_neighbors(current_points, k=12)
                consecutive_failures = 0
            else:
                # Small perturbation and retry
                perturbation = np.random.normal(0, 0.002, size=current_points.shape)
                current_points = current_points + perturbation
                norms = np.linalg.norm(current_points, axis=1)
                mask = norms > (1.0 - best_r)
                if np.any(mask):
                    current_points[mask] -= (norms[mask, np.newaxis] - (1.0 - best_r)) * current_points[mask] / norms[mask, np.newaxis]
        
        stage += 1

    return best_points

# EVOLVE-BLOCK-END
