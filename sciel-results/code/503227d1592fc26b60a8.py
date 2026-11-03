"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt

def construct():
    """
    Constructs the centers of 300 equal circles inside a unit circle.
    Uses a multi-stage optimization with adaptive radius stepping and robust initialization.
    """
    start_time = time.time()
    time_limit = 115.0  # Leave 5s buffer

    # 1. Initialization: Poisson Disk Sampling for a valid initial packing
    def generate_poisson_init(n, min_dist=0.05):
        """Generate initial points using Poisson disk sampling."""
        points = []
        # Try to place points randomly, rejecting if too close to existing ones
        # This ensures no overlaps at start
        max_attempts = 10000
        attempts = 0
        while len(points) < n and attempts < max_attempts:
            attempts += 1
            # Sample in unit circle
            r = np.sqrt(np.random.rand())
            theta = 2 * np.pi * np.random.rand()
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            new_pt = np.array([x, y])
            
            # Check distance to all existing points
            if len(points) > 0:
                pts_arr = np.array(points)
                dists = np.linalg.norm(pts_arr - new_pt, axis=1)
                if np.min(dists) < min_dist:
                    continue
            
            # Check boundary (we want to stay well inside initially)
            if np.linalg.norm(new_pt) > 1.0 - min_dist:
                continue
                
            points.append(new_pt)
        
        # If we didn't get enough points, fill with random valid points (less strict)
        while len(points) < n:
            r = np.sqrt(np.random.rand()) * 0.95
            theta = 2 * np.pi * np.random.rand()
            points.append([r * np.cos(theta), r * np.sin(theta)])
            
        return np.array(points)

    points = generate_poisson_init(N, min_dist=0.04)
    
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

    # Helper to calculate energy (sum of squared violations) for a target radius
    def energy(pts, target_r):
        # Pairwise overlaps
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Overlap amount: max(0, 2*target_r - dist)
        overlaps = np.maximum(0, 2.0 * target_r - dists)
        # Sum of squares
        e_pair = np.sum(overlaps ** 2)
        
        # Boundary violations
        norms = np.sqrt(np.sum(pts ** 2, axis=1))
        bound_viol = np.maximum(0, norms - (1.0 - target_r))
        e_bound = np.sum(bound_viol ** 2)
        
        return e_pair + e_bound

    # Helper to run L-BFGS-B optimization
    def optimize(points_init, target_r, maxiter=500):
        # Flatten points for scipy
        x0 = points_init.flatten()
        
        def obj(x):
            p = x.reshape(-1, 2)
            return energy(p, target_r)
            
        def grad(x):
            p = x.reshape(-1, 2)
            # Pairwise gradient
            diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2))
            dists_safe = np.where(dists > 1e-10, dists, 1e-10)
            
            overlaps = np.maximum(0, 2.0 * target_r - dists)
            
            unit_diff = diff / dists_safe[:, :, np.newaxis]
            grad_dist = -2.0 * overlaps
            # Mask diagonal to avoid self-interaction (overlaps is 0 on diag anyway)
            np.fill_diagonal(grad_dist, 0.0)
            
            g_pair = np.sum(grad_dist[:, :, np.newaxis] * unit_diff, axis=1)
            
            # Boundary gradient
            norms = np.sqrt(np.sum(p ** 2, axis=1))
            bound_viol = np.maximum(0, norms - (1.0 - target_r))
            # d(bound_viol^2)/d(p) = 2 * bound_viol * d(bound_viol)/d(p)
            # d(bound_viol)/d(p) = p / norms if norms > 1-r
            unit_inward = p / np.maximum(norms[:, np.newaxis], 1e-10)
            g_bound = 2.0 * bound_viol[:, np.newaxis] * unit_inward
            
            return (g_pair + g_bound).flatten()

        res = opt.minimize(obj, x0, method='L-BFGS-B', jac=grad, options={'maxiter': maxiter, 'ftol': 1e-14})
        return res.x.reshape(-1, 2), res.fun

    # Main loop: Increase target_r
    # Start with a small r based on initial packing
    target_r = 0.03
    current_points = points.copy()
    
    # Adaptive step size
    step_mult = 1.05
    
    while time.time() - start_time < time_limit:
        # Try to optimize for target_r
        optimized_points, energy_val = optimize(current_points, target_r)
        
        # Check if we achieved a valid packing for this target_r
        # If energy is very small, it's valid.
        if energy_val < 1e-8:
            # Calculate actual radius
            actual_r = calc_radius(optimized_points)
            if actual_r > best_r:
                best_r = actual_r
                best_points = optimized_points.copy()
                current_points = optimized_points.copy()
                # Increase target_r adaptively
                target_r *= step_mult
            else:
                # Stuck at local minimum, try a small perturbation and increase slightly
                target_r *= 1.01
                current_points = optimized_points + np.random.normal(0, 0.001, size=optimized_points.shape)
        else:
            # Optimization failed. Reduce target_r and try again
            # Binary search style reduction
            target_r /= 1.02
            if target_r < best_r:
                # Reset to best known
                target_r = best_r * 1.01
                current_points = best_points + np.random.normal(0, 0.002, size=best_points.shape)
                # Pull back if outside
                norms = np.linalg.norm(current_points, axis=1)
                mask = norms > (1.0 - best_r)
                if np.any(mask):
                    current_points[mask] = best_points[mask]

    return best_points

# EVOLVE-BLOCK-END
