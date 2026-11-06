"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint, Bounds


def construct():
    """
    Place 200 equal circles in [-0.5, 0.5]^2 to maximize the minimum radius.
    Uses a hexagonal grid initialization followed by continuous local optimization
    using a smooth surrogate for the minimum distance, with a final polish pass.
    """
    start_time = time.time()
    deadline = 115.0  # seconds, safety margin before 120s limit

    # --- Helper: compute the packing radius for a given set of centers ---
    def _compute_radius(centers):
        """Compute the largest common radius for the given centers."""
        centers = np.asarray(centers, dtype=np.float64)
        n = len(centers)
        if n < 2:
            return 0.5 - np.max(np.abs(centers))
        
        # Compute all pairwise distances
        # Use a vectorized approach for efficiency
        # For 200 points, 200*199/2 = 19900 pairs, which is fine
        diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=2))
        # Set diagonal to infinity
        np.fill_diagonal(dists, np.inf)
        min_pairwise = np.min(dists)
        
        # Distance to walls
        # centers are in [-0.5, 0.5]^2
        wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
        min_wall = np.min(wall_dists)
        
        return min(min_pairwise / 2.0, min_wall)

    # --- Helper: smooth surrogate for the objective ---
    # We want to maximize min(min_pairwise/2, min_wall)
    # Equivalently, we want to maximize t subject to:
    #   ||c_i - c_j|| >= 2t for all i<j
    #   0.5 - |c_ik| >= t for all i,k
    # 
    # A smooth approach: minimize -t + lambda * sum of violations
    # Or use a barrier/penalty method.
    # 
    # Alternative: minimize a smooth approximation of the max of violations.
    # Let f(t, c) = max(2t - min_pairwise, 2t - min_wall*2) ... this is messy.
    #
    # Better: parameterize by t and minimize the maximum constraint violation.
    # Or, use a log-sum-exp soft-min.
    #
    # Let's use a different approach: optimize the positions directly to maximize
    # a soft-min of the pairwise distances and wall distances.
    #
    # Objective: maximize S(c) = softmin( [d_ij/2 for all i<j], [0.5-|c_ik| for all i,k] )
    # softmin(x, beta) = -1/beta * log(sum(exp(-beta*x)))
    # For large beta, this approximates min(x).
    
    def _softmin(values, beta):
        """Compute soft-min of an array of values."""
        values = np.asarray(values)
        # Clip for numerical stability
        max_val = np.max(values)
        exp_vals = np.exp(-beta * (values - max_val))
        log_sum = np.log(np.sum(exp_vals))
        return max_val - log_sum / beta

    def _objective_and_grad(flat_c, beta, t_init):
        """
        Objective to minimize: -softmin of distances + penalty for t > feasible.
        Actually, let's just minimize -softmin of the distances.
        """
        c = flat_c.reshape(-1, 2)
        n = len(c)
        
        # Pairwise distances
        diffs = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        pairwise_half = dists / 2.0
        
        # Wall distances
        wall_dists = 0.5 - np.max(np.abs(c), axis=1)
        
        # Combine all distances
        all_dists = np.concatenate([pairwise_half[np.triu_indices(n, k=1)], wall_dists])
        
        # Soft-min
        sm = _softmin(all_dists, beta)
        
        return -sm
    
    # --- Initialization: hexagonal grid ---
    def _hex_grid_init(n_circles):
        """Generate a hexagonal grid of n_circles points in [-0.5, 0.5]^2."""
        # Approximate number of rows
        # For hexagonal packing, density is pi/(2*sqrt(3)) ~ 0.9069
        # Area per circle ~ 1/n_circles
        # Let's just try a few configurations
        
        # Start with a square grid, then shift every other row
        m = int(np.ceil(np.sqrt(n_circles)))
        
        # Try different m values
        best_centers = None
        best_score = -np.inf
        
        for m_try in range(m-2, m+3):
            if m_try < 1:
                continue
            # Number of rows and columns
            # For hexagonal: offset every other row
            n_rows = m_try
            n_cols = m_try
            
            # Generate hexagonal grid
            # Spacing
            # For n_circles circles, the spacing should be roughly 1/sqrt(n_circles)
            # But we need to fit in the square.
            
            # Let's use a standard hexagonal grid
            # Row i, column j:
            # x = j * dx + (i % 2) * dx/2
            # y = i * dy
            # where dx = 2*r, dy = sqrt(3)*r
            
            # We want to fit n_circles points.
            # Let's try to generate more than n_circles and take the first n_circles.
            
            # Try different spacings
            for scale in [0.9, 0.95, 1.0, 1.05, 1.1]:
                # Base spacing
                base_dx = 1.0 / m_try
                dx = base_dx * scale
                dy = base_dx * np.sqrt(3) * scale / 2.0  # Adjust for hexagonal
                
                # Actually, for hexagonal packing with circle radius r:
                # dx = 2r, dy = sqrt(3)*r
                # So if we want m_try columns, total width = (m_try-1)*dx + dx/2 (for offset)
                # Let's just generate a grid and center it.
                
                centers_list = []
                for i in range(n_rows):
                    for j in range(n_cols):
                        x = j * dx + (i % 2) * dx / 2.0
                        y = i * dy
                        centers_list.append([x, y])
                
                if len(centers_list) < n_circles:
                    continue
                
                centers = np.array(centers_list[:n_circles])
                
                # Center the grid
                centers -= np.mean(centers, axis=0)
                
                # Scale to fit in [-0.5, 0.5]^2
                max_abs = np.max(np.abs(centers))
                if max_abs > 0:
                    centers *= 0.5 / max_abs
                
                # Ensure within bounds
                centers = np.clip(centers, -0.5, 0.5)
                
                # Score
                score = _compute_radius(centers)
                if score > best_score:
                    best_score = score
                    best_centers = centers.copy()
        
        if best_centers is None:
            # Fallback to square grid
            m = int(np.ceil(np.sqrt(n_circles)))
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y) for y in ticks for x in ticks])
            best_centers = grid[:n_circles]
        
        return best_centers

    # --- Optimization ---
    def _optimize(centers0, max_iter=200, tol=1e-12):
        """Optimize centers using gradient descent on the soft-min objective."""
        c = centers0.copy().astype(np.float64)
        n = len(c)
        flat_c = c.flatten()
        
        # Start with a moderate beta and increase it
        betas = [10.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
        
        for beta in betas:
            if time.time() - start_time > deadline:
                break
            
            # Use L-BFGS-B for optimization
            # We need gradients. Let's use numerical gradients or analytical.
            # For speed, let's use a simple gradient descent with analytical gradients.
            
            # Actually, let's use scipy's minimize with BFGS or L-BFGS-B.
            # The objective is -softmin, so we minimize it.
            
            def obj(flat_x):
                x = flat_x.reshape(-1, 2)
                # Ensure within bounds (penalty)
                # Clip for now
                x_clipped = np.clip(x, -0.5, 0.5)
                return _objective_and_grad(x_clipped.flatten(), beta, None)
            
            # Use L-BFGS-B with bounds
            bounds = [(-0.5, 0.5)] * (2 * n)
            
            try:
                result = minimize(
                    obj,
                    flat_c,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol}
                )
                flat_c = result.x
            except Exception:
                pass
        
        c = flat_c.reshape(-1, 2)
        # Final clip
        c = np.clip(c, -0.5, 0.5)
        return c

    # --- Main algorithm ---
    # 1. Initialize with hexagonal grid
    centers = _hex_grid_init(N)
    current_score = _compute_radius(centers)
    
    # 2. Optimize
    if time.time() - start_time < deadline - 30:
        centers = _optimize(centers, max_iter=100, tol=1e-10)
        new_score = _compute_radius(centers)
        if new_score > current_score:
            current_score = new_score
    
    # 3. Polish pass with tighter tolerance
    if time.time() - start_time < deadline - 10:
        centers = _optimize(centers, max_iter=50, tol=1e-14)
        new_score = _compute_radius(centers)
        if new_score > current_score:
            current_score = new_score
    
    # 4. Final check and cleanup
    centers = np.clip(centers, -0.5, 0.5)
    
    return centers
# EVOLVE-BLOCK-END
