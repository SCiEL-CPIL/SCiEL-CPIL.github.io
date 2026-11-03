import numpy as np
import time

N = 50

def construct():
    # Spectral Initialization Strategy
    # We generate points by projecting the top N eigenvectors of a random SPD matrix
    # into 3D space. This leverages orthogonality to ensure initial spread.
    
    def spectral_init(dim, n_points, seed):
        """
        Generate n_points in 3D by projecting top n_points eigenvectors
        of a random dim x dim SPD matrix.
        """
        np.random.seed(seed)
        # Generate random symmetric matrix
        A = np.random.randn(dim, dim)
        A = (A + A.T) / 2
        # Ensure positive definite by adding identity
        A = A + np.eye(dim) * 0.1
        
        # Compute eigenvectors
        # We only need top n_points, but numpy doesn't have a direct partial eigenvector
        # function that is fast enough for large dim if we do it many times.
        # For dim=100, full eig is fast (~ms).
        vals, vecs = np.linalg.eigh(A)
        
        # Sort by eigenvalue descending
        idx = np.argsort(vals)[::-1]
        top_vecs = vecs[:, idx[:n_points]]
        
        # top_vecs is (dim, n_points). Rows are eigenvectors.
        # We want n_points points in 3D.
        # Project using a fixed random projection matrix P (dim x 3)
        P = np.random.randn(dim, 3)
        P = P / np.linalg.norm(P, axis=0, keepdims=True)
        
        # Centers = top_vecs.T @ P  -> (n_points, 3)
        centers = top_vecs.T @ P
        
        # Normalize and scale to fit in cube [-0.4, 0.4]
        # Center
        centers -= np.mean(centers, axis=0)
        # Scale
        max_abs = np.max(np.abs(centers))
        if max_abs > 1e-9:
            centers /= max_abs
        centers *= 0.4
        
        # Clip to valid range
        centers = np.clip(centers, -0.49, 0.49)
        
        return centers

    def compute_radius(centers):
        """Compute the maximum valid radius for given centers."""
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        d2 = np.sum(diff**2, axis=2)
        np.fill_diagonal(d2, 1e-9)
        d_min = np.sqrt(np.min(d2[np.triu_indices(N, k=1)]))
        
        # Wall distances
        # min(0.5 - |c_ik|)
        wall_dists = 0.5 - np.abs(centers)
        w_min = np.min(wall_dists)
        
        r = min(d_min / 2.0, w_min)
        return r

    def optimize(centers, iterations=2000, lr_start=0.01, lr_end=0.001):
        """
        Force-directed optimization with adaptive step size.
        """
        centers = centers.copy()
        best_r = compute_radius(centers)
        best_centers = centers.copy()
        
        for i in range(iterations):
            # Decay learning rate
            lr = lr_start * (lr_end / lr_start) ** (i / iterations)
            
            # Calculate forces
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            d2 = np.sum(diff**2, axis=2)
            np.fill_diagonal(d2, 1e-9)
            d = np.sqrt(d2)
            np.fill_diagonal(d, 1e-9)
            
            # Repulsive force: (c_i - c_j) / d^3
            d3 = d ** 3
            np.fill_diagonal(d3, 1e-9)
            
            force_spheres = np.sum(diff / d3[:, :, np.newaxis], axis=1)
            
            # Wall forces
            # Push away from walls if close
            force_walls = np.zeros_like(centers)
            K_wall = 10.0
            eps = 1e-6
            
            for k in range(3):
                c = centers[:, k]
                # Distance to +0.5 wall
                dist_pos = 0.5 - c
                # Distance to -0.5 wall
                dist_neg = c + 0.5
                
                # Only apply strong force if very close to wall to avoid instability
                # Use a smooth potential? Or just hard clip?
                # Let's use a soft penalty gradient.
                # Potential U = K / dist^2 for dist < 0.1
                # dU/dc = 2K / dist^3 * sign
                # But we want to minimize U, so force is -dU/dc.
                # If c > 0, dist = 0.5 - c. dU/dc = 2K / (0.5-c)^3. Force = -2K/(0.5-c)^3.
                # This is huge. Let's use a milder penalty or just clip.
                
                # Simpler: Linear force if outside "safe" zone
                safe = 0.1
                f_pos = np.where(dist_pos < safe, -K_wall / (dist_pos + eps), 0.0)
                f_neg = np.where(dist_neg < safe, K_wall / (dist_neg + eps), 0.0)
                
                force_walls[:, k] = f_pos + f_neg
            
            total_force = force_spheres + force_walls
            
            # Adaptive Step Size
            # Normalize force to unit vector per particle, then scale by lr
            force_norms = np.linalg.norm(total_force, axis=1, keepdims=True)
            force_norms = np.where(force_norms < 1e-9, 1.0, force_norms)
            normalized_force = total_force / force_norms
            
            step = lr * normalized_force
            
            # Clip step to prevent explosion
            max_step = 0.005
            step_norms = np.linalg.norm(step, axis=1, keepdims=True)
            mask = step_norms > max_step
            if np.any(mask):
                step[mask] = step[mask] * (max_step / step_norms[mask])
            
            centers += step
            
            # Clip to cube
            centers = np.clip(centers, -0.49, 0.49)
            
            # Check radius
            r_current = compute_radius(centers)
            if r_current > best_r:
                best_r = r_current
                best_centers = centers.copy()
            
            # Early stopping if no improvement for a long time?
            # Keep it simple: run all iterations.
        
        return best_centers, best_r

    # Main Loop
    import time
    start_time = time.time()
    time_limit = 110.0 # Leave 10s buffer
    
    best_overall_r = 0.0
    best_overall_centers = None
    
    # Try several spectral initializations with different dimensions
    # Higher dim -> more orthogonal -> better spread
    dims = [50, 100, 200]
    seeds = [1, 2, 3, 4, 5]
    
    for dim in dims:
        for seed in seeds:
            if time.time() - start_time > time_limit:
                break
            
            centers = spectral_init(dim, N, seed)
            centers, r = optimize(centers, iterations=1500, lr_start=0.02, lr_end=0.001)
            
            if r > best_overall_r:
                best_overall_r = r
                best_overall_centers = centers
                
    # Polish the best solution
    if time.time() - start_time < time_limit and best_overall_centers is not None:
        centers, r = optimize(best_overall_centers, iterations=2000, lr_start=0.005, lr_end=0.0001)
        if r > best_overall_r:
            best_overall_r = r
            best_overall_centers = centers

    return best_overall_centers
