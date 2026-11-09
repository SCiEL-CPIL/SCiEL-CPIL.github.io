import numpy as np
import time
import scipy.optimize as opt

N = 50
DEADLINE = 115.0  # Safety margin from 120s (evaluator kills at 150s, but keep margin)

def construct():
    start_time = time.time()
    
    # Helper to compute the actual score (radius) of a configuration
    def compute_radius(centers):
        """Compute the radius for a given set of centers."""
        centers = np.asarray(centers)
        if len(centers) != N:
            return 0.0
        # Check for non-finite values
        if not np.all(np.isfinite(centers)):
            return 0.0
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        # Boundary distances
        norms = np.sqrt(np.sum(centers**2, axis=1))
        min_bound = np.min(1.0 - norms)
        # Radius is half of the minimum of these
        r = 0.5 * min(min_dist, min_bound)
        if r < 0:
            return 0.0
        return r

    # Helper to generate hexagonal lattice initial points
    def get_hex_initial(spacing=None):
        if spacing is None:
            spacing = 0.25
        points = []
        k = 12
        for i in range(-k, k+1):
            for j in range(-k, k+1):
                x = i * spacing + (j % 2) * spacing * 0.5
                y = j * spacing * np.sqrt(3) / 2
                if x**2 + y**2 <= 0.95:
                    points.append([x, y])
        points = np.array(points)
        if len(points) < N:
            return np.random.uniform(-0.85, 0.85, (N, 2))
        # Select N points closest to center
        dists = np.linalg.norm(points, axis=1)
        indices = np.argsort(dists)[:N]
        return points[indices]

    # Helper to generate ring-based initial points
    def get_ring_initial():
        """Generate points in concentric rings."""
        points = []
        # Center point
        points.append([0.0, 0.0])
        # Ring 1: 6 points
        r1 = 0.25
        for i in range(6):
            angle = 2 * np.pi * i / 6
            points.append([r1 * np.cos(angle), r1 * np.sin(angle)])
        # Ring 2: 12 points
        r2 = 0.5
        for i in range(12):
            angle = 2 * np.pi * i / 12 + np.pi / 12
            points.append([r2 * np.cos(angle), r2 * np.sin(angle)])
        # Ring 3: 12 points
        r3 = 0.72
        for i in range(12):
            angle = 2 * np.pi * i / 12
            points.append([r3 * np.cos(angle), r3 * np.sin(angle)])
        # Ring 4: 10 points
        r4 = 0.9
        for i in range(10):
            angle = 2 * np.pi * i / 10 + np.pi / 10
            points.append([r4 * np.cos(angle), r4 * np.sin(angle)])
        
        points = np.array(points[:N])
        if len(points) < N:
            # Fill with random points
            extra = np.random.uniform(-0.8, 0.8, (N - len(points), 2))
            points = np.vstack([points, extra])
        return points

    # Objective function: minimize negative of the radius (equivalent to maximizing radius)
    # Using a smooth approximation for differentiability
    def objective(c, sharpness=100.0):
        """Minimize this to maximize the minimum radius."""
        centers = c.reshape(N, 2)
        
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Boundary distances
        norms = np.sqrt(np.sum(centers**2, axis=1))
        bound_dists = 1.0 - norms
        
        # Combine all constraints
        all_dists = np.concatenate([dists[np.triu_indices(N, k=1)], bound_dists])
        
        # Use soft min: -log(sum(exp(-sharpness * d))) / sharpness
        # This approximates -min(dists) for large sharpness
        # To avoid overflow, use the log-sum-exp trick
        min_d = np.min(all_dists)
        exp_terms = np.exp(-sharpness * (all_dists - min_d))
        soft_min = min_d + np.log(np.sum(exp_terms)) / sharpness
        
        # We want to maximize min_d, so minimize -soft_min
        return -soft_min

    # Gradient-based objective with hard constraints (for L-BFGS-B or similar)
    def objective_hard(c):
        """Objective that returns a large penalty if constraints are violated."""
        centers = c.reshape(N, 2)
        
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Boundary distances
        norms = np.sqrt(np.sum(centers**2, axis=1))
        bound_dists = 1.0 - norms
        
        all_dists = np.concatenate([dists[np.triu_indices(N, k=1)], bound_dists])
        min_d = np.min(all_dists)
        
        return -min_d

    # Start with multiple initial configurations
    initial_configs = []
    
    # Hexagonal lattice with different spacings
    for spacing in [0.22, 0.24, 0.26, 0.28]:
        init = get_hex_initial(spacing)
        # Add small noise
        init += np.random.normal(0, 0.005, init.shape)
        # Clip to inside
        norms = np.linalg.norm(init, axis=1)
        scale = np.where(norms > 0.95, 0.95 / norms, 1.0)
        init *= scale[:, np.newaxis]
        initial_configs.append(init)
    
    # Ring-based
    init_ring = get_ring_initial()
    init_ring += np.random.normal(0, 0.005, init_ring.shape)
    norms = np.linalg.norm(init_ring, axis=1)
    scale = np.where(norms > 0.95, 0.95 / norms, 1.0)
    init_ring *= scale[:, np.newaxis]
    initial_configs.append(init_ring)
    
    # Random
    init_rand = np.random.uniform(-0.85, 0.85, (N, 2))
    norms = np.linalg.norm(init_rand, axis=1)
    scale = np.where(norms > 0.95, 0.95 / norms, 1.0)
    init_rand *= scale[:, np.newaxis]
    initial_configs.append(init_rand)
    
    best_centers = initial_configs[0]
    best_r = compute_radius(best_centers)
    
    # Optimize each initial configuration
    for init in initial_configs:
        if time.time() > DEADLINE - 10:
            break
        
        c0 = init.flatten()
        
        # Try L-BFGS-B with the soft-min objective
        try:
            res = opt.minimize(
                objective,
                c0,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
            )
            
            final_centers = res.x.reshape(N, 2)
            final_r = compute_radius(final_centers)
            
            if final_r > best_r:
                best_r = final_r
                best_centers = final_centers.copy()
                
        except Exception as e:
            pass
        
        # If time permits, try a second optimization pass with Nelder-Mead from the best result
        if time.time() < DEADLINE - 15:
            try:
                c1 = best_centers.flatten()
                res2 = opt.minimize(
                    objective_hard,
                    c1,
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                
                final_centers2 = res2.x.reshape(N, 2)
                final_r2 = compute_radius(final_centers2)
                
                if final_r2 > best_r:
                    best_r = final_r2
                    best_centers = final_centers2.copy()
                    
            except Exception as e:
                pass
    
    # Final verification: ensure all centers are inside the unit circle
    norms = np.linalg.norm(best_centers, axis=1)
    if np.any(norms >= 1.0):
        # Scale down slightly
        scale = 0.999 / np.maximum(norms, 1e-10)
        best_centers *= scale[:, np.newaxis]
    
    return best_centers

if __name__ == "__main__":
    c = construct()
    r = compute_radius(c)
    print(f"Radius: {r}")
