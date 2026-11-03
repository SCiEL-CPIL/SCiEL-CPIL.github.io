"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    """
    Place 50 equal non-overlapping circles in the unit square [-0.5, 0.5]^2
    to maximize the common radius.
    
    Uses physics-based force-directed relaxation with multi-start strategy.
    """
    
    def compute_radius(points):
        """Compute the maximum radius for a configuration."""
        # Pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        # Boundary distances
        boundary_dists = 0.5 - np.abs(points)
        min_boundary = np.min(boundary_dists)
        
        return min(min_pair / 2.0, min_boundary)
    
    def force_directed_optimize(points, n_iter=2000, lr=0.01, decay=0.999):
        """
        Optimize using force-directed method: push apart overlapping circles
        and pull toward center to maximize minimum distance.
        """
        points = points.copy().astype(float)
        current_lr = lr
        
        for i in range(n_iter):
            # Compute forces
            forces = np.zeros_like(points)
            
            # Pairwise repulsion: push apart if too close
            for a in range(N):
                for b in range(a + 1, N):
                    d = points[a] - points[b]
                    dist = np.sqrt(np.sum(d * d))
                    if dist < 1e-10:
                        # Random push for coincident points
                        d = np.random.randn(2)
                        dist = 1.0
                    # Strong repulsion when close, weaker when far
                    if dist < 0.2:
                        force_mag = (0.2 - dist) / 0.2
                        forces[a] += force_mag * d / dist
                        forces[b] -= force_mag * d / dist
            
            # Boundary repulsion: push inward if too close to edge
            for a in range(N):
                for dim in range(2):
                    if points[a, dim] > 0.4:
                        forces[a, dim] -= (points[a, dim] - 0.4) * 2
                    elif points[a, dim] < -0.4:
                        forces[a, dim] += (-0.4 - points[a, dim]) * 2
            
            # Centering force: gentle pull toward center to prevent drift
            forces -= 0.01 * points
            
            # Update positions
            points += current_lr * forces
            points = np.clip(points, -0.5, 0.5)
            
            # Decay learning rate
            current_lr *= decay
        
        return points
    
    def gradient_optimize(points):
        """
        Use gradient-based optimization to maximize the minimum radius.
        """
        def negative_radius(params):
            pts = params.reshape(N, 2)
            # Clip to valid range
            pts = np.clip(pts, -0.5, 0.5)
            r = compute_radius(pts)
            return -r
        
        def negative_radius_grad(params):
            pts = params.reshape(N, 2)
            pts = np.clip(pts, -0.5, 0.5)
            
            # Compute which constraint is active
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff * diff, axis=2))
            np.fill_diagonal(dists, np.inf)
            min_pair = np.min(dists)
            
            boundary_dists = 0.5 - np.abs(pts)
            min_boundary = np.min(boundary_dists)
            
            r = min(min_pair / 2.0, min_boundary)
            
            # Gradient of negative radius
            grad = np.zeros_like(pts)
            
            if min_pair / 2.0 <= min_boundary:
                # Pairwise distance is limiting
                # Find all pairs achieving minimum distance
                for a in range(N):
                    for b in range(a + 1, N):
                        if abs(dists[a, b] - min_pair) < 1e-8:
                            d = pts[a] - pts[b]
                            dist = dists[a, b]
                            if dist > 1e-10:
                                grad[a] += d / dist
                                grad[b] -= d / dist
                grad /= 2.0  # Because r = min_pair / 2
            else:
                # Boundary is limiting
                for a in range(N):
                    for dim in range(2):
                        if abs(boundary_dists[a, dim] - min_boundary) < 1e-8:
                            if pts[a, dim] > 0:
                                grad[a, dim] -= 1.0
                            else:
                                grad[a, dim] += 1.0
            
            return -grad  # Negative because we're minimizing negative radius
        
        initial_params = points.flatten()
        result = minimize(negative_radius, initial_params, 
                         method='L-BFGS-B',
                         jac=negative_radius_grad,
                         bounds=[(-0.5, 0.5)] * (2 * N),
                         options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
        
        return result.x.reshape(N, 2)
    
    def random_init(seed=42):
        """Generate random initial configuration."""
        rng = np.random.RandomState(seed)
        # Random positions with some structure
        points = rng.uniform(-0.4, 0.4, size=(N, 2))
        return points
    
    def grid_init_with_perturbation(rows, cols, seed=42):
        """Initialize on a grid with small perturbations."""
        rng = np.random.RandomState(seed)
        
        # Create grid
        if rows * cols >= N:
            xs = np.linspace(-0.45, 0.45, cols)
            ys = np.linspace(-0.45, 0.45, rows)
            X, Y = np.meshgrid(xs, ys)
            points = np.column_stack((X.ravel(), Y.ravel()))
            # Take first N points
            points = points[:N]
        else:
            # Fallback to random
            points = random_init(seed)
        
        # Add small perturbations
        points += rng.normal(0, 0.02, size=points.shape)
        points = np.clip(points, -0.5, 0.5)
        
        return points
    
    # Multi-start optimization
    np.random.seed(42)
    overall_best_points = None
    overall_best_score = -1
    
    start_time = time.time()
    time_limit = 115.0
    
    # Different initialization strategies
    init_configs = []
    
    # Grid-based inits
    for seed in [42, 123, 456, 789, 101, 202, 303, 404]:
        init_configs.append(('grid', grid_init_with_perturbation(10, 5, seed)))
        init_configs.append(('grid', grid_init_with_perturbation(5, 10, seed + 1000)))
    
    # Random inits
    for seed in [1, 2, 3, 4, 5, 6, 7, 8]:
        init_configs.append(('random', random_init(seed)))
    
    # Process each configuration
    for config_type, init_points in init_configs:
        elapsed = time.time() - start_time
        if elapsed > time_limit - 10:
            break
        
        # Phase 1: Force-directed relaxation
        fd_points = force_directed_optimize(init_points, n_iter=1000, lr=0.02, decay=0.9995)
        fd_score = compute_radius(fd_points)
        
        # Phase 2: Gradient optimization
        if time.time() - start_time < time_limit - 5:
            grad_points = gradient_optimize(fd_points)
            grad_score = compute_radius(grad_points)
        else:
            grad_points = fd_points
            grad_score = fd_score
        
        # Keep best
        if grad_score > overall_best_score:
            overall_best_score = grad_score
            overall_best_points = grad_points.copy()
    
    # Final refinement if time allows
    if overall_best_points is not None and time.time() - start_time < time_limit - 5:
        final_points = gradient_optimize(overall_best_points)
        final_score = compute_radius(final_points)
        if final_score > overall_best_score:
            overall_best_points = final_points
            overall_best_score = final_score
    
    if overall_best_points is None:
        # Fallback
        overall_best_points = grid_init_with_perturbation(10, 5, 42)
    
    return overall_best_points

# EVOLVE-BLOCK-END
