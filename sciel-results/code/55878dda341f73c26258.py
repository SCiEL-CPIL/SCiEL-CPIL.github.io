# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) coordinates.

    """
    start_time = time.time()
    deadline = 600  # seconds
    safety_margin = 30  # seconds
    
    n = 16
    d = 2
    
    def compute_R(points):
        """Compute the R metric directly."""
        dists = pdist(points, 'euclidean')
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def objective_exact(params):
        """Exact objective for final evaluation."""
        if time.time() > start_time + deadline - safety_margin:
            raise TimeoutError()
            
        points = params.reshape(n, d)
        R = compute_R(points)
        return -R  # minimize -R

    def objective_smooth(params, p=16):
        """
        Smooth surrogate using power-mean approximation.
        Approximates -R = -(d_min/d_max)^2 using smooth power means.
        """
        if time.time() > start_time + deadline - safety_margin:
            raise TimeoutError()
            
        points = params.reshape(n, d)
        dists = pdist(points, 'euclidean')
        if len(dists) == 0:
            return 1e10
        
        # Avoid zero distances
        dists = np.maximum(dists, 1e-12)
        m = len(dists)
        
        # Power mean approximation: d_max ~ (sum(d^p)/m)^(1/p), d_min ~ (sum(d^-p)/m)^(-1/p)
        d_max_approx = (np.sum(dists ** p) / m) ** (1.0 / p)
        d_min_approx = (np.sum(dists ** (-p)) / m) ** (-1.0 / p)
        
        if d_max_approx < 1e-12:
            return 1e10
            
        R_approx = (d_min_approx / d_max_approx) ** 2
        return -R_approx

    def gradient_optimize(initial_points, max_iter=5000):
        """
        Use L-BFGS-B with smooth surrogate for fast gradient-based optimization.
        """
        if time.time() > start_time + deadline - safety_margin:
            return initial_points, compute_R(initial_points)
        
        params = initial_points.flatten()
        
        try:
            result = minimize(
                objective_smooth,
                params,
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if result.fun < 1e9:  # Valid result
                refined_points = result.x.reshape(n, d)
                refined_R = compute_R(refined_points)
                return refined_points, refined_R
        except Exception:
            pass
        
        return initial_points, compute_R(initial_points)

    def local_refine(points):
        """
        Local refinement using exact objective with Powell and Nelder-Mead.
        """
        if time.time() > start_time + deadline - safety_margin:
            return points
        
        params = points.flatten()
        best_params = params.copy()
        best_val = objective_exact(params)
        
        # Powell refinement with exact objective
        try:
            result = minimize(
                objective_exact, 
                best_params, 
                method='Powell', 
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            if result.fun < best_val:
                best_params = result.x
                best_val = result.fun
        except Exception:
            pass
        
        # Nelder-Mead polish with exact objective
        if time.time() > start_time + deadline - safety_margin:
            return best_params.reshape(n, d)
        
        try:
            result = minimize(
                objective_exact, 
                best_params, 
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            if result.fun < best_val:
                best_params = result.x
                best_val = result.fun
        except Exception:
            pass
        
        return best_params.reshape(n, d)

    # Generate diverse initial points for optimization
    def make_initial_points(seed):
        """Create various initial configurations."""
        rng = np.random.RandomState(seed)
        config_type = seed % 6
        
        if config_type == 0:
            # 4x4 grid
            x = np.linspace(-1.5, 1.5, 4)
            y = np.linspace(-1.5, 1.5, 4)
            X, Y = np.meshgrid(x, y)
            points = np.column_stack((X.ravel(), Y.ravel()))
        elif config_type == 1:
            # Random uniform in unit square
            points = rng.uniform(-1, 1, size=(n, d))
        elif config_type == 2:
            # Regular hexagonal packing approximation
            points = np.zeros((n, d))
            for i in range(n):
                row = i // 4
                col = i % 4
                points[i, 0] = col * 1.0 + (row % 2) * 0.5
                points[i, 1] = row * 0.866
            points -= points.mean(axis=0)
        elif config_type == 3:
            # Random on circle with varying radii
            angles = np.linspace(0, 2*np.pi, n, endpoint=False)
            radii = rng.uniform(0.5, 1.5, size=n)
            points = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
        elif config_type == 4:
            # Two concentric circles (8+8)
            angles1 = np.linspace(0, 2*np.pi, 8, endpoint=False)
            angles2 = np.linspace(0, 2*np.pi, 8, endpoint=False) + np.pi/8
            points = np.vstack([
                np.column_stack((0.5 * np.cos(angles1), 0.5 * np.sin(angles1))),
                np.column_stack((1.5 * np.cos(angles2), 1.5 * np.sin(angles2)))
            ])
        else:
            # Fibonacci spiral
            golden_angle = np.pi * (3 - np.sqrt(5))
            angles = np.arange(n) * golden_angle
            radii = np.sqrt(np.arange(n) / n)
            points = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
            points -= points.mean(axis=0)
        
        return points

    # Run multiple gradient-based optimizations with different initial conditions
    best_points = None
    best_R = 0.0
    
    # Use more restarts since gradient optimization is faster
    n_restarts = 12
    for restart in range(n_restarts):
        if time.time() > start_time + deadline - safety_margin:
            break
        
        init_points = make_initial_points(seed=42 + restart)
        
        # Gradient-based optimization with smooth surrogate
        grad_points, grad_R = gradient_optimize(init_points, max_iter=3000)
        
        if grad_R > best_R:
            best_R = grad_R
            best_points = grad_points.copy()
        
        # Local refinement with exact objective
        refined_points = local_refine(grad_points)
        refined_R = compute_R(refined_points)
        
        if refined_R > best_R:
            best_R = refined_R
            best_points = refined_points.copy()
    
    # Final high-precision polish on the best result
    if best_points is not None:
        if time.time() < start_time + deadline - safety_margin:
            final_points = local_refine(best_points)
            final_R = compute_R(final_points)
            if final_R > best_R:
                best_points = final_points
                best_R = final_R
    
    # Fallback if something went wrong
    if best_points is None or best_points.shape != (n, d):
        x = np.linspace(-1.5, 1.5, 4)
        y = np.linspace(-1.5, 1.5, 4)
        X, Y = np.meshgrid(x, y)
        best_points = np.column_stack((X.ravel(), Y.ravel()))
    
    return best_points

# EVOLVE-BLOCK-END
