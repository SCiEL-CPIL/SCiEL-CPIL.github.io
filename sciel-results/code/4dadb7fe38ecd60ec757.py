"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, None))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _cover_radius_grid(pts, n_grid=8000):
    """Compute covering radius using a Fibonacci grid."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    i = np.arange(n_grid) + 0.5
    z = 1.0 - 2.0 * i / n_grid
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, None))
    grid = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    dots = grid @ pts.T
    min_dots = np.max(dots, axis=1)
    return np.arccos(np.clip(min_dots, -1.0, 1.0)).max()

def _pairwise_potential_obj_grad(x, sigma):
    """
    Objective: Sum of exponential pairwise potentials exp(-d_ij / sigma).
    This minimizes the sum, pushing points apart.
    Gradient is the repulsive force.
    """
    p = x.reshape(-1, 3)
    norms = np.linalg.norm(p, axis=1, keepdims=True)
    p = p / norms
    
    # Compute all pairwise dot products
    dots = p @ p.T
    np.fill_diagonal(dots, 1.0)
    
    # Clip for arccos stability
    dots_clipped = np.clip(dots, -1.0 + 1e-15, 1.0 - 1e-15)
    
    # Angular distances
    dists = np.arccos(dots_clipped)
    
    # Potential
    # Mask diagonal to avoid self-interaction
    mask = ~np.eye(len(p), dtype=bool)
    potential = np.sum(np.exp(-dists[mask] / sigma))
    
    # Gradient
    # d/dp_i [ exp(-d_ij/sigma) ] = exp(-d_ij/sigma) * (-1/sigma) * d(d_ij)/d(p_i)
    # d(d_ij)/d(p_i) = -p_j / sqrt(1 - dots_ij^2)
    # So term is exp(-d_ij/sigma) * (1/sigma) * p_j / sqrt(1 - dots_ij^2)
    
    # Compute denominator for gradient
    denom = np.sqrt(np.clip(1.0 - dots * dots, 1e-15, None))
    np.fill_diagonal(denom, 1.0) # Avoid div by zero on diag, though we mask it
    
    # Weights for gradient
    weights = np.exp(-dists / sigma) / sigma
    np.fill_diagonal(weights, 0.0)
    
    # grad_p[i] = sum_j weights[i,j] * (p[j] / denom[i,j])
    # This is a matrix multiplication: (weights / denom) @ p
    grad_weights = weights / denom
    grad_p = grad_weights @ p
    
    return potential, grad_p.flatten()

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Initial configurations
    inits = [_fibonacci_sphere(N)]
    for seed in [1, 2, 3, 4, 5]:
        rng = np.random.RandomState(seed)
        pts = rng.randn(N, 3)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        inits.append(pts)
        
    best_pts = None
    best_radius = np.inf
    
    # Phase 1: Pairwise Separation Optimization (NQP-like)
    # Use different sigmas to control the "reach" of the repulsion
    sigmas = [0.1, 0.2, 0.5, 1.0]
    
    for init_idx, init_pts in enumerate(inits):
        if time.time() > deadline - 30.0:
            break
            
        current_pts = init_pts.copy()
        current_radius = _cover_radius_grid(current_pts, n_grid=5000)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_pts = current_pts.copy()
            
        for sigma in sigmas:
            if time.time() > deadline - 20.0:
                break
                
            def obj_grad(x, sigma=sigma):
                return _pairwise_potential_obj_grad(x, sigma)
                
            try:
                result = minimize(
                    obj_grad,
                    current_pts.flatten(),
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-5, 'maxfun': 200}
                )
                new_pts = result.x.reshape(-1, 3)
                new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
                
                new_radius = _cover_radius_grid(new_pts, n_grid=5000)
                if new_radius < current_radius:
                    current_pts = new_pts
                    current_radius = new_radius
                    if current_radius < best_radius:
                        best_radius = current_radius
                        best_pts = current_pts.copy()
            except Exception:
                pass
    
    # Phase 2: Covering Radius Refinement using LSE (from original logic, simplified)
    if best_pts is not None and time.time() < deadline - 10.0:
        # Use a grid for LSE refinement
        n_grid = 5000
        i = np.arange(n_grid) + 0.5
        z = 1.0 - 2.0 * i / n_grid
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.clip(1.0 - z * z, 0, None))
        grid = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
        def lse_obj_grad(x, grid=grid, beta=50.0):
            p = x.reshape(-1, 3)
            norms = np.linalg.norm(p, axis=1, keepdims=True)
            p = p / norms
            dots = grid @ p.T
            max_dots = np.max(dots, axis=1)
            max_dots_clipped = np.clip(max_dots, -1.0 + 1e-15, 1.0 - 1e-15)
            angular_dists = np.arccos(max_dots_clipped)
            
            max_ad = np.max(angular_dists)
            lse = max_ad + np.log(np.sum(np.exp(beta * (angular_dists - max_ad))))
            
            weights = beta * np.exp(beta * (angular_dists - max_ad))
            weights /= np.sum(weights)
            
            denom = np.sqrt(np.clip(1.0 - max_dots * max_dots, 1e-15, None))
            dad = -weights / denom
            
            argmax_j = np.argmax(dots, axis=1)
            
            grad = np.zeros_like(x)
            for i_idx in range(len(grid)):
                j_idx = argmax_j[i_idx]
                grad[j_idx*3:(j_idx+1)*3] += dad[i_idx] * grid[i_idx]
            
            return lse, grad
        
        try:
            result = minimize(
                lse_obj_grad,
                best_pts.flatten(),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 100}
            )
            new_pts = result.x.reshape(-1, 3)
            new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
            new_radius = _cover_radius_grid(new_pts, n_grid=8000)
            if new_radius < best_radius:
                best_radius = new_radius
                best_pts = new_pts.copy()
        except Exception:
            pass
            
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)
        
    best_pts = best_pts / np.linalg.norm(best_pts, axis=1, keepdims=True)
    return best_pts
# EVOLVE-BLOCK-END
