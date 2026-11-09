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

def _lse_objective_and_grad(x, grid, beta):
    """
    Log-sum-exp surrogate for the covering radius.
    Objective: log(sum(exp(beta * d_i))) where d_i = arccos(max_j dot(grid_i, p_j))
    We minimize this.
    """
    p = x.reshape(-1, 3)
    norms = np.linalg.norm(p, axis=1, keepdims=True)
    p = p / norms
    
    # dots[i, j] = dot(grid[i], p[j])
    dots = grid @ p.T  # (n_grid, N)
    
    # For each grid point, find the max dot product (nearest point)
    max_dots = np.max(dots, axis=1)  # (n_grid,)
    
    # Angular distance = arccos(max_dot)
    # We want to minimize max(arccos(max_dot))
    # Use log-sum-exp: log(sum(exp(beta * arccos(max_dot))))
    
    # For numerical stability, compute arccos carefully
    max_dots_clipped = np.clip(max_dots, -1.0 + 1e-15, 1.0 - 1e-15)
    angular_dists = np.arccos(max_dots_clipped)  # (n_grid,)
    
    # Log-sum-exp
    max_ad = np.max(angular_dists)
    lse = max_ad + np.log(np.sum(np.exp(beta * (angular_dists - max_ad))))
    
    # Gradient
    # d(lse)/d(angular_dists_i) = beta * exp(beta * (angular_dists_i - max_ad)) / sum(exp(...))
    weights = beta * np.exp(beta * (angular_dists - max_ad))
    weights /= np.sum(weights)
    
    # d(angular_dists_i)/d(max_dots_i) = -1/sqrt(1 - max_dots_i^2)
    denom = np.sqrt(np.clip(1.0 - max_dots * max_dots, 1e-15, None))
    dad = -weights / denom  # (n_grid,)
    
    # d(max_dots_i)/d(p_j): max_dots_i = max_j dot(grid_i, p_j)
    # The gradient is only from the argmax j for each i
    argmax_j = np.argmax(dots, axis=1)  # (n_grid,)
    
    # grad_x[k] = sum over i where argmax_j[i] == k of dad[i] * grid[i]
    grad = np.zeros_like(x)
    for i in range(len(grid)):
        j = argmax_j[i]
        grad[j*3:(j+1)*3] += dad[i] * grid[i]
    
    return lse, grad

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    # Generate multiple initial configurations
    inits = []
    # Fibonacci sphere
    inits.append(_fibonacci_sphere(N))
    
    # Random with rejection for minimum separation
    rng = np.random.RandomState(42)
    for seed in [1, 2, 3]:
        rng = np.random.RandomState(seed)
        pts = rng.randn(N, 3)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        inits.append(pts)
    
    best_pts = None
    best_radius = np.inf
    
    for init_idx, init_pts in enumerate(inits):
        elapsed = time.time() - start_time
        if elapsed > deadline - 20.0:
            break
        
        current_pts = init_pts.copy()
        current_radius = _cover_radius_grid(current_pts, n_grid=5000)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_pts = current_pts.copy()
        
        # Phase 1: Coarse optimization with moderate beta
        grid_coarse = None
        n_grid_coarse = 3000
        i = np.arange(n_grid_coarse) + 0.5
        z = 1.0 - 2.0 * i / n_grid_coarse
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.clip(1.0 - z * z, 0, None))
        grid_coarse = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
        beta_coarse = 20.0
        
        def obj_grad_coarse(x):
            return _lse_objective_and_grad(x, grid_coarse, beta_coarse)
        
        try:
            result = minimize(
                obj_grad_coarse,
                current_pts.flatten(),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6, 'maxfun': 500}
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
        
        # Phase 2: Finer grid, higher beta
        if time.time() > deadline - 15.0:
            break
        
        n_grid_fine = 6000
        i = np.arange(n_grid_fine) + 0.5
        z = 1.0 - 2.0 * i / n_grid_fine
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.clip(1.0 - z * z, 0, None))
        grid_fine = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
        for beta in [50.0, 100.0, 200.0]:
            if time.time() > deadline - 10.0:
                break
            
            def obj_grad_fine(x, grid=grid_fine, beta=beta):
                return _lse_objective_and_grad(x, grid, beta)
            
            try:
                result = minimize(
                    obj_grad_fine,
                    current_pts.flatten(),
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 300}
                )
                new_pts = result.x.reshape(-1, 3)
                new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
                new_radius = _cover_radius_grid(new_pts, n_grid=8000)
                if new_radius < current_radius:
                    current_pts = new_pts
                    current_radius = new_radius
                    if current_radius < best_radius:
                        best_radius = current_radius
                        best_pts = current_pts.copy()
            except Exception:
                pass
    
    # Final polish with very fine grid
    if best_pts is not None and time.time() < deadline - 5.0:
        n_grid_vf = 10000
        i = np.arange(n_grid_vf) + 0.5
        z = 1.0 - 2.0 * i / n_grid_vf
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.clip(1.0 - z * z, 0, None))
        grid_vf = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
        for beta in [200.0, 400.0]:
            if time.time() > deadline - 3.0:
                break
            
            def obj_grad_vf(x, grid=grid_vf, beta=beta):
                return _lse_objective_and_grad(x, grid, beta)
            
            try:
                result = minimize(
                    obj_grad_vf,
                    best_pts.flatten(),
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 50, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 200}
                )
                new_pts = result.x.reshape(-1, 3)
                new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
                new_radius = _cover_radius_grid(new_pts, n_grid=12000)
                if new_radius < best_radius:
                    best_radius = new_radius
                    best_pts = new_pts.copy()
            except Exception:
                pass
    
    # Ensure valid output
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)
    
    best_pts = best_pts / np.linalg.norm(best_pts, axis=1, keepdims=True)
    
    return best_pts
# EVOLVE-BLOCK-END
