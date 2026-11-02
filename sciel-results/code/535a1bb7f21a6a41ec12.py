"""Optimized 100-point logarithmic energy minimizer on the unit sphere."""
import numpy as np
import time
import math
from scipy.optimize import minimize, check_grad

N = 100
DEADLINE_SECONDS = 115.0  # safety margin before 120s kill
START_TIME = time.time()

def _remaining_time():
    return max(0.0, DEADLINE_SECONDS - (time.time() - START_TIME))

def _energy_and_grad(X):
    """
    Compute logarithmic energy sum_{i<j} log(1/||x_i - x_j||) and its gradient
    for points X of shape (N, 3) on the unit sphere.
    Energy = sum_{i<j} -log(||x_i - x_j||)
    Gradient w.r.t. x_i = sum_{j!=i} (x_i - x_j) / ||x_i - x_j||^2
    """
    # Pairwise differences: D[i,j] = X[i] - X[j]
    D = X[:, np.newaxis, :] - X[np.newaxis, :, :]  # (N, N, 3)
    dists = np.linalg.norm(D, axis=2)  # (N, N)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dists, 1.0)
    
    # Energy: sum over i<j of -log(dists[i,j])
    # Use upper triangle
    iu = np.triu_indices(N, k=1)
    energy = -np.sum(np.log(dists[iu]))
    
    # Gradient: for each i, grad_i = sum_{j!=i} (X[i] - X[j]) / dists[i,j]^2
    # D[i,j] / dists[i,j]^2, summed over j for each i
    # But we need to be careful: D[i,j] = X[i] - X[j]
    # grad_i = sum_j D[i,j] / dists[i,j]^2  (j != i)
    
    # Build weight matrix: W[i,j] = 1/dists[i,j]^2 for i!=j, 0 for i==j
    W = np.zeros((N, N))
    W[iu[0], iu[1]] = 1.0 / (dists[iu] ** 2)
    W[iu[1], iu[0]] = 1.0 / (dists[iu] ** 2)
    
    # grad = D @ W, where D is (N,N,3) and W is (N,N)
    # grad[i] = sum_j D[i,j] * W[i,j]
    grad = np.einsum('ijk,jk->i', D, W)
    
    return energy, grad

def _project_to_sphere(X):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return X / norms

def _riemannian_cg(X0, max_iter=500, tol=1e-12, deadline_check=None):
    """
    Riemannian conjugate gradient on the sphere for minimizing logarithmic energy.
    """
    X = _project_to_sphere(X0.copy())
    f, g = _energy_and_grad(X)
    
    # Project gradient to tangent space
    g_tan = g - np.sum(g * X, axis=1, keepdims=True) * X
    
    # Initial search direction
    d = -g_tan
    
    for k in range(max_iter):
        if deadline_check and deadline_check() < 2.0:
            break
            
        # Check convergence
        if np.linalg.norm(g_tan) < tol:
            break
        
        # Line search: minimize f(X + t*d) projected to sphere
        # Use simple backtracking line search
        alpha = 1.0
        c1 = 1e-4
        for ls_iter in range(30):
            X_new = _project_to_sphere(X + alpha * d)
            f_new, g_new = _energy_and_grad(X_new)
            g_new_tan = g_new - np.sum(g_new * X_new, axis=1, keepdims=True) * X_new
            
            # Armijo condition
            if f_new <= f + c1 * alpha * np.sum(g_tan * d):
                break
            alpha *= 0.5
        
        # Update
        X = X_new
        f = f_new
        g = g_new
        g_tan = g_new_tan
        
        # Compute beta (Polak-Ribiere)
        g_new_tan_dot_d = np.sum(g_new_tan * d)
        d_old_norm_sq = np.sum(d * d)
        beta = max(0.0, g_new_tan_dot_d / d_old_norm_sq) if d_old_norm_sq > 1e-15 else 0.0
        
        d = -g_new_tan + beta * d
        # Re-orthogonalize direction to tangent space
        d = d - np.sum(d * X, axis=1, keepdims=True) * X
        
        # Restart if direction not good
        if np.sum(d * g_new_tan) >= 0:
            d = -g_new_tan
    
    return X, f

def _lbfgs_polish(X0, max_iter=1000, tol=1e-14):
    """
    L-BFGS-B polishing with penalty method for sphere constraint.
    """
    X_flat = X0.flatten()
    
    def penalty_obj(v):
        X = v.reshape(N, 3)
        # Penalty for not being on sphere
        norms = np.linalg.norm(X, axis=1)
        penalty = 1e6 * np.sum((norms - 1.0) ** 2)
        f, g = _energy_and_grad(_project_to_sphere(X))
        # Gradient of penalty
        g_pen = 2e6 * (norms - 1.0) * X / np.maximum(norms, 1e-12)
        return f + penalty, g + g_pen
    
    res = minimize(penalty_obj, X_flat, method='L-BFGS-B', 
                   options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-12, 'maxls': 50})
    X_opt = _project_to_sphere(res.x.reshape(N, 3))
    f_opt, _ = _energy_and_grad(X_opt)
    return X_opt, f_opt

def _diverse_initializations():
    """Generate multiple diverse initial configurations."""
    inits = []
    
    # 1. Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    X_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    inits.append(X_fib)
    
    # 2. Icosahedral-based: start from icosahedron vertices and subdivide
    # Simple approach: use more random points
    rng = np.random.default_rng(42)
    for seed in [0, 1, 2, 3, 4, 5, 6, 7]:
        rng = np.random.default_rng(seed)
        # Random points on sphere
        theta = rng.uniform(0, 2 * np.pi, N)
        z = rng.uniform(-1, 1, N)
        r = np.sqrt(1 - z**2)
        X_rand = np.column_stack([r * np.cos(theta), r * np.sin(theta), z])
        inits.append(X_rand)
    
    # 3. Grid-based initialization
    # Approximate spherical grid
    n_lat = 10
    n_lon = 10
    lats = np.linspace(-np.pi/2 + np.pi/(2*n_lat), np.pi/2 - np.pi/(2*n_lat), n_lat)
    lons = np.linspace(0, 2*np.pi, n_lon, endpoint=False)
    X_grid = []
    for lat in lats:
        for lon in lons:
            x = np.cos(lat) * np.cos(lon)
            y = np.cos(lat) * np.sin(lon)
            z = np.sin(lat)
            X_grid.append([x, y, z])
    X_grid = np.array(X_grid[:N])
    if len(X_grid) < N:
        # Pad with random
        pad = np.random.randn(N - len(X_grid), 3)
        pad /= np.linalg.norm(pad, axis=1, keepdims=True)
        X_grid = np.vstack([X_grid, pad])
    inits.append(X_grid)
    
    return inits

def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_X = None
    best_E = np.inf
    
    # Generate diverse initializations
    inits = _diverse_initializations()
    
    for X0 in inits:
        if _remaining_time() < 5.0:
            break
        
        # Riemannian CG refinement
        X_cg, E_cg = _riemannian_cg(X0, max_iter=300, tol=1e-12, 
                                      deadline_check=_remaining_time)
        
        if E_cg < best_E:
            best_E = E_cg
            best_X = X_cg.copy()
        
        # L-BFGS polishing if time permits
        if _remaining_time() > 10.0:
            X_lb, E_lb = _lbfgs_polish(X_cg, max_iter=500, tol=1e-14)
            if E_lb < best_E:
                best_E = E_lb
                best_X = X_lb.copy()
    
    # Final high-precision polish on best solution
    if best_X is not None and _remaining_time() > 5.0:
        X_final, E_final = _lbfgs_polish(best_X, max_iter=2000, tol=1e-15)
        if E_final < best_E:
            best_E = E_final
            best_X = X_final
    
    # Ensure valid output
    if best_X is None:
        # Fallback: Fibonacci
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Final projection and validation
    best_X = _project_to_sphere(best_X)
    
    # Check for near-duplicate points
    D = best_X[:, np.newaxis, :] - best_X[np.newaxis, :, :]
    dists = np.linalg.norm(D, axis=2)
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 1e-9:
        # Perturb close points
        for _ in range(100):
            min_dist = np.min(dists)
            if min_dist >= 1e-9:
                break
            # Find the closest pair
            idx = np.unravel_index(np.argmin(dists), dists.shape)
            i, j = idx
            # Move point i slightly
            noise = np.random.randn(3) * 1e-6
            best_X[i] = _project_to_sphere(best_X[i] + noise)
            # Recompute distances
            D = best_X[:, np.newaxis, :] - best_X[np.newaxis, :, :]
            dists = np.linalg.norm(D, axis=2)
            np.fill_diagonal(dists, np.inf)
    
    # Ensure all values are finite
    if not np.all(np.isfinite(best_X)):
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        best_X = _project_to_sphere(best_X)
    
    return best_X
