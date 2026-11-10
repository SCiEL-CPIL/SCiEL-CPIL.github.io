import numpy as np
import math
import time
from scipy.optimize import minimize

N = 150

def _energy_fsum(points):
    """Compute logarithmic energy using math.fsum for precision (evaluator-compatible)."""
    n = points.shape[0]
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(n, k=1)
    terms = [-math.log(d) for d in dist[iu[0], iu[1]]]
    return math.fsum(terms)

def _grad_and_energy(points):
    """Compute log energy and Euclidean gradient for points on unit sphere (vectorized)."""
    n = points.shape[0]
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu[0], iu[1]]))
    
    # Gradient: E = -sum_{i<j} log(d_ij)
    # dE/dx_i = -sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    mask = ~np.eye(n, dtype=bool)
    coeff = np.zeros_like(dist_sq)
    coeff[mask] = -1.0 / dist_sq[mask]
    
    row_sums = coeff.sum(axis=1)
    grad = points * row_sums[:, np.newaxis] - coeff @ points
    
    return energy, grad

def _project_to_sphere(points):
    """Project points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return points / norms

def _spherical_to_cartesian(params):
    """Convert spherical coordinates (theta, phi) to Cartesian."""
    n = params.shape[0] // 2
    theta = params[0::2]
    phi = params[1::2]
    
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    return np.column_stack([x, y, z])

def _cartesian_to_spherical(X):
    """Convert Cartesian to spherical coordinates."""
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    phi = np.arctan2(y, x)
    phi = phi % (2 * np.pi)
    
    params = np.zeros(2 * X.shape[0])
    params[0::2] = theta
    params[1::2] = phi
    return params

def _objective_spherical(params):
    """Objective function for spherical coordinates."""
    X = _spherical_to_cartesian(params)
    energy, _ = _grad_and_energy(X)
    return energy

def _gradient_spherical(params):
    """Gradient of objective w.r.t. spherical coordinates."""
    n = params.shape[0] // 2
    theta = params[0::2]
    phi = params[1::2]
    
    X = _spherical_to_cartesian(params)
    energy, grad_euclidean = _grad_and_energy(X)
    
    # Project Euclidean gradient to tangent space
    dot = np.sum(grad_euclidean * X, axis=1)
    grad_tangent = grad_euclidean - dot[:, np.newaxis] * X
    
    # Basis vectors
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    cos_p = np.cos(phi)
    sin_p = np.sin(phi)
    
    dX_dtheta = np.column_stack([cos_t * cos_p, cos_t * sin_p, -sin_t])
    dX_dphi = np.column_stack([-sin_t * sin_p, sin_t * cos_p, np.zeros(n)])
    
    grad_theta = np.sum(grad_tangent * dX_dtheta, axis=1)
    grad_phi = np.sum(grad_tangent * dX_dphi, axis=1)
    
    grad_params = np.zeros(2 * n)
    grad_params[0::2] = grad_theta
    grad_params[1::2] = grad_phi
    
    return grad_params

def _fibonacci_spiral(n, offset=0.5):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(n, seed=None):
    """Generate random points on unit sphere."""
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    pts = rng.standard_normal((n, 3))
    return _project_to_sphere(pts)

def _backtracking_line_search(X, d, e, g, max_iter=25, c1=1e-4):
    """Backtracking line search on the sphere."""
    alpha = 1.0
    for _ in range(max_iter):
        X_new = X + alpha * d
        X_new = _project_to_sphere(X_new)
        e_new, g_new = _grad_and_energy(X_new)
        # Check Armijo condition
        if e_new <= e + c1 * alpha * np.sum(g * d):
            return X_new, e_new, g_new, alpha
        alpha *= 0.5
    return X, e, g, alpha

def _riemannian_cg(points, max_iter=500, tol=1e-12, deadline=None):
    """Riemannian conjugate gradient on the sphere with backtracking line search."""
    n = points.shape[0]
    X = points.copy()
    e, g = _grad_and_energy(X)
    
    def _tangent(v, x):
        dot = np.sum(v * x, axis=1, keepdims=True)
        return v - dot * x
    
    g_t = _tangent(g, X)
    
    if np.linalg.norm(g_t) < tol:
        return X, e
    
    d = -g_t
    g_old_t = g_t
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Backtracking line search
        X, e, g, alpha = _backtracking_line_search(X, d, e, g, max_iter=25)
        
        g_t = _tangent(g, X)
        if np.linalg.norm(g_t) < tol:
            break
        
        # Update direction
        denom = np.sum(g_old_t * g_old_t) + 1e-30
        beta = np.sum(g_t * g_t) / denom
        d = -g_t + beta * d
        d = _tangent(d, X)
        
        g_old_t = g_t
    
    return X, e

def _lbfgs_spherical(params_init, time_budget, max_iter=200, gtol=1e-8, ftol=1e-12):
    """Run L-BFGS-B on spherical coordinates with deadline guard."""
    def callback(params):
        if time.time() > time_budget:
            raise RuntimeError("time_limit")
        return None
    
    try:
        res = minimize(
            _objective_spherical,
            params_init,
            jac=_gradient_spherical,
            method='L-BFGS-B',
            bounds=[(0, np.pi)] * N + [(0, 2 * np.pi)] * N,
            options={
                'maxiter': max_iter,
                'maxls': 20,
                'gtol': gtol,
                'ftol': ftol,
                'maxfun': 500,
            },
            callback=callback,
        )
        X_opt = _spherical_to_cartesian(res.x)
        X_opt = _project_to_sphere(X_opt)
        return X_opt
    except (RuntimeError, Exception):
        return None

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s limit
    
    # Generate initial configurations
    inits = [_fibonacci_spiral(N)]
    for s in range(4):
        inits.append(_random_points(N, seed=42 + s))
    
    best_X = None
    best_energy = np.inf
    
    # Phase 1: L-BFGS in spherical coords on all initial configurations
    for init_idx, init_pts in enumerate(inits):
        if time.time() > deadline - 60.0:
            break
        
        params_init = _cartesian_to_spherical(init_pts)
        cand = _lbfgs_spherical(params_init, time_budget=deadline - 55.0, max_iter=150, gtol=1e-8, ftol=1e-10)
        
        if cand is not None:
            e_cand = _grad_and_energy(cand)[0]
            if e_cand < best_energy:
                best_energy = e_cand
                best_X = cand.copy()
    
    # If L-BFGS didn't produce anything, fall back to Fibonacci
    if best_X is None:
        best_X = _fibonacci_spiral(N)
        best_energy = _grad_and_energy(best_X)[0]
    
    # Phase 2: Riemannian CG polish from best solution
    if time.time() < deadline - 40.0:
        try:
            cand, _ = _riemannian_cg(best_X, max_iter=300, tol=1e-12, deadline=deadline - 35.0)
            e_cand = _grad_and_energy(cand)[0]
            if e_cand < best_energy:
                best_energy = e_cand
                best_X = cand.copy()
        except Exception:
            pass
    
    # Phase 3: Try a few more random restarts with L-BFGS + RCG if time permits
    rng = np.random.default_rng(123)
    n_extra_restarts = 5
    for r in range(n_extra_restarts):
        if time.time() > deadline - 25.0:
            break
        
        # Random restart
        if r % 2 == 0:
            init_pts = _random_points(N, seed=100 + r)
        else:
            # Perturb best solution
            init_pts = best_X + 0.3 * rng.standard_normal((N, 3))
            init_pts = _project_to_sphere(init_pts)
        
        params_init = _cartesian_to_spherical(init_pts)
        cand = _lbfgs_spherical(params_init, time_budget=deadline - 20.0, max_iter=100, gtol=1e-7, ftol=1e-10)
        
        if cand is not None:
            # Quick RCG polish
            try:
                cand, _ = _riemannian_cg(cand, max_iter=100, tol=1e-10, deadline=deadline - 15.0)
            except Exception:
                pass
            e_cand = _grad_and_energy(cand)[0]
            if e_cand < best_energy:
                best_energy = e_cand
                best_X = cand.copy()
    
    # Phase 4: Final tight Riemannian CG polish
    if time.time() < deadline - 10.0:
        try:
            cand, _ = _riemannian_cg(best_X, max_iter=300, tol=1e-14, deadline=deadline - 5.0)
            e_cand = _grad_and_energy(cand)[0]
            if e_cand < best_energy:
                best_energy = e_cand
                best_X = cand.copy()
        except Exception:
            pass
    
    # Phase 5: Final L-BFGS polish with very tight tolerances
    if time.time() < deadline - 5.0:
        params_best = _cartesian_to_spherical(best_X)
        cand = _lbfgs_spherical(params_best, time_budget=deadline - 3.0, max_iter=100, gtol=1e-12, ftol=1e-14)
        if cand is not None:
            e_cand = _grad_and_energy(cand)[0]
            if e_cand < best_energy:
                best_energy = e_cand
                best_X = cand.copy()
    
    # Final validation and projection
    best_X = _project_to_sphere(best_X)
    if not np.all(np.isfinite(best_X)):
        best_X = _fibonacci_spiral(N)
    
    # Check for near-duplicate points
    dists = np.sqrt(np.sum((best_X[:, np.newaxis, :] - best_X[np.newaxis, :, :]) ** 2, axis=2))
    if np.min(dists[~np.eye(N, dtype=bool)]) < 1e-9:
        # Nudge the closest pair apart
        min_i, min_j = np.unravel_index(np.argmin(dists[~np.eye(N, dtype=bool)]), (N, N))
        if min_i > min_j:
            min_i, min_j = min_j, min_i
        # Move point j slightly
        direction = best_X[min_i] - best_X[min_j]
        direction = direction / (np.linalg.norm(direction) + 1e-15)
        best_X[min_j] = best_X[min_j] - 1e-8 * direction
        best_X = _project_to_sphere(best_X)
    
    return best_X
