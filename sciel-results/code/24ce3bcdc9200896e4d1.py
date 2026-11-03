"""Rewritten program for logarithmic energy minimization of 150 points on the sphere."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize


def _fibonacci_sphere(n):
    """Generate n points on the unit sphere using the golden-angle spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_uniform_sphere(n, seed=None):
    """Generate n uniformly random points on the unit sphere."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    x = rng.randn(n, 3)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def _project_to_sphere(points):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _energy_and_grad(flat_x):
    """Compute energy and gradient for optimization.
    flat_x is a 1D array of shape (3*N,) representing N points on sphere.
    Returns (energy, gradient) where gradient is with respect to flat_x.
    """
    x = flat_x.reshape(N, 3)
    
    # Compute pairwise distances
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    dists = np.sqrt(dists_sq)
    
    # Avoid log(0) on diagonal
    np.fill_diagonal(dists, 1.0)
    
    # Energy: sum_{i<j} log(1/d) = -sum_{i<j} log(d)
    iu = np.triu_indices(N, k=1)
    log_dists = np.log(dists[iu[0], iu[1]])
    energy = -math.fsum(log_dists.tolist())
    
    # Gradient: dE/dx_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    # For the log energy: dE/dx_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    inv_d2 = 1.0 / (dists_sq)
    np.fill_diagonal(inv_d2, 0.0)
    grad = np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)
    
    return energy, grad.flatten()


def _energy(x):
    """Compute just the energy for a set of points."""
    x = x.reshape(N, 3)
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    dists = np.sqrt(dists_sq)
    np.fill_diagonal(dists, 1.0)
    iu = np.triu_indices(N, k=1)
    log_dists = np.log(dists[iu[0], iu[1]])
    return -math.fsum(log_dists.tolist())


def _tangent_project(grad, x):
    """Project gradient to tangent space at each point."""
    x = x.reshape(N, 3)
    grad = grad.reshape(N, 3)
    radial = np.sum(grad * x, axis=1, keepdims=True)
    return grad - radial * x


def _lbfgs_optimize(x0, maxiter=200, ftol=1e-14, gtol=1e-12, deadline=None):
    """Run L-BFGS-B optimization on the sphere using projection."""
    
    def objective_and_grad(flat_x):
        x = _project_to_sphere(flat_x.reshape(N, 3))
        e, g = _energy_and_grad(x.flatten())
        grad_t = _tangent_project(g, x)
        return e, grad_t.flatten()
    
    try:
        result = minimize(
            objective_and_grad,
            x0.flatten(),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol, 'maxls': 30}
        )
        x_opt = _project_to_sphere(result.x.reshape(N, 3))
        e_opt = _energy(x_opt)
        return x_opt, e_opt
    except Exception:
        return x0, _energy(x0)


def _gradient_descent_polish(x, max_iter=5000, tol=1e-15, deadline=None):
    """Polish with gradient descent on the sphere using adaptive step size."""
    x = _project_to_sphere(x.copy())
    
    best_x = x.copy()
    best_e = _energy(x)
    
    e, g = _energy_and_grad(x.flatten())
    grad_t = _tangent_project(g, x)
    grad_norm = np.linalg.norm(grad_t)
    
    # Adaptive step size based on gradient norm
    alpha = 0.5 / max(grad_norm, 1e-10)
    alpha = min(alpha, 0.1)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Try step
        x_trial = _project_to_sphere(x + alpha * grad_t)
        e_trial, g_trial = _energy_and_grad(x_trial.flatten())
        grad_t_trial = _tangent_project(g_trial, x_trial)
        
        if e_trial < best_e - 1e-16:
            # Accept step
            x = x_trial
            best_x = x.copy()
            best_e = e_trial
            grad_t = grad_t_trial
            # Slightly increase step size if we made good progress
            if it % 5 == 0:
                alpha = min(alpha * 1.05, 0.2)
        else:
            # Decrease step size
            alpha *= 0.5
            if alpha < 1e-16:
                break
        
        # Check convergence
        grad_norm = np.linalg.norm(grad_t)
        if grad_norm < tol:
            break
    
    return best_x, best_e


def _conjugate_gradient(x, max_iter=3000, tol=1e-14, deadline=None):
    """Conjugate gradient method on the sphere."""
    x = _project_to_sphere(x.copy())
    best_x = x.copy()
    best_e = _energy(x)
    
    e, g = _energy_and_grad(x.flatten())
    grad_t = _tangent_project(g, x)
    
    d = -grad_t.copy()
    alpha = 0.01
    beta = 0.0
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Line search
        for _ in range(30):
            x_trial = _project_to_sphere(x + alpha * d)
            e_trial, g_trial = _energy_and_grad(x_trial.flatten())
            grad_t_trial = _tangent_project(g_trial, x_trial)
            
            if e_trial < best_e - 1e-16:
                # Accept
                x = x_trial
                best_x = x.copy()
                best_e = e_trial
                grad_t = grad_t_trial
                break
            else:
                alpha *= 0.5
                if alpha < 1e-16:
                    break
        else:
            break
        
        # Update direction
        grad_norm_sq = np.sum(grad_t * grad_t)
        if grad_norm_sq < tol * tol:
            break
        
        if beta > 0:
            d = -grad_t + beta * d
            # Project d to tangent space
            radial_d = np.sum(d * x, axis=1, keepdims=True)
            d = d - radial_d * x
        else:
            d = -grad_t
            # Project to tangent space
            radial_d = np.sum(d * x, axis=1, keepdims=True)
            d = d - radial_d * x
        
        # Polak-Ribiere beta
        e_new, g_new = _energy_and_grad(x.flatten())
        grad_t_new = _tangent_project(g_new, x)
        dot = np.sum(grad_t_new * grad_t)
        if grad_norm_sq > 1e-30:
            beta = max(0.0, dot / grad_norm_sq)
        else:
            beta = 0.0
        grad_t = grad_t_new
    
    return best_x, best_e


def _basin_hopping(x0, n_restarts=5, max_iter=500, deadline=None):
    """Basin hopping: random perturbations followed by local minimization."""
    x = _project_to_sphere(x0.copy())
    best_x = x.copy()
    best_e = _energy(x)
    
    rng = np.random.RandomState(42)
    
    for restart in range(n_restarts):
        if deadline is not None and time.time() > deadline:
            break
        
        # Random perturbation
        perturbation = rng.randn(N, 3) * 0.1
        x_trial = _project_to_sphere(x + perturbation)
        
        # Local minimization
        x_opt, e_opt = _lbfgs_optimize(x_trial, maxiter=100, ftol=1e-12, gtol=1e-10, deadline=deadline)
        if e_opt < best_e:
            best_x = x_opt
            best_e = e_opt
    
    return best_x, best_e


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    # Multiple initializations
    initializations = []
    
    # 1. Fibonacci spiral
    x_fib = _fibonacci_sphere(N)
    x_fib = _project_to_sphere(x_fib)
    initializations.append(x_fib)
    
    # 2. Random with different seeds
    for seed in [1, 2, 3, 4, 5]:
        x_rand = _random_uniform_sphere(N, seed=seed)
        initializations.append(x_rand)
    
    best_x = initializations[0].copy()
    best_e = _energy(best_x)
    
    # Optimize each initialization
    for idx, x_init in enumerate(initializations):
        if time.time() > deadline - 30:
            break
        
        # L-BFGS-B optimization
        try:
            x_opt, e_opt = _lbfgs_optimize(x_init, maxiter=150, ftol=1e-13, gtol=1e-11, deadline=deadline - 20)
            if e_opt < best_e:
                best_x = x_opt
                best_e = e_opt
        except Exception:
            pass
        
        # Conjugate gradient polish
        if time.time() > deadline - 20:
            break
        try:
            x_cg, e_cg = _conjugate_gradient(best_x, max_iter=2000, tol=1e-14, deadline=deadline - 10)
            if e_cg < best_e:
                best_x = x_cg
                best_e = e_cg
        except Exception:
            pass
    
    # Basin hopping with the best configuration
    if time.time() < deadline - 30:
        try:
            x_bh, e_bh = _basin_hopping(best_x, n_restarts=3, max_iter=300, deadline=deadline - 15)
            if e_bh < best_e:
                best_x = x_bh
                best_e = e_bh
        except Exception:
            pass
    
    # Final tight polish
    if time.time() < deadline - 10:
        try:
            x_final, e_final = _gradient_descent_polish(best_x, max_iter=8000, tol=1e-16, deadline=deadline - 2)
            if e_final < best_e:
                best_x = x_final
                best_e = e_final
        except Exception:
            pass
    
    # Another CG pass if time remains
    if time.time() < deadline - 5:
        try:
            x_cg2, e_cg2 = _conjugate_gradient(best_x, max_iter=3000, tol=1e-15, deadline=deadline - 1)
            if e_cg2 < best_e:
                best_x = x_cg2
                best_e = e_cg2
        except Exception:
            pass
    
    # Final projection and validation
    best_x = _project_to_sphere(best_x)
    
    # Ensure no points are too close
    min_dist = 1e-9
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_x[i] - best_x[j])
            if d < min_dist:
                rng = np.random.RandomState(12345 + i * 100 + j)
                best_x[j] += 1e-10 * rng.randn(3)
                best_x = _project_to_sphere(best_x)
    
    # Final check for non-finite values
    if not np.all(np.isfinite(best_x)):
        best_x = _fibonacci_sphere(N)
        best_x = _project_to_sphere(best_x)
    
    return best_x
# EVOLVE-BLOCK-END
