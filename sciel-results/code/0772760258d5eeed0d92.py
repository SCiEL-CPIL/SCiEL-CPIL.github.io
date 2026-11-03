"""Rewritten program: multi-restart L-BFGS-B with gradient flow, Sobol/Fibonacci/random
initializations, deadline guard, and high-precision final polish for logarithmic energy
minimization of 100 points on the unit sphere."""
import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

N = 100
DEADLINE_SECONDS = 145.0  # safety margin under 150s wall-clock


def _energy_and_grad(X):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i-x_j|) and its Euclidean gradient.
    X is (N,3) on the unit sphere. Returns (energy, grad)."""
    diff = X[:, None, :] - X[None, :, :]  # (N, N, 3)
    dist = np.linalg.norm(diff, axis=2)   # (N, N)
    np.fill_diagonal(dist, 1.0)
    
    iu = np.triu_indices(N, k=1)
    d_ij = dist[iu]
    e = -np.sum(np.log(d_ij))
    
    inv_dist2 = np.zeros_like(dist)
    mask = ~np.eye(N, dtype=bool)
    inv_dist2[mask] = 1.0 / (dist[mask] ** 2)
    
    row_sum = np.sum(inv_dist2, axis=1)
    weighted_sum = inv_dist2 @ X
    grad = X * row_sum[:, None] - weighted_sum
    
    return e, grad


def _project_to_sphere(X):
    """Normalize each row of X to unit length."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return X / norms


def _fibonacci_sphere(n, seed_offset=0.0):
    """Generate n points on the unit sphere using Fibonacci spiral with offset."""
    i = np.arange(n) + 0.5 + seed_offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _sobol_sphere(n, seed=0):
    """Generate n points on the unit sphere using a Sobol quasi-random sequence."""
    sampler = qmc.Sobol(d=3, seed=seed, scramble=True)
    points = sampler.random(2048)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    points = points / norms
    return points[-n:]


def _random_sphere(n, rng):
    """Generate n random points on the unit sphere."""
    v = rng.normal(size=(n, 3))
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return v / norms


def _gradient_flow(X, n_steps=500, lr=0.01, cooling=True, deadline=None):
    """Run gradient flow on the sphere with optional annealing."""
    X = X.copy()
    current_lr = lr
    
    for step in range(n_steps):
        if deadline is not None and time.time() > deadline:
            break
        
        _, grad = _energy_and_grad(X)
        
        grad_dot = np.sum(grad * X, axis=1, keepdims=True)
        grad_tangent = grad - grad_dot * X
        
        if cooling:
            current_lr = lr * (1.0 - step / n_steps) + lr * 0.1 * (step / n_steps)
        
        X = X - current_lr * grad_tangent
        X = _project_to_sphere(X)
    
    e, _ = _energy_and_grad(X)
    return X, e


def _objective(X_flat):
    """Objective for scipy minimize."""
    X = X_flat.reshape(N, 3)
    X = _project_to_sphere(X)
    e, _ = _energy_and_grad(X)
    return e


def _objective_grad(X_flat):
    """Objective and gradient for scipy minimize."""
    X = X_flat.reshape(N, 3)
    X = _project_to_sphere(X)
    e, grad = _energy_and_grad(X)
    return e, grad


def construct():
    start_time = time.time()
    
    best_X = None
    best_energy = float('inf')
    
    rng = np.random.default_rng(42)
    
    # Generate multiple initial configurations
    initial_configs = [_fibonacci_sphere(N)]
    
    # Random perturbations of Fibonacci
    for scale in [0.05, 0.1, 0.15, 0.2]:
        X0 = _fibonacci_sphere(N)
        noise = rng.normal(scale=scale, size=X0.shape)
        X0 = _project_to_sphere(X0 + noise)
        initial_configs.append(X0)
    
    # Fully random configurations
    for _ in range(4):
        initial_configs.append(_random_sphere(N, rng))
    
    # Scaled Fibonacci with different offsets
    for offset in [0.05, 0.1, 0.2, 0.3, 0.5]:
        initial_configs.append(_fibonacci_sphere(N, seed_offset=offset))
    
    # Sobol sequence initializations
    for sobol_seed in [0, 1, 2, 3, 4]:
        initial_configs.append(_sobol_sphere(N, seed=sobol_seed))
    
    # Time allocation
    deadline_init = start_time + DEADLINE_SECONDS * 0.55
    deadline_flow = start_time + DEADLINE_SECONDS * 0.80
    deadline_polish = start_time + DEADLINE_SECONDS - 3.0
    
    # Phase 1: L-BFGS optimization from each initial config
    for X0 in initial_configs:
        if time.time() > deadline_init:
            break
        
        x0_flat = X0.flatten()
        
        try:
            res = minimize(
                _objective_grad,
                x0_flat,
                method='L-BFGS-B',
                options={
                    'maxiter': 3000,
                    'ftol': 1e-14,
                    'gtol': 1e-12,
                    'maxfun': 8000,
                }
            )
        except Exception:
            continue
        
        X_opt = _project_to_sphere(res.x.reshape(N, 3))
        e_opt, _ = _energy_and_grad(X_opt)
        
        if e_opt < best_energy:
            best_energy = e_opt
            best_X = X_opt.copy()
    
    # Phase 2: Gradient flow with annealing on best solution
    if best_X is not None and time.time() < deadline_flow:
        try:
            X_flow, e_flow = _gradient_flow(
                best_X, 
                n_steps=500, 
                lr=0.005, 
                cooling=True,
                deadline=deadline_flow
            )
            if e_flow < best_energy:
                best_energy = e_flow
                best_X = X_flow.copy()
        except Exception:
            pass
        
        # Second gradient flow with different parameters
        if best_X is not None and time.time() < deadline_flow:
            try:
                X_flow2, e_flow2 = _gradient_flow(
                    best_X, 
                    n_steps=300, 
                    lr=0.002, 
                    cooling=True,
                    deadline=deadline_flow
                )
                if e_flow2 < best_energy:
                    best_energy = e_flow2
                    best_X = X_flow2.copy()
            except Exception:
                pass
    
    # Phase 3: Final high-precision polish
    if best_X is not None and time.time() < deadline_polish:
        try:
            x0_flat = best_X.flatten()
            res = minimize(
                _objective_grad,
                x0_flat,
                method='L-BFGS-B',
                options={
                    'maxiter': 10000,
                    'ftol': 1e-16,
                    'gtol': 1e-15,
                    'maxfun': 20000,
                }
            )
            X_polish = _project_to_sphere(res.x.reshape(N, 3))
            e_polish, _ = _energy_and_grad(X_polish)
            if e_polish < best_energy:
                best_energy = e_polish
                best_X = X_polish.copy()
        except Exception:
            pass
    
    if best_X is None:
        best_X = _fibonacci_sphere(N)
    
    return best_X
