"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100
import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.optimize import differential_evolution

# EVOLVE-BLOCK-START

def _energy_and_grad(X):
    """
    Compute the logarithmic energy and its gradient for points on the unit sphere.
    X: (N, 3) array of points.
    Returns: energy (float), grad (N, 3) array.
    """
    N = X.shape[0]
    # Compute pairwise distances
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    D = np.sqrt(np.sum(diff**2, axis=2))
    
    # Avoid division by zero for i==j
    np.fill_diagonal(D, 1.0)
    
    # Energy: sum_{i<j} log(1/D[i,j]) = - sum_{i<j} log(D[i,j])
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    # Energy calculation
    log_D = np.log(D)
    energy = -np.sum(log_D[mask])
    
    # Gradient calculation
    inv_D2 = 1.0 / (D**2)
    np.fill_diagonal(inv_D2, 0.0)
    
    # grad[i] = - sum_j (X[i] - X[j]) / D[i,j]^2
    grad = -np.sum(diff * inv_D2[:, :, np.newaxis], axis=1)
    
    return energy, grad

def _objective_and_grad(x_flat):
    """
    Objective and gradient for flat vector input.
    """
    X = x_flat.reshape(N, 3)
    # Project to sphere
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / norms
    
    energy, grad = _energy_and_grad(X)
    
    # Project gradient to tangent space
    # grad_tangent = grad - (grad . X) * X
    grad_dot_X = np.sum(grad * X, axis=1, keepdims=True)
    grad_tangent = grad - grad_dot_X * X
    
    # Flatten for scipy
    grad_flat = grad_tangent.flatten()
    
    return energy, grad_flat

def _fibonacci_spiral():
    """Generate Fibonacci spiral initialization."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _random_points():
    """Generate random points on sphere."""
    # Use uniform distribution on sphere
    theta = np.random.uniform(0, 2 * np.pi, N)
    phi = np.arccos(2 * np.random.uniform(0, 1, N) - 1)
    X = np.column_stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ])
    return X

def _optimize_from_start(X0, time_budget):
    """
    Optimize from a starting point using L-BFGS-B.
    Returns best X found.
    """
    start_time = time.time()
    
    x0 = X0.flatten()
    
    # Use L-BFGS-B with proper options
    result = minimize(
        _objective_and_grad,
        x0,
        method='L-BFGS-B',
        jac=True,
        options={
            'maxiter': 2000,
            'ftol': 1e-15,
            'gtol': 1e-12,
            'maxls': 20,
            'maxfun': 5000
        }
    )
    
    X_opt = result.x.reshape(N, 3)
    X_opt = X_opt / np.linalg.norm(X_opt, axis=1, keepdims=True)
    
    # Check time
    elapsed = time.time() - start_time
    
    return X_opt, elapsed

def _simplex_to_points(simplex_params):
    """
    Convert parameters to N points on the sphere using angular coordinates.
    simplex_params: array of shape (2*N,) representing (theta, phi) for each point
    """
    theta = simplex_params[0:N]
    phi = simplex_params[N:2*N]
    
    # Convert to Cartesian
    X = np.column_stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ])
    
    # Ensure on unit sphere
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _de_objective_angular(x_flat):
    """
    Objective function for DE using angular parameters.
    x_flat: array of shape (2*N,) with theta in [-pi, pi], phi in [0, pi]
    """
    X = _simplex_to_points(x_flat)
    energy, _ = _energy_and_grad(X)
    return energy

def _run_de_angular(time_budget):
    """
    Run differential evolution on angular parameters (2N dimensions).
    """
    start_time = time.time()
    
    # theta: [-pi, pi], phi: [0, pi]
    bounds = [(-np.pi, np.pi)] * N + [(0, np.pi)] * N
    
    try:
        result = differential_evolution(
            _de_objective_angular,
            bounds,
            maxiter=30,
            popsize=20,
            tol=1e-3,
            seed=42,
            polish=False,
            init='latinhypercube',
            updating='deferred'
        )
        
        X_de = _simplex_to_points(result.x)
        
        if np.all(np.isfinite(X_de)):
            return [X_de]
        
    except Exception as e:
        pass
    
    return []

def construct():
    start_time = time.time()
    time_limit = 110.0  # Leave 10s margin for evaluation overhead
    
    best_X = None
    best_energy = np.inf
    
    # Multiple starting points
    starts = []
    
    # 1. Fibonacci spiral
    starts.append(_fibonacci_spiral())
    
    # 2-5. Random starts with different seeds
    for seed in [42, 123, 456, 789]:
        np.random.seed(seed)
        starts.append(_random_points())
    
    # 6. Slightly perturbed Fibonacci
    X_fib = _fibonacci_spiral()
    noise = np.random.normal(0, 0.1, (N, 3))
    X_perturbed = X_fib + noise
    X_perturbed = X_perturbed / np.linalg.norm(X_perturbed, axis=1, keepdims=True)
    starts.append(X_perturbed)
    
    # 7. Try differential evolution on angular parameters if we have time
    if time.time() - start_time < time_limit * 0.4:
        de_results = _run_de_angular(time_limit)
        starts.extend(de_results)
    
    # Try each start
    for i, X0 in enumerate(starts):
        if time.time() - start_time > time_limit:
            break
        
        remaining_time = time_limit - (time.time() - start_time)
        
        # Skip if not enough time for at least one optimization
        if remaining_time < 5.0:
            break
        
        try:
            X_opt, elapsed = _optimize_from_start(X0, remaining_time)
            
            # Check validity
            if not np.all(np.isfinite(X_opt)):
                continue
            
            # Check minimum distance
            diff = X_opt[:, np.newaxis, :] - X_opt[np.newaxis, :, :]
            D = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(D, np.inf)
            min_dist = np.min(D)
            
            if min_dist < 1e-9:
                continue
            
            # Compute energy
            energy, _ = _energy_and_grad(X_opt)
            
            if energy < best_energy:
                best_energy = energy
                best_X = X_opt.copy()
                
        except Exception as e:
            continue
    
    # If no valid result, return Fibonacci spiral as fallback
    if best_X is None:
        best_X = _fibonacci_spiral()
    
    return best_X

# EVOLVE-BLOCK-END
