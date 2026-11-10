"""
Different strategy: Use scipy.optimize.minimize with L-BFGS-B on a parameterized
representation (spherical coordinates with clamping), combined with a multi-start
approach using diverse initializations and a final high-precision polish pass.
This avoids the complexity of Riemannian geometry and uses a robust, well-tested
optimizer from SciPy.
"""
import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

N = 100

# EVOLVE-BLOCK-START

def _energy_from_params(params):
    """
    Compute logarithmic energy from spherical coordinate parameters.
    params: (N, 2) array of (theta, phi) where theta in [-pi/2, pi/2], phi in [0, 2pi)
    """
    theta = params[:, 0]
    phi = params[:, 1]
    
    # Convert to Cartesian
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    x = cos_theta * np.cos(phi)
    y = cos_theta * np.sin(phi)
    z = sin_theta
    pts = np.column_stack([x, y, z])
    
    # Normalize to ensure unit sphere (should be already, but for safety)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    
    # Compute pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    # Energy: sum_{i<j} log(1/d) = -sum_{i<j} log(d)
    mask = ~np.eye(N, dtype=bool)
    log_dists = -np.log(dist)
    np.fill_diagonal(log_dists, 0.0)
    energy = 0.5 * np.sum(log_dists)
    
    return energy

def _energy_and_grad_from_params(params):
    """
    Compute logarithmic energy and gradient with respect to (theta, phi) parameters.
    Uses finite differences for gradient if analytical is complex, but let's try analytical.
    Actually, let's use scipy's numerical gradient or compute it properly.
    
    For simplicity and robustness, we'll use a function that returns just energy,
    and let L-BFGS-B use numerical gradients if needed, or provide analytical.
    Let's provide analytical gradient for speed.
    """
    theta = params[:, 0]
    phi = params[:, 1]
    
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    x = cos_theta * np.cos(phi)
    y = cos_theta * np.sin(phi)
    z = sin_theta
    pts = np.column_stack([x, y, z])
    
    # Normalize
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    # Energy
    mask = ~np.eye(N, dtype=bool)
    log_dists = -np.log(dist)
    np.fill_diagonal(log_dists, 0.0)
    energy = 0.5 * np.sum(log_dists)
    
    # Gradient computation
    # We need dE/dtheta_i and dE/dphi_i
    # E = -sum_{j != i} log(||x_i - x_j||)
    # dE/dx_i = sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
    # Then dE/dtheta_i = (dE/dx_i) dot (dx_i/dtheta_i)
    # dE/dphi_i = (dE/dx_i) dot (dx_i/dphi_i)
    
    # Compute dE/dx_i
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    grad_x = np.sum(diff * inv_dist_sq[..., np.newaxis], axis=1)
    
    # dx/dtheta = (-sin_theta * cos_phi, -sin_theta * sin_phi, cos_theta)
    dx_dtheta = np.column_stack([
        -sin_theta * np.cos(phi),
        -sin_theta * np.sin(phi),
        cos_theta
    ])
    
    # dx/dphi = (-cos_theta * sin_phi, cos_theta * cos_phi, 0)
    dx_dphi = np.column_stack([
        -cos_theta * np.sin(phi),
        cos_theta * np.cos(phi),
        np.zeros(N)
    ])
    
    grad_theta = np.sum(grad_x * dx_dtheta, axis=1)
    grad_phi = np.sum(grad_x * dx_dphi, axis=1)
    
    grad = np.column_stack([grad_theta, grad_phi])
    
    return energy, grad

def _optimize_from_start(params_init, time_limit=20.0):
    """
    Run L-BFGS-B optimization from a given starting point.
    """
    start_time = time.time()
    
    # Bounds for theta and phi
    bounds = [(-math.pi/2 + 1e-6, math.pi/2 - 1e-6), (0, 2*math.pi)] * N
    
    result = minimize(
        _energy_from_params,
        params_init,
        jac=_energy_and_grad_from_params,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
    )
    
    if time.time() - start_time > time_limit:
        return params_init, _energy_from_params(params_init)
    
    return result.x, result.fun

def _fibonacci_sphere(n):
    """Generate Fibonacci lattice points on sphere, return (theta, phi)."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    theta = np.arcsin(z)
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    phi = phi % (2 * math.pi)
    return np.column_stack([theta, phi])

def _sobol_sphere(n, seed=0):
    """Generate points using Sobol sequence mapped to sphere."""
    sampler = qmc.Sobol(d=2, seed=seed)
    m = 10
    while (1 << m) < n:
        m += 1
    samples = sampler.random(1 << m)
    samples = samples[:n]
    
    u1 = samples[:, 0]
    u2 = samples[:, 1]
    
    theta = (2.0 * u1 - 1.0) * math.pi / 2
    phi = 2.0 * math.pi * u2
    
    return np.column_stack([theta, phi])

def _random_sphere(n, seed=0):
    """Generate random points on sphere, return (theta, phi)."""
    rng = np.random.RandomState(seed)
    theta = rng.uniform(-math.pi/2, math.pi/2, n)
    phi = rng.uniform(0, 2*math.pi, n)
    return np.column_stack([theta, phi])

def _perturbed_fibonacci(n, seed=0, scale=0.1):
    """Perturb Fibonacci points."""
    base = _fibonacci_sphere(n)
    rng = np.random.RandomState(seed)
    perturbation = rng.randn(n, 2) * scale
    perturbed = base + perturbation
    # Wrap phi
    perturbed[:, 1] = perturbed[:, 1] % (2 * math.pi)
    # Clamp theta
    perturbed[:, 0] = np.clip(perturbed[:, 0], -math.pi/2 + 1e-6, math.pi/2 - 1e-6)
    return perturbed

def _params_to_sphere(params):
    """Convert (theta, phi) parameters to unit sphere points."""
    theta = params[:, 0]
    phi = params[:, 1]
    
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    x = cos_theta * np.cos(phi)
    y = cos_theta * np.sin(phi)
    z = sin_theta
    pts = np.column_stack([x, y, z])
    
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_params = None
    best_energy = float('inf')
    
    # Multi-start strategy with diverse initializations
    initializers = [
        lambda: _fibonacci_sphere(N),
        lambda: _sobol_sphere(N, seed=0),
        lambda: _random_sphere(N, seed=1),
        lambda: _random_sphere(N, seed=2),
        lambda: _random_sphere(N, seed=3),
        lambda: _perturbed_fibonacci(N, seed=1, scale=0.05),
        lambda: _perturbed_fibonacci(N, seed=2, scale=0.1),
        lambda: _perturbed_fibonacci(N, seed=3, scale=0.2),
    ]
    
    # Allocate time: give more time to better initializers
    # Fibonacci and Sobol are good starting points
    time_allocations = [30, 25, 15, 15, 15, 10, 10, 10]
    
    for i, (init_func, time_alloc) in enumerate(zip(initializers, time_allocations)):
        if time.time() > deadline - 20.0:
            break
        
        try:
            params_init = init_func()
            
            # Check if we have enough time
            remaining = deadline - time.time() - 10.0  # Reserve 10s for final polish
            if remaining < 5.0:
                break
            
            actual_time = min(time_alloc, remaining)
            
            params_opt, energy = _optimize_from_start(params_init, time_limit=actual_time)
            
            if energy < best_energy:
                best_energy = energy
                best_params = params_opt.copy()
                
        except Exception as e:
            continue
    
    # If we found a good solution, do a final high-precision polish
    if best_params is not None and time.time() < deadline - 5.0:
        try:
            # Polish with tighter tolerances
            bounds = [(-math.pi/2 + 1e-8, math.pi/2 - 1e-8), (0, 2*math.pi)] * N
            result = minimize(
                _energy_from_params,
                best_params,
                jac=_energy_and_grad_from_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-18, 'gtol': 1e-15}
            )
            
            if result.fun < best_energy:
                best_energy = result.fun
                best_params = result.x.copy()
                
        except Exception:
            pass
    
    if best_params is None:
        # Fallback: Fibonacci
        best_params = _fibonacci_sphere(N)
    
    # Convert to sphere points
    pts = _params_to_sphere(best_params)
    
    # Final check: ensure no points are too close
    # If any points are closer than 1e-9, this would be invalid
    # But our optimizer should avoid this
    
    return pts

# EVOLVE-BLOCK-END
