"""Optimized logarithmic energy minimizer for 150 points on the unit sphere."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150
DEADLINE = 120.0
SAFETY_MARGIN = 5.0


def _energy_and_grad(x):
    """Compute logarithmic energy and gradient for points x of shape (N, 3).
    
    Energy = sum_{i<j} log(1 / ||x_i - x_j||) = -sum_{i<j} log(||x_i - x_j||)
    Gradient:
    dE/dx_k = sum_{j != k} (x_j - x_k) / ||x_j - x_k||^2
    """
    # Ensure x is on the sphere for gradient correctness, though the optimizer handles projection
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]  # (N, N, 3)
    dist_sq = np.sum(diff * diff, axis=2)  # (N, N)
    
    # Avoid division by zero
    dist_sq_safe = np.maximum(dist_sq, 1e-30)
    dist = np.sqrt(dist_sq_safe)
    
    # Energy: sum over i<j of log(1/dist) = -sum over i<j of log(dist)
    iu = np.triu_indices(N, k=1)
    log_dists = np.log(dist[iu])
    energy = -np.sum(log_dists)
    
    # Gradient computation
    # We need sum_{j != i} (x_j - x_i) / dist(i,j)^2
    # Note: (x_j - x_i) = -diff[i, j]
    # So term is -diff[i, j] / dist_sq[i, j]
    
    # Create matrix of weights: 1/dist_sq for i != j, 0 otherwise
    inv_dist_sq = np.zeros_like(dist_sq_safe)
    inv_dist_sq[iu] = 1.0 / dist_sq_safe[iu]
    inv_dist_sq[(iu[1], iu[0])] = 1.0 / dist_sq_safe[iu] # Symmetric
    
    # Gradient for point i is sum_j (x_j - x_i) / dist_sq(i,j)
    # = sum_j x_j / dist_sq(i,j) - x_i * sum_j 1/dist_sq(i,j)
    
    # Term 1: sum_j x_j * w_ij
    grad_term1 = inv_dist_sq @ x
    
    # Term 2: x_i * sum_j w_ij
    row_sums = np.sum(inv_dist_sq, axis=1)
    grad_term2 = x * row_sums[:, np.newaxis]
    
    grad = grad_term1 - grad_term2
    
    return energy, grad


def _log_energy_fast(x):
    """Fast energy computation only (no gradient)."""
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    dist_sq = np.maximum(dist_sq, 1e-30)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(N, k=1)
    return -np.sum(np.log(dist[iu]))


def _fibonacci_sphere(n):
    """Generate n points using Fibonacci spiral on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _halton_sphere(n):
    """Generate n points using Halton sequence on unit sphere."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r
    
    points = []
    for i in range(1, n + 1):
        p = [halton(i, b) for b in [2, 3, 5]]
        theta = 2 * np.pi * p[0]
        phi = np.arccos(2 * p[1] - 1)
        pt = [np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)]
        norm = np.linalg.norm(pt)
        points.append(pt / norm)
    return np.array(points)


def _sobol_sphere(n):
    """Generate n points using Sobol sequence on unit sphere."""
    from scipy.stats import qmc
    sampler = qmc.Sobol(d=3, scramble=True)
    samples = sampler.random(n)
    # Map [0,1]^3 to sphere
    # Use spherical coordinates
    u1, u2, u3 = samples[:, 0], samples[:, 1], samples[:, 2]
    theta = 2 * np.pi * u1
    phi = np.arccos(2 * u2 - 1)
    points = np.column_stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ])
    return points


def _normalize(x):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return x / norms


def _riemannian_lbfgs(x0, deadline):
    """Run L-BFGS-B optimization with deadline guard."""
    try:
        result = minimize(
            _energy_and_grad,
            x0.flatten(),
            jac=True,
            method='L-BFGS-B',
            options={
                'maxiter': 5000,
                'gtol': 1e-12,
                'ftol': 1e-14,
                'disp': False,
                'maxls': 50
            }
        )
        x_opt = _normalize(result.x.reshape(N, 3))
        return x_opt
    except Exception:
        return x0


def _polish(x0, deadline):
    """High-precision polish pass with very tight tolerances."""
    remaining = deadline - time.time()
    if remaining < 5.0:
        return x0
    
    try:
        result = minimize(
            _energy_and_grad,
            x0.flatten(),
            jac=True,
            method='L-BFGS-B',
            options={
                'maxiter': 10000,
                'gtol': 1e-15,
                'ftol': 1e-16,
                'disp': False,
                'maxls': 100
            }
        )
        x_opt = _normalize(result.x.reshape(N, 3))
        
        # Validate
        if not np.all(np.isfinite(x_opt)):
            return x0
        if np.any(np.linalg.norm(x_opt, axis=1) < 1e-15):
            return x0
        
        # Check minimum distance
        diff = x_opt[:, np.newaxis, :] - x_opt[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        min_dist_sq = np.min(dist_sq)
        if min_dist_sq < (1e-9) ** 2:
            return x0
        
        return x_opt
    except Exception:
        return x0


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    best_x = None
    best_energy = float('inf')
    
    # Generate multiple initializations
    initializations = [
        ("fibonacci", _fibonacci_sphere(N)),
        ("halton", _halton_sphere(N)),
        ("sobol", _sobol_sphere(N)),
    ]
    
    # Add random initializations
    np.random.seed(42)
    for _ in range(5):
        x_rand = np.random.randn(N, 3)
        x_rand = x_rand / np.linalg.norm(x_rand, axis=1, keepdims=True)
        initializations.append(("random", x_rand))
    
    # Sort initializations by initial energy to prioritize good starts
    init_energies = []
    for name, x0 in initializations:
        x0 = _normalize(x0)
        e = _log_energy_fast(x0)
        init_energies.append((e, name, x0))
    
    init_energies.sort(key=lambda x: x[0])
    
    for e_init, name, x0 in init_energies:
        if time.time() > deadline - 10.0:
            break
        
        # Strategy 1: L-BFGS
        if time.time() < deadline - 15.0:
            x_opt = _riemannian_lbfgs(x0, deadline)
            if np.all(np.isfinite(x_opt)) and np.min(np.linalg.norm(x_opt, axis=1)) > 1e-15:
                energy = _log_energy_fast(x_opt)
                # Check min distance
                diff = x_opt[:, np.newaxis, :] - x_opt[np.newaxis, :, :]
                dist_sq = np.sum(diff * diff, axis=2)
                np.fill_diagonal(dist_sq, np.inf)
                if np.min(dist_sq) >= (1e-9) ** 2:
                    if energy < best_energy:
                        best_energy = energy
                        best_x = x_opt.copy()
        
        # If we have a very good result, we can stop early
        if best_energy < -2364.632242592:
            break

    # Final polish on the best result
    if best_x is not None and time.time() < deadline - 3.0:
        x_polished = _polish(best_x, deadline)
        if np.all(np.isfinite(x_polished)):
            energy_polished = _log_energy_fast(x_polished)
            diff = x_polished[:, np.newaxis, :] - x_polished[np.newaxis, :, :]
            dist_sq = np.sum(diff * diff, axis=2)
            np.fill_diagonal(dist_sq, np.inf)
            if np.min(dist_sq) >= (1e-9) ** 2 and energy_polished < best_energy:
                best_energy = energy_polished
                best_x = x_polished.copy()
    
    if best_x is None:
        best_x = _fibonacci_sphere(N)
    
    return _normalize(best_x)
