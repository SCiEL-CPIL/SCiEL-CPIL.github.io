"""Optimized logarithmic energy minimization for 100 points on the unit sphere."""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
import time
import math

N = 100
DEADLINE = 148.0  # Safety margin before 150s kill


def _energy_and_grad(points):
    """Compute logarithmic energy and gradient for points on unit sphere.
    
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Using chordal distance |x_i - x_j| = sqrt(2 - 2*x_i·x_j)
    """
    n = points.shape[0]
    # Compute pairwise dot products
    dots = points @ points.T
    # Clip to valid range for numerical stability
    dots = np.clip(dots, -1.0, 1.0)
    # Chordal distances squared: |xi - xj|^2 = 2 - 2*dot
    d2 = 2.0 - 2.0 * dots
    # Avoid log(0) on diagonal
    d2_flat = d2.ravel()
    d2_flat[::n+1] = 1.0  # set diagonal to 1.0
    
    # log(1/d) = -log(d) = -0.5*log(d2)
    log_d2 = np.log(d2)
    # Energy = sum_{i<j} -0.5 * log(d2_ij)
    # Sum over upper triangle using mask
    iu = np.triu_indices(n, k=1)
    energy = -0.5 * np.sum(log_d2[iu])
    
    # Gradient computation:
    # grad_i = sum_{j!=i} (x_i - x_j) / d2_ij
    # = x_i * sum_{j!=i} 1/d2_ij - sum_{j!=i} x_j / d2_ij
    # Let w_ij = 1/d2_ij for i!=j, 0 for i=j
    # grad_i = x_i * row_sum_i - (W @ points)_i
    # where W[i,j] = 1/d2_ij if i!=j else 0
    
    W = 1.0 / d2
    W_flat = W.ravel()
    W_flat[::n+1] = 0.0  # zero diagonal
    
    row_sums = W.sum(axis=1, keepdims=True)  # shape (n, 1)
    grad = points * row_sums - W @ points
    
    # Project gradient onto tangent space: grad_i -= (grad_i · x_i) * x_i
    radial = np.sum(grad * points, axis=1, keepdims=True)
    grad = grad - radial * points
    
    return energy, grad


def _energy_only(points):
    """Just the energy value (no gradient)."""
    n = points.shape[0]
    dots = points @ points.T
    dots = np.clip(dots, -1.0, 1.0)
    d2 = 2.0 - 2.0 * dots
    d2_flat = d2.ravel()
    d2_flat[::n+1] = 1.0  # set diagonal to 1.0
    log_d2 = np.log(d2)
    iu = np.triu_indices(n, k=1)
    energy = -0.5 * np.sum(log_d2[iu])
    return energy


def _normalize(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _riemannian_gradient_descent(points, max_iter=2000, tol=1e-14, deadline=None):
    """Riemannian gradient descent with backtracking line search on the sphere.
    
    Uses the exponential map: x_new = x * cos(alpha*|g|) + g * sin(alpha*|g|)/|g|
    where g is the Riemannian gradient (tangent vector).
    """
    n = points.shape[0]
    x = _normalize(points.copy())
    best_energy = _energy_only(x)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        e, g = _energy_and_grad(x)
        if e < best_energy:
            best_energy = e
            best_x = x.copy()
        
        # Check convergence
        g_norm = np.linalg.norm(g)
        if g_norm < tol:
            break
        
        # Backtracking line search
        alpha = 1.0
        for _ in range(30):
            # Exponential map on sphere
            if g_norm < 1e-15:
                break
            cos_a = np.cos(alpha * g_norm)
            sin_a_over_g = np.sin(alpha * g_norm) / g_norm
            x_new = x * cos_a + g * sin_a_over_g
            x_new = _normalize(x_new)
            
            e_new = _energy_only(x_new)
            if e_new < e - 1e-4 * alpha * g_norm**2:
                x = x_new
                break
            alpha *= 0.5
        else:
            # If no step accepted, try smaller fixed step
            alpha = 1e-6
            if g_norm < 1e-15:
                break
            cos_a = np.cos(alpha * g_norm)
            sin_a_over_g = np.sin(alpha * g_norm) / g_norm
            x_new = x * cos_a + g * sin_a_over_g
            x = _normalize(x_new)
    
    return x, best_energy


def _fibonacci_spiral(n, offset=0.5, scale=1.0):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i * scale
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)


def _random_points(n, rng):
    """Generate random points on unit sphere."""
    pts = rng.standard_normal((n, 3))
    return _normalize(pts)


def _shifted_spiral(n, shift):
    """Generate a shifted spiral pattern."""
    i = np.arange(n) + 0.5 + shift
    z = 1.0 - 2.0 * np.clip(i / n, 0, 1)
    phi = np.pi * (1.0 + np.sqrt(5.0)) * (i + shift * n)
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)


def _sobol_points(n, seed=42):
    """Generate well-distributed points using Sobol sequence."""
    sampler = qmc.Sobol(d=3, seed=seed, scramble=True)
    samples = sampler.random(n)
    # Map [0,1]^3 to sphere using inverse CDF for uniform distribution
    # Use the fact that for uniform on sphere: z = 2*u - 1, theta = 2*pi*v
    z = 2.0 * samples[:, 0] - 1.0
    theta = 2.0 * np.pi * samples[:, 1]
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(theta), r * np.sin(theta), z])
    return _normalize(pts)


def _structured_lattice(n, rng):
    """Generate a structured lattice-like distribution with perturbation."""
    # Use a grid in spherical coordinates with jitter
    n_theta = int(np.sqrt(n))
    n_phi = n // n_theta
    if n_phi * n_theta < n:
        n_phi += 1
    
    thetas = np.linspace(0, np.pi, n_theta + 1)[1:-1]
    phis = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    
    theta_grid, phi_grid = np.meshgrid(thetas, phis, indexing='ij')
    theta_flat = theta_grid.ravel()
    phi_flat = phi_grid.ravel()
    
    # Take first n points
    if len(theta_flat) > n:
        theta_flat = theta_flat[:n]
        phi_flat = phi_flat[:n]
    elif len(theta_flat) < n:
        # Pad with random
        extra_n = n - len(theta_flat)
        theta_flat = np.concatenate([theta_flat, rng.uniform(0, np.pi, extra_n)])
        phi_flat = np.concatenate([phi_flat, rng.uniform(0, 2*np.pi, extra_n)])
    
    # Add small perturbation
    theta_flat += rng.normal(0, 0.05, n)
    phi_flat += rng.normal(0, 0.05, n)
    
    x = np.sin(theta_flat) * np.cos(phi_flat)
    y = np.sin(theta_flat) * np.sin(phi_flat)
    z = np.cos(theta_flat)
    pts = np.column_stack([x, y, z])
    return _normalize(pts)


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Standard Fibonacci spiral
    initializations.append(_fibonacci_spiral(N))
    
    # 2. Perturbed Fibonacci spiral
    base = _fibonacci_spiral(N)
    perturbed = base + 0.05 * rng.standard_normal((N, 3))
    initializations.append(_normalize(perturbed))
    
    # 3. Shifted spirals
    for shift in [0.1, 0.3, 0.7]:
        initializations.append(_shifted_spiral(N, shift))
    
    # 4. Sobol-based points (better space-filling)
    for seed in [42, 123, 456]:
        initializations.append(_sobol_points(N, seed=seed))
    
    # 5. Structured lattice with perturbation
    initializations.append(_structured_lattice(N, rng))
    
    # 6. Random points
    for _ in range(3):
        initializations.append(_random_points(N, rng))
    
    # 7. Anti-symmetric pairs (better distribution)
    half = N // 2
    pts_half = _random_points(half, rng)
    pts_full = np.vstack([pts_half, -pts_half])
    if N % 2 == 1:
        pts_full = np.vstack([pts_full, np.array([[0, 0, 1.0]])])
    initializations.append(_normalize(pts_full))
    
    # 8. More perturbed spirals
    for scale in [0.8, 1.2]:
        initializations.append(_fibonacci_spiral(N, scale=scale))
    
    # Optimize each initialization
    best_points = initializations[0]
    best_energy = _energy_only(best_points)
    
    # Pre-compute initial energies to sort by quality
    init_energies = []
    for init_pts in initializations:
        init_energies.append(_energy_only(init_pts))
    
    # Sort initializations by energy (best first)
    sorted_indices = np.argsort(init_energies)
    sorted_initializations = [initializations[i] for i in sorted_indices]
    
    # Time budget per restart - reserve time for final polish
    total_restarts = len(sorted_initializations)
    polish_time = 25.0  # Reserve 25s for final Riemannian GD polish
    time_per_restart = (DEADLINE - polish_time) / total_restarts
    
    for init_idx, init_pts in enumerate(sorted_initializations):
        if time.time() > deadline - polish_time:
            break
        
        # Phase 1: Coarse optimization with L-BFGS-B
        try:
            result = minimize(
                fun=lambda x: _energy_only(_normalize(x)),
                x0=init_pts.flatten(),
                jac=lambda x: _energy_and_grad(_normalize(x))[1].flatten(),
                method='L-BFGS-B',
                options={
                    'maxiter': 500,
                    'ftol': 1e-12,
                    'gtol': 1e-8,
                    'maxls': 50
                }
            )
            if result.fun < best_energy:
                best_energy = result.fun
                best_points = _normalize(result.x.reshape(N, 3))
        except Exception:
            continue
        
        if time.time() > deadline - polish_time:
            break
        
        # Phase 2: Fine optimization with tighter tolerances
        try:
            result2 = minimize(
                fun=lambda x: _energy_only(_normalize(x)),
                x0=best_points.flatten(),
                jac=lambda x: _energy_and_grad(_normalize(x))[1].flatten(),
                method='L-BFGS-B',
                options={
                    'maxiter': 1000,
                    'ftol': 1e-15,
                    'gtol': 1e-10,
                    'maxls': 60
                }
            )
            if result2.fun < best_energy:
                best_energy = result2.fun
                best_points = _normalize(result2.x.reshape(N, 3))
        except Exception:
            continue
    
    # Final high-precision polish: Riemannian gradient descent with backtracking
    if time.time() < deadline - 5.0:
        try:
            refined_pts, refined_energy = _riemannian_gradient_descent(
                best_points, max_iter=5000, tol=1e-15, deadline=deadline - 5.0
            )
            if refined_energy < best_energy:
                best_energy = refined_energy
                best_points = refined_pts
        except Exception:
            pass
    
    # Additional L-BFGS-B polish if time remains
    if time.time() < deadline - 3.0:
        try:
            result_final = minimize(
                fun=lambda x: _energy_only(_normalize(x)),
                x0=best_points.flatten(),
                jac=lambda x: _energy_and_grad(_normalize(x))[1].flatten(),
                method='L-BFGS-B',
                options={
                    'maxiter': 2000,
                    'ftol': 1e-16,
                    'gtol': 1e-13,
                    'maxls': 80
                }
            )
            if result_final.fun < best_energy:
                best_energy = result_final.fun
                best_points = _normalize(result_final.x.reshape(N, 3))
        except Exception:
            pass
    
    # Final Riemannian GD polish
    if time.time() < deadline - 2.0:
        try:
            refined_pts, refined_energy = _riemannian_gradient_descent(
                best_points, max_iter=3000, tol=1e-16, deadline=deadline - 2.0
            )
            if refined_energy < best_energy:
                best_energy = refined_energy
                best_points = refined_pts
        except Exception:
            pass
    
    # Ensure valid output
    best_points = _normalize(best_points)
    # Check for any points too close together
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(best_points[i] - best_points[j])
            if d < 1e-9:
                # Nudge apart
                best_points[j] = best_points[j] + 1e-8 * rng.standard_normal(3)
                best_points = _normalize(best_points)
    
    return best_points
