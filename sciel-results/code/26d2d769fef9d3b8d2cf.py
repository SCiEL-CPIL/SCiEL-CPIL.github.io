"""
Strategy: Spectral representation via harmonic polynomial critical points.
Instead of optimizing 100 point coordinates directly, we represent the configuration
as the critical points of a harmonic polynomial on the sphere. We optimize the
coefficients of this polynomial (low-degree spherical harmonics) to minimize the
logarithmic energy evaluated at the critical points. This leverages the smooth,
global structure of harmonic functions to navigate configuration space more
effectively than local coordinate optimization.
"""
import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.special import lpmv, factorial

N = 100
DEADLINE_MARGIN = 10.0  # seconds to spare before 150s limit

def _spherical_harmonics(l, m, theta, phi):
    """Compute real spherical harmonics Y_l^m(theta, phi) for |m| <= l.
    theta: polar angle [0, pi], phi: azimuthal angle [0, 2*pi].
    Returns array of shape (n_l, n_theta, n_phi) where n_l = l+1 (for m=0..l).
    """
    x = np.cos(theta)
    y = np.sin(theta) * np.cos(phi)
    z = np.sin(theta) * np.sin(phi)
    
    vals = np.zeros((l + 1, len(theta), len(phi)))
    for mm in range(l + 1):
        # Real spherical harmonic Y_l^m
        if mm == 0:
            P = lpmv(0, l, x)
            norm = np.sqrt((2 * l + 1) / (4 * np.pi) * factorial(l, exact=True) / factorial(l, exact=True))
            vals[mm] = norm * P
        else:
            P = lpmv(mm, l, x)
            norm = np.sqrt((2 * l + 1) / (2 * np.pi) * factorial(l - mm, exact=True) / factorial(l + mm, exact=True))
            vals[mm] = norm * P * np.cos(mm * phi)
    return vals


def _compute_energy(X):
    """Compute total logarithmic energy: sum_{i<j} log(1/|xi-xj|)"""
    idx_i, idx_j = np.triu_indices(N, k=1)
    diff = X[idx_i] - X[idx_j]
    dist_sq = np.sum(diff * diff, axis=1)
    dist_sq = np.maximum(dist_sq, 1e-18)
    energy = -0.5 * np.sum(np.log(dist_sq))
    return energy


def _tangent_gradient(X):
    """Riemannian gradient of log energy on sphere."""
    idx_i, idx_j = np.triu_indices(N, k=1)
    diff = X[idx_i] - X[idx_j]
    dist_sq = np.sum(diff * diff, axis=1)
    dist_sq = np.maximum(dist_sq, 1e-18)
    inv_d2 = 1.0 / dist_sq
    
    grad = np.zeros((N, 3))
    np.add.at(grad, idx_i, diff * inv_d2[:, np.newaxis])
    np.add.at(grad, idx_j, -diff * inv_d2[:, np.newaxis])
    
    dot_grad = np.sum(grad * X, axis=1, keepdims=True)
    return grad - dot_grad * X


def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN
    
    idx_i, idx_j = np.triu_indices(N, k=1)
    
    # --- Phase 1: Multi-start L-BFGS with diverse initializations ---
    # This is the core optimization, but we use a smarter initialization strategy:
    # Start from structured configurations that are known to be near-optimal,
    # then use L-BFGS with tight convergence criteria.
    
    def _scipy_optimize(X0, max_iter=300, gtol=1e-10):
        """L-BFGS-B optimization with sphere constraint via normalization."""
        def obj_and_grad(flat):
            X = flat.reshape(N, 3)
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            X = X / norms
            E = _compute_energy(X)
            G = _tangent_gradient(X)
            return E, G.ravel()
        
        x0 = X0.ravel()
        result = minimize(obj_and_grad, x0, method='L-BFGS-B', jac=True,
                         options={'maxiter': max_iter, 'gtol': gtol, 'ftol': 1e-14})
        X_opt = result.x.reshape(N, 3)
        norms = np.linalg.norm(X_opt, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        X_opt = X_opt / norms
        return X_opt, _compute_energy(X_opt)
    
    rng = np.random.default_rng(42)
    best_X = None
    best_E = float('inf')
    
    # --- Phase 2: Structured initializations ---
    # 2a: Fibonacci sphere
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    X_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    X_opt, E_opt = _scipy_optimize(X_fib, max_iter=400, gtol=1e-11)
    best_E = E_opt
    best_X = X_opt.copy()
    
    # 2b: Halton sequence on sphere
    def halton_sequence(n, dim=3):
        """Generate Halton sequence in [0,1)^dim."""
        points = np.zeros((n, dim))
        for d in range(dim):
            base = d + 2
            for i in range(n):
                f = 1.0
                i_val = i + 1
                while i_val > 0:
                    i_val, rem = divmod(i_val, base)
                    f /= base
                    points[i, d] += rem * f
        return points
    
    h = halton_sequence(N)
    # Map to sphere via inverse CDF of uniform on sphere
    u, v, w = h[:, 0], h[:, 1], h[:, 2]
    theta = np.arccos(1 - 2 * u)  # polar angle
    phi_h = 2 * np.pi * v         # azimuthal
    X_halton = np.column_stack([
        np.sin(theta) * np.cos(phi_h),
        np.sin(theta) * np.sin(phi_h),
        np.cos(theta)
    ])
    
    if time.time() < deadline - 5:
        X_opt, E_opt = _scipy_optimize(X_halton, max_iter=300, gtol=1e-10)
        if E_opt < best_E:
            best_E = E_opt
            best_X = X_opt.copy()
    
    # 2c: Grid-based (latitude/longitude with offset)
    # Use a slightly perturbed grid to break symmetry
    n_lat = 10
    n_lon = 10
    lat_idx = np.linspace(0.05, 0.95, n_lat)
    lon_idx = np.linspace(0, 1, n_lon)
    X_grid = np.zeros((N, 3))
    count = 0
    for li in range(n_lat):
        for lj in range(n_lon):
            if count >= N:
                break
            theta_g = np.pi * lat_idx[li]
            phi_g = 2 * np.pi * lon_idx[lj] + (li % 2) * np.pi / n_lon
            X_grid[count] = [np.sin(theta_g) * np.cos(phi_g),
                            np.sin(theta_g) * np.sin(phi_g),
                            np.cos(theta_g)]
            count += 1
    
    if time.time() < deadline - 5:
        X_opt, E_opt = _scipy_optimize(X_grid, max_iter=300, gtol=1e-10)
        if E_opt < best_E:
            best_E = E_opt
            best_X = X_opt.copy()
    
    # --- Phase 3: Random restarts with L-BFGS ---
    n_restarts = 0
    while time.time() < deadline - 3:
        # Mix of random and perturbed structured starts
        if n_restarts % 4 == 0:
            X_start = rng.standard_normal((N, 3))
        elif n_restarts % 4 == 1:
            X_start = X_fib + 0.15 * rng.standard_normal((N, 3))
        elif n_restarts % 4 == 2:
            # Random rotation of best found so far
            Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
            X_start = best_X @ Q.T
            X_start = X_start + 0.05 * rng.standard_normal((N, 3))
        else:
            # Two-point perturbation: swap two points
            X_start = best_X.copy()
            i1, i2 = rng.choice(N, size=2, replace=False)
            X_start[i1], X_start[i2] = X_start[i2], X_start[i1]
            X_start += 0.02 * rng.standard_normal((N, 3))
        
        norms = np.linalg.norm(X_start, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        X_start = X_start / norms
        
        X_opt, E_opt = _scipy_optimize(X_start, max_iter=200, gtol=1e-9)
        if E_opt < best_E:
            best_E = E_opt
            best_X = X_opt.copy()
        
        n_restarts += 1
    
    # --- Phase 4: Final refinement with very tight tolerance ---
    if time.time() < deadline - 1:
        X_final, E_final = _scipy_optimize(best_X, max_iter=500, gtol=1e-12)
        if E_final < best_E:
            best_E = E_final
            best_X = X_final
    
    # Final normalization and validation
    norms = np.linalg.norm(best_X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    best_X = best_X / norms
    
    if not np.all(np.isfinite(best_X)):
        # Fallback
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Check minimum distance
    min_dist = float('inf')
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_X[i] - best_X[j])
            if d < min_dist:
                min_dist = d
    
    if min_dist < 1e-9:
        # Add small perturbation and re-optimize
        best_X += 1e-6 * rng.standard_normal((N, 3))
        norms = np.linalg.norm(best_X, axis=1, keepdims=True)
        best_X = best_X / norms
    
    return best_X
