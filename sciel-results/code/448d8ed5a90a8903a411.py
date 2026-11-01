"""Optimized logarithmic energy minimizer for 150 points on the unit sphere using Riemannian optimization with momentum and high-precision polish."""
import time
import math
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 150
DEADLINE = 120.0
SAFETY_MARGIN = 5.0


def _energy_and_grad(x):
    """
    Compute logarithmic energy and Euclidean gradient for points x of shape (N, 3).
    Energy = sum_{i<j} log(1 / ||x_i - x_j||) = -sum_{i<j} log(||x_i - x_j||)
    Gradient is the Euclidean gradient in R^3.
    """
    # Compute pairwise differences (N, N, 3)
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    # Compute squared distances (N, N)
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Avoid log(0) and division by zero
    dist_sq = np.maximum(dist_sq, 1e-30)
    dist = np.sqrt(dist_sq)
    
    # Mask for upper triangle i < j
    iu = np.triu_indices(N, k=1)
    log_dists = np.log(dist[iu])
    energy = -np.sum(log_dists)
    
    # Gradient computation
    # grad[i] = - sum_{j!=i} (x_i - x_j) / ||x_i - x_j||
    grad = np.zeros_like(x)
    
    # Get distances and vectors for i < j
    d_ij = dist[iu]  # (M,)
    dvec = diff[iu]  # (M, 3) -> x_i - x_j
    
    i_idx = iu[0]
    j_idx = iu[1]
    
    contrib = dvec / d_ij[:, np.newaxis]
    
    np.add.at(grad, i_idx, -contrib)
    np.add.at(grad, j_idx, contrib)
    
    return energy, grad


def _log_energy_fast(x):
    """Fast energy computation only (no gradient)."""
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


def _farthest_point_init(n, max_iter=5000):
    """
    Farthest point sampling: start with a random point, then iteratively
    add the point farthest from the current set. This produces well-spread points.
    """
    rng = np.random.default_rng(12345)
    points = [rng.normal(size=3)]
    points[0] = points[0] / np.linalg.norm(points[0])
    
    for _ in range(n - 1):
        pts_arr = np.array(points)
        # Compute min distance from each candidate to existing points
        # Sample many candidates and pick the one with max min-distance
        candidates = rng.normal(size=(max_iter, 3))
        candidates = candidates / np.linalg.norm(candidates, axis=1, keepdims=True)
        
        # Compute distances from candidates to all existing points
        # (max_iter, len(points), 3) -> (max_iter, len(points))
        diffs = candidates[:, np.newaxis, :] - pts_arr[np.newaxis, :, :]
        dists_sq = np.sum(diffs * diffs, axis=2)
        min_dists_sq = np.min(dists_sq, axis=1)
        
        best_idx = np.argmax(min_dists_sq)
        points.append(candidates[best_idx])
    
    return np.array(points)


def _normalize(x):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return x / norms


def _riemannian_grad_descent(x0, deadline, max_iter=20000, tol=1e-10, momentum=0.9):
    """
    Riemannian Gradient Descent with Nesterov momentum and backtracking line search.
    Uses projection retraction onto the sphere.
    """
    x = x0.copy()
    energy, grad = _energy_and_grad(x)
    
    # Project gradient to tangent space
    grad_tangent = grad - np.sum(grad * x, axis=1, keepdims=True) * x
    
    # Momentum velocity
    v = np.zeros_like(x)
    
    # Initial step size
    alpha = 0.05
    prev_energy = energy
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Check convergence
        if np.linalg.norm(grad_tangent) < tol:
            break
            
        # Nesterov momentum: look ahead
        v = momentum * v - alpha * grad_tangent
        step_dir = v
        
        step_norm = np.linalg.norm(step_dir, axis=1, keepdims=True)
        
        if np.max(step_norm) < 1e-15:
            break
            
        found_alpha = False
        alpha_trial = alpha
        for _ in range(30):
            x_trial = x + alpha_trial * step_dir
            x_trial = _normalize(x_trial)
            
            energy_trial = _log_energy_fast(x_trial)
            
            if energy_trial < energy:
                found_alpha = True
                break
            alpha_trial *= 0.5
            
        if not found_alpha:
            x_trial = x + 1e-9 * step_dir
            x_trial = _normalize(x_trial)
            energy_trial = _log_energy_fast(x_trial)
            if energy_trial >= energy:
                break
        
        x = x_trial
        energy = energy_trial
        
        # Compute new gradient
        _, grad = _energy_and_grad(x)
        grad_tangent = grad - np.sum(grad * x, axis=1, keepdims=True) * x
        
        # Adapt alpha: increase slowly if we took full step
        if found_alpha and alpha_trial == alpha:
            alpha = min(alpha * 1.05, 0.2)
        elif not found_alpha:
            alpha = max(alpha * 0.7, 0.001)
        
    return x


def _lbfgs_on_manifold(x0, deadline):
    """
    Alternative: Use L-BFGS but enforce constraints more strictly?
    Actually, standard L-BFGS on R^3 with normalization is often okay if started well,
    but Riemannian GD is more principled.
    Let's try a hybrid: L-BFGS on the tangent space? No, that's complex.
    
    Let's try running L-BFGS on the full vector but with a very tight ftol/gtol
    and frequent normalization? No, L-BFGS doesn't support custom retraction.
    
    Instead, let's use a high-precision polish using Riemannian CG or Newton.
    For now, we'll rely on Riemannian GD with many iterations.
    """
    pass


def _polish(x0, deadline):
    """
    High-precision polish pass.
    Uses Riemannian Gradient Descent with very tight tolerance and small steps,
    followed by a local L-BFGS refinement in tangent space.
    """
    remaining = deadline - time.time()
    if remaining < 2.0:
        return x0
    
    # Phase 1: RGD with tighter tolerance
    x_opt = _riemannian_grad_descent(x0, deadline - 10.0, max_iter=30000, tol=1e-11, momentum=0.95)
    
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
    
    # Phase 2: Local L-BFGS refinement using parameterization
    # Use the fact that near the optimum, we can parameterize small perturbations
    # in the tangent space and use L-BFGS
    remaining = deadline - time.time()
    if remaining > 15.0:
        x_refined = _lbfgs_tangent_polish(x_opt, deadline)
        if x_refined is not None:
            if np.all(np.isfinite(x_refined)):
                if np.min(np.linalg.norm(x_refined, axis=1)) > 1e-15:
                    diff = x_refined[:, np.newaxis, :] - x_refined[np.newaxis, :, :]
                    dist_sq = np.sum(diff * diff, axis=2)
                    np.fill_diagonal(dist_sq, np.inf)
                    if np.min(dist_sq) >= (1e-9) ** 2:
                        return x_refined
    
    return x_opt


def _lbfgs_tangent_polish(x0, deadline):
    """
    L-BFGS polish in tangent space. Parameterize points as exp(x0, v) where v is in tangent space.
    For small v, exp(x, v) ≈ x + v - 0.5*(v·x)*x + ...
    We optimize v (450-dim) to minimize energy(exp(x0, v)).
    """
    def _energy_from_tangent(v_flat):
        v = v_flat.reshape(N, 3)
        # Exponential map approximation: x_new = x0 + v - 0.5*(v·x0)*x0
        dot_vx = np.sum(v * x0, axis=1, keepdims=True)
        x_new = x0 + v - 0.5 * dot_vx * x0
        x_new = _normalize(x_new)
        return _log_energy_fast(x_new)
    
    def _grad_from_tangent(v_flat):
        v = v_flat.reshape(N, 3)
        dot_vx = np.sum(v * x0, axis=1, keepdims=True)
        x_new = x0 + v - 0.5 * dot_vx * x0
        x_new = _normalize(x_new)
        _, grad_euclidean = _energy_and_grad(x_new)
        # Project to tangent at x_new, then pull back to tangent at x0
        grad_tangent_new = grad_euclidean - np.sum(grad_euclidean * x_new, axis=1, keepdims=True) * x_new
        # Approximate pullback: for small v, tangent spaces are nearly aligned
        # Use identity as approximation for small perturbations
        return grad_tangent_new.flatten()
    
    v0 = np.zeros(N * 3)
    
    try:
        result = minimize(
            _energy_from_tangent,
            v0,
            jac=_grad_from_tangent,
            method='L-BFGS-B',
            options={
                'maxiter': 200,
                'ftol': 1e-14,
                'gtol': 1e-10,
                'maxfun': 500,
            }
        )
        
        if result.success or result.fun < _log_energy_fast(x0):
            v_opt = result.x.reshape(N, 3)
            dot_vx = np.sum(v_opt * x0, axis=1, keepdims=True)
            x_new = x0 + v_opt - 0.5 * dot_vx * x0
            x_new = _normalize(x_new)
            
            # Validate
            if np.all(np.isfinite(x_new)):
                if np.min(np.linalg.norm(x_new, axis=1)) > 1e-15:
                    return x_new
    except Exception:
        pass
    
    return None


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    best_x = None
    best_energy = float('inf')
    
    # Generate multiple initializations
    initializations = [
        ("fibonacci", _fibonacci_sphere(N)),
        ("halton", _halton_sphere(N)),
        ("farthest", _farthest_point_init(N)),
    ]
    
    # Add random initializations
    np.random.seed(42)
    for _ in range(4):
        x_rand = np.random.randn(N, 3)
        x_rand = x_rand / np.linalg.norm(x_rand, axis=1, keepdims=True)
        initializations.append(("random", x_rand))
    
    for name, x0 in initializations:
        if time.time() > deadline - 15.0:
            break
        
        x0 = _normalize(x0)
        
        # Run Riemannian Gradient Descent with momentum
        x_opt = _riemannian_grad_descent(x0, deadline - 10.0, max_iter=15000, tol=1e-10, momentum=0.9)
        
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
    
    # Final polish on the best result
    if best_x is not None and time.time() < deadline - 5.0:
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
