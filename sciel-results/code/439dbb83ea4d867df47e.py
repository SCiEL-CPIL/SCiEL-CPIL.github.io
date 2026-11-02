"""Logarithmic energy minimization for 150 points on the unit sphere."""
import numpy as np
from scipy.optimize import minimize
import time
import math

N = 150
DEADLINE = 130.0  # seconds, leave 20s margin under 150s limit


def _log_energy_and_grad(x):
    """
    Compute logarithmic energy E = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    and its gradient with respect to x (3N coordinates).
    d_ij = |x_i - x_j| (chordal distance on unit sphere, but we use Euclidean distance in R^3)
    
    For points on the sphere, the gradient of -log(|x_i - x_j|) w.r.t. x_i is:
        (x_i - x_j) / |x_i - x_j|^2
    (since d/dx_i log(1/|x_i-x_j|) = -d/dx_i log|x_i-x_j| = -(x_i-x_j)/|x_i-x_j|^2 ... wait)
    
    Let me be careful:
    E = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    
    d_ij = ||x_i - x_j||
    d/dx_i log(d_ij) = (x_i - x_j)^T / ||x_i - x_j||^2 * (x_i - x_j) ... no
    
    Actually: d/dx_i ||x_i - x_j|| = (x_i - x_j) / ||x_i - x_j||
    So d/dx_i log(||x_i - x_j||) = (x_i - x_j) / ||x_i - x_j||^2
    
    Therefore: d/dx_i [-log(||x_i - x_j||)] = -(x_i - x_j) / ||x_i - x_j||^2
    
    For the full sum, the gradient at point i is:
    grad_i = sum_{j != i} -(x_i - x_j) / ||x_i - x_j||^2
    
    But we need to project onto the tangent space of the sphere since we're constrained to the sphere.
    However, if we optimize in R^3 and project back, or use a parameterization...
    
    Actually, let's just optimize in R^3 with a sphere constraint via projection, or use the fact
    that for points on the sphere, the gradient projected to tangent space is:
    grad_tangent_i = grad_i - (grad_i . x_i) x_i
    
    But scipy's L-BFGS-B doesn't handle sphere constraints well. Let's use a different approach:
    optimize on the sphere using a retraction.
    
    Alternative: Use spherical coordinates or just project after each step.
    
    Let me use a simpler approach: minimize in R^3 with penalty for non-unit norm, 
    or use the fact that we can project back to sphere.
    
    Actually, the cleanest approach for sphere-constrained optimization is to:
    1. Work in R^3
    2. At each iteration, compute gradient in R^3
    3. Project gradient to tangent space: g_t = g - (g.x)x
    4. Take a step and project back to sphere
    
    But scipy.optimize.minimize doesn't support this directly. Let's use a custom approach.
    
    Alternative: Use the Gram matrix or quaternion representation.
    
    Simplest robust approach: Use scipy.optimize.minimize with method='L-BFGS-B' 
    on the 3N coordinates, but with a large penalty for deviation from unit sphere.
    
    Or better: use a parameterization. But 150 points on sphere with 2 params each = 300 params.
    
    Let me use the approach of optimizing in R^3 and projecting. I'll implement a simple
    projected gradient / conjugate gradient method.
    """
    pts = x.reshape(N, 3)
    
    # Compute pairwise distances
    # diff[i,j] = pts[i] - pts[j]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 3)
    dist = np.linalg.norm(diff, axis=2)  # (N, N)
    
    # Avoid division by zero
    dist = np.maximum(dist, 1e-15)
    
    # Energy: sum over i<j of log(1/d_ij) = -sum_{i<j} log(d_ij)
    # Use upper triangle
    iu = np.triu_indices(N, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient: for each point i, grad_i = sum_{j != i} -(pts[i] - pts[j]) / dist[i,j]^2
    # grad shape: (N, 3)
    # For pair (i,j), contribution to grad_i is -(pts[i]-pts[j])/dist^2
    #            contribution to grad_j is +(pts[i]-pts[j])/dist^2 = -(pts[j]-pts[i])/dist^2
    
    # diff[i,j] = pts[i] - pts[j]
    # inv_dist_sq = 1 / dist^2
    inv_dist_sq = 1.0 / (dist * dist)
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # grad[i] = sum_j -(diff[i,j]) * inv_dist_sq[i,j]
    # = -sum_j diff[i,j] * inv_dist_sq[i,j]
    grad = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)  # (N, 3)
    
    # Project gradient to tangent space of sphere
    # For point x_i on sphere, tangent projection: g_t = g - (g.x_i)x_i
    grad_tangent = grad - np.sum(grad * pts, axis=1)[:, np.newaxis] * pts
    
    return energy, grad_tangent.reshape(3 * N)


def _project_to_sphere(x):
    """Project 3N vector back to unit sphere points."""
    pts = x.reshape(N, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return (pts / norms).reshape(-1)


def _energy_only(x):
    """Just compute energy for evaluation."""
    e, _ = _log_energy_and_grad(x)
    return e


def construct():
    t_start = time.time()
    
    # Initial guess: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    x0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z]).reshape(-1)
    
    # Project to ensure unit sphere
    x0 = _project_to_sphere(x0)
    
    best_x = x0.copy()
    best_energy = _energy_only(x0)
    
    # Use L-BFGS-B with sphere projection via penalty
    # Or better: implement a custom projected conjugate gradient
    
    # Let's try scipy's minimize with a penalty for non-unit norm
    def penalized_energy_grad(x):
        e, g = _log_energy_and_grad(x)
        pts = x.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1)
        # Penalty for deviation from unit norm
        penalty = 1e6 * np.sum((norms - 1.0) ** 2)
        # Gradient of penalty: 2e6 * (norms - 1) * pts / norms
        pen_grad = 2e6 * (norms - 1.0)[:, np.newaxis] * pts / np.maximum(norms[:, np.newaxis], 1e-15)
        g_pen = g + pen_grad.reshape(-1)
        return e + penalty, g_pen
    
    # Try L-BFGS-B
    try:
        res = minimize(penalized_energy_grad, x0, method='L-BFGS-B',
                       jac=True, options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12})
        if res.x is not None and np.all(np.isfinite(res.x)):
            x_candidate = _project_to_sphere(res.x)
            e_candidate = _energy_only(x_candidate)
            if e_candidate < best_energy:
                best_x = x_candidate
                best_energy = e_candidate
    except Exception:
        pass
    
    if time.time() - t_start > DEADLINE - 30:
        return best_x.reshape(N, 3)
    
    # Second phase: more refined optimization with tighter tolerances
    # Use the result from first phase as starting point
    try:
        res2 = minimize(penalized_energy_grad, best_x, method='L-BFGS-B',
                        jac=True, options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-14})
        if res2.x is not None and np.all(np.isfinite(res2.x)):
            x_candidate = _project_to_sphere(res2.x)
            e_candidate = _energy_only(x_candidate)
            if e_candidate < best_energy:
                best_x = x_candidate
                best_energy = e_candidate
    except Exception:
        pass
    
    if time.time() - t_start > DEADLINE - 15:
        return best_x.reshape(N, 3)
    
    # Third phase: local refinement using BFGS on tangent space
    # Implement a simple projected gradient descent with line search
    x_cur = best_x.copy()
    for step in range(5000):
        if time.time() - t_start > DEADLINE - 5:
            break
        
        e, g = _log_energy_and_grad(x_cur)
        if e < best_energy:
            best_energy = e
            best_x = x_cur.copy()
        
        # Check convergence
        if np.max(np.abs(g)) < 1e-12:
            break
        
        # Backtracking line search
        alpha = 0.1
        for _ in range(30):
            x_new = _project_to_sphere(x_cur - alpha * g)
            e_new = _energy_only(x_new)
            if e_new < e:
                break
            alpha *= 0.5
        
        if e_new < e:
            x_cur = x_new
        else:
            # Try smaller step or break
            alpha *= 0.1
            if alpha < 1e-10:
                break
            x_new = _project_to_sphere(x_cur - alpha * g)
            if _energy_only(x_new) < e:
                x_cur = x_new
            else:
                break
    
    # Final polish: one more L-BFGS-B pass
    try:
        res3 = minimize(penalized_energy_grad, best_x, method='L-BFGS-B',
                        jac=True, options={'maxiter': 5000, 'ftol': 1e-17, 'gtol': 1e-15})
        if res3.x is not None and np.all(np.isfinite(res3.x)):
            x_candidate = _project_to_sphere(res3.x)
            e_candidate = _energy_only(x_candidate)
            if e_candidate < best_energy:
                best_x = x_candidate
                best_energy = e_candidate
    except Exception:
        pass
    
    # Ensure final result is on unit sphere and valid
    result = _project_to_sphere(best_x).reshape(N, 3)
    
    # Validate
    if not np.all(np.isfinite(result)):
        # Fallback to Fibonacci spiral
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
        result = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    return result
