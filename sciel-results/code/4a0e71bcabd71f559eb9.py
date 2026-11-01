"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math


def _log_energy(pts):
    """Compute the logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    n = len(pts)
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-15:
                return float('inf')
            energy += math.log(1.0 / d)
    return energy


def _log_energy_vectorized(pts):
    """Compute the logarithmic energy using vectorized operations for speed."""
    n = len(pts)
    # Compute pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Only upper triangle
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    if np.any(d < 1e-15):
        return float('inf')
    return float(math.fsum(math.log(1.0 / di) for di in d))


def _gradient(pts):
    """Compute gradient of the logarithmic energy with respect to points."""
    n = len(pts)
    grad = np.zeros_like(pts)
    for i in range(n):
        for j in range(i + 1, n):
            d_vec = pts[i] - pts[j]
            d = np.linalg.norm(d_vec)
            if d < 1e-15:
                continue
            # d/dx_i log(1/d) = -1/d * d/dx_i d = -1/d * d_vec/d = -d_vec/d^2
            # Wait: log(1/d) = -log(d)
            # d/dx_i (-log(d)) = -1/d * d_vec/d = -d_vec/d^2
            # So gradient contribution to point i is -d_vec/d^2
            # Gradient contribution to point j is +d_vec/d^2
            term = -d_vec / (d * d)
            grad[i] += term
            grad[j] -= term
    return grad


def _project_to_sphere(pts):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _fibonacci_sphere(n):
    """Generate initial points using golden angle spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def construct():
    # Start with Fibonacci spiral as initial guess
    pts = _fibonacci_sphere(N).astype(np.float64)
    
    # Use L-BFGS-B with sphere projection via penalty or direct constraint
    # We'll optimize in Cartesian coordinates with a penalty for deviation from sphere
    
    def objective_and_grad(x_flat):
        pts = x_flat.reshape(N, 3)
        # Project to sphere for energy computation
        pts_norm = _project_to_sphere(pts)
        
        # Energy
        energy = _log_energy_vectorized(pts_norm)
        
        # Penalty for not being on sphere
        norms = np.linalg.norm(pts, axis=1)
        penalty = 1e6 * np.sum((norms - 1.0) ** 2)
        
        total = energy + penalty
        
        # Gradient
        grad_pts = np.zeros_like(pts)
        
        # Gradient of energy w.r.t. pts_norm, then chain rule through projection
        # For simplicity, use finite difference or analytic gradient on projected points
        # Actually, let's just compute gradient on the projected points and project back
        
        if np.isfinite(energy):
            g = _gradient(pts_norm)
            # Chain rule: d(pts_norm)/d(pts) is not trivial, but for small deviations
            # from sphere, we can approximate
            # Simpler: just use the gradient on the sphere and project
            grad_pts = g
        
        # Add penalty gradient
        norms_full = np.linalg.norm(pts, axis=1, keepdims=True)
        penalty_grad = 2 * 1e6 * (norms - 1.0) * pts / norms_full
        
        grad_total = grad_pts + penalty_grad
        grad_flat = grad_total.flatten()
        
        return total, grad_flat
    
    # Optimize
    x0 = pts.flatten()
    result = minimize(
        objective_and_grad,
        x0,
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
    )
    
    pts_opt = _project_to_sphere(result.x.reshape(N, 3))
    
    # Local refinement with steepest descent in tangent space
    for it in range(500):
        pts_norm = _project_to_sphere(pts_opt)
        g = _gradient(pts_norm)
        
        # Project gradient to tangent space
        g_tangent = g - np.sum(g * pts_norm, axis=1, keepdims=True) * pts_norm
        
        # Adaptive step size
        step = 1e-3
        pts_new = pts_norm + step * g_tangent
        pts_new = _project_to_sphere(pts_new)
        
        e_old = _log_energy_vectorized(pts_norm)
        e_new = _log_energy_vectorized(pts_new)
        
        if e_new < e_old:
            pts_opt = pts_new
            step = min(step * 1.5, 0.1)
        else:
            step *= 0.5
            if step < 1e-10:
                break
    
    pts_final = _project_to_sphere(pts_opt)
    
    # Ensure no points are too close
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(pts_final[i] - pts_final[j])
            if d < 1e-9:
                # Perturb slightly
                pts_final[j] += 1e-8 * np.random.randn(3)
                pts_final[j] = _project_to_sphere(pts_final[j].reshape(1, 3)).flatten()
    
    return pts_final
# EVOLVE-BLOCK-END
