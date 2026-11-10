"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math


def _log_energy(points):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|) using math.fsum."""
    n = len(points)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(points[i] - points[j])
            if d < 1e-9:
                return 1e10  # penalty for too-close points
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)


def _log_energy_grad(points):
    """Compute gradient of logarithmic energy with respect to points on sphere (projected)."""
    n = len(points)
    grad = np.zeros_like(points)
    for i in range(n):
        for j in range(i + 1, n):
            d_vec = points[i] - points[j]
            d = np.linalg.norm(d_vec)
            if d < 1e-9:
                continue
            # d/dx_i log(1/d) = -1/d * d/dx_i d = -1/d * (x_i - x_j)/d = -(x_i - x_j)/d^2
            # d/dx_j log(1/d) = +1/d * d/dx_j d = +1/d * (x_j - x_i)/d = (x_i - x_j)/d^2
            factor = 1.0 / (d * d)
            grad[i] -= d_vec * factor
            grad[j] += d_vec * factor
    # Project gradient onto tangent space of sphere: grad_tangent = grad - (grad . x) * x
    for i in range(n):
        proj = np.dot(grad[i], points[i])
        grad[i] -= proj * points[i]
    return grad


def construct():
    # Start with a good initial configuration: Fibonacci sphere
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Flatten for optimization
    x0_flat = x0.flatten()
    
    def energy_flat(flat):
        pts = flat.reshape(N, 3)
        # Normalize each point to unit sphere
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        pts = pts / norms
        return _log_energy(pts)
    
    def grad_flat(flat):
        pts = flat.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        pts = pts / norms
        g = _log_energy_grad(pts)
        return g.flatten()
    
    def optimize_from(x0_flat):
        """Run L-BFGS-B from a given starting point."""
        result = minimize(
            energy_flat,
            x0_flat,
            jac=grad_flat,
            method='L-BFGS-B',
            options={'maxiter': 10000, 'ftol': 1e-15, 'gtol': 1e-14, 'maxls': 100}
        )
        opt_flat = result.x
        pts = opt_flat.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        pts = pts / norms
        return pts, _log_energy(pts)
    
    # Multi-start: try several different initializations
    best_pts = None
    best_energy = float('inf')
    
    # Start 1: Fibonacci sphere
    pts, e = optimize_from(x0_flat.copy())
    if e < best_energy:
        best_energy = e
        best_pts = pts.copy()
    
    # Start 2: Perturbed Fibonacci sphere
    rng = np.random.default_rng(42)
    for trial in range(3):
        x0_perturbed = x0_flat.copy()
        noise = rng.normal(0, 0.1, size=x0_flat.shape)
        x0_perturbed += noise
        pts, e = optimize_from(x0_perturbed)
        if e < best_energy:
            best_energy = e
            best_pts = pts.copy()
    
    # Start 3: Random initialization
    x0_random = rng.normal(0, 1, size=(N, 3))
    x0_random /= np.linalg.norm(x0_random, axis=1, keepdims=True)
    pts, e = optimize_from(x0_random.flatten())
    if e < best_energy:
        best_energy = e
        best_pts = pts.copy()
    
    # Final refinement from best point found
    if best_pts is not None:
        result = minimize(
            energy_flat,
            best_pts.flatten(),
            jac=grad_flat,
            method='L-BFGS-B',
            options={'maxiter': 20000, 'ftol': 1e-16, 'gtol': 1e-15, 'maxls': 200}
        )
        opt_flat = result.x
        pts = opt_flat.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        pts = pts / norms
        final_e = _log_energy(pts)
        if final_e < best_energy:
            best_pts = pts
            best_energy = final_e
    
    return best_pts
# EVOLVE-BLOCK-END
