"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize


def _log_energy(pts):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    n = len(pts)
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-15:
                return 1e300
            total += math.log(1.0 / d)
    return total


def _log_energy_vec(pts):
    """Vectorized logarithmic energy."""
    n = len(pts)
    # Compute all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    # Mask diagonal and lower triangle
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d = dist[mask]
    d = np.maximum(d, 1e-15)
    return np.sum(np.log(1.0 / d))


def _log_energy_grad(pts):
    """Gradient of logarithmic energy w.r.t. point positions on sphere.
    
    E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log|x_i - x_j|
    dE/dx_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = len(pts)
    grad = np.zeros_like(pts)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d_vec = pts[i] - pts[j]
            d = np.linalg.norm(d_vec)
            if d < 1e-15:
                continue
            grad[i] += d_vec / (d * d)
    return grad


def _project_to_sphere(pts):
    """Project points back to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _log_energy_constrained(pts_flat):
    """Energy for scipy minimize with spherical constraint handled by projection."""
    pts = pts_flat.reshape(-1, 3)
    pts = _project_to_sphere(pts)
    return _log_energy_vec(pts)


def _golden_spiral():
    """Generate golden-angle Fibonacci spiral points."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_points():
    """Generate random points on unit sphere."""
    theta = np.random.uniform(0, 2 * np.pi, N)
    phi = np.arccos(np.random.uniform(-1, 1, N))
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.column_stack([x, y, z])


def construct():
    deadline = time.time() + 140  # Safety margin
    
    best_pts = _golden_spiral()
    best_energy = _log_energy(best_pts)
    
    # Try multiple initializations and local optimizations
    initializations = [best_pts.copy()]
    
    # Add random initializations
    for _ in range(5):
        if time.time() > deadline - 5:
            break
        initializations.append(_random_points())
    
    # Add perturbed golden spiral
    for _ in range(3):
        if time.time() > deadline - 5:
            break
        perturbed = best_pts + 0.01 * np.random.randn(N, 3)
        perturbed = _project_to_sphere(perturbed)
        initializations.append(perturbed)
    
    for init_pts in initializations:
        if time.time() > deadline - 10:
            break
        
        # Use L-BFGS-B with projection
        x0 = init_pts.flatten()
        
        def fun(x):
            pts = x.reshape(-1, 3)
            pts = _project_to_sphere(pts)
            return _log_energy_vec(pts)
        
        def jac(x):
            pts = x.reshape(-1, 3)
            pts = _project_to_sphere(pts)
            g = _log_energy_grad(pts)
            # Project gradient to tangent space
            for i in range(N):
                g[i] -= np.dot(g[i], pts[i]) * pts[i]
            return g.flatten()
        
        try:
            result = minimize(
                fun, x0, jac=jac,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'gtol': 1e-12, 'ftol': 1e-14}
            )
            
            if result.success or result.fun < best_energy:
                new_pts = _project_to_sphere(result.x.reshape(-1, 3))
                new_energy = _log_energy(new_pts)
                
                # Validate: check no points too close
                valid = True
                for i in range(N):
                    for j in range(i + 1, N):
                        if np.linalg.norm(new_pts[i] - new_pts[j]) < 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid and np.all(np.isfinite(new_pts)):
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_pts = new_pts
        except Exception:
            continue
    
    # Final high-precision polish if time remains
    if time.time() < deadline - 15:
        x0 = best_pts.flatten()
        
        def fun2(x):
            pts = x.reshape(-1, 3)
            pts = _project_to_sphere(pts)
            return _log_energy_vec(pts)
        
        def jac2(x):
            pts = x.reshape(-1, 3)
            pts = _project_to_sphere(pts)
            g = _log_energy_grad(pts)
            for i in range(N):
                g[i] -= np.dot(g[i], pts[i]) * pts[i]
            return g.flatten()
        
        try:
            result = minimize(
                fun2, x0, jac=jac2,
                method='L-BFGS-B',
                options={'maxiter': 5000, 'gtol': 1e-14, 'ftol': 1e-16}
            )
            if result.fun < best_energy:
                new_pts = _project_to_sphere(result.x.reshape(-1, 3))
                new_energy = _log_energy(new_pts)
                valid = True
                for i in range(N):
                    for j in range(i + 1, N):
                        if np.linalg.norm(new_pts[i] - new_pts[j]) < 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if valid and np.all(np.isfinite(new_pts)):
                    best_energy = new_energy
                    best_pts = new_pts
        except Exception:
            pass
    
    # Ensure final output is on unit sphere
    best_pts = _project_to_sphere(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
