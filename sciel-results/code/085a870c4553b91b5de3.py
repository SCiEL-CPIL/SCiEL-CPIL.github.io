"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 100
DEADLINE = 115.0  # safety margin before 120s kill

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math


def _log_energy_and_grad(pts):
    """Compute logarithmic energy and its gradient.
    
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Gradient w.r.t. x_i = -sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    """
    # pts shape (N, 3), assumed on unit sphere
    # Compute pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 3)
    dist = np.sqrt(np.sum(diff * diff, axis=2))  # (N, N)
    
    # Avoid division by zero (diagonal)
    np.fill_diagonal(dist, 1.0)
    
    # Energy: sum over i<j of -log(dist_ij)
    # Use upper triangle
    iu = np.triu_indices(N, k=1)
    energy = -np.sum(np.log(dist[iu[0], iu[1]]))
    
    # Gradient: for each i, grad_i = -sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    # diff[i,j] = x_i - x_j
    # grad_i = -sum_j diff[i,j] / dist[i,j]^2
    inv_dist_sq = 1.0 / (dist * dist)  # (N, N)
    # Set diagonal to 0
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # grad[i] = -sum_j diff[i,j] * inv_dist_sq[i,j]
    grad = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)  # (N, 3)
    
    return energy, grad


def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.sqrt(np.sum(pts * pts, axis=1, keepdims=True))
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _golden_spiral():
    """Golden-angle (Fibonacci) spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_start(seed):
    """Generate random points on the sphere."""
    rng = np.random.RandomState(seed)
    theta = rng.uniform(0, 2 * np.pi, N)
    phi = np.arccos(rng.uniform(-1, 1, N))
    pts = np.column_stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ])
    return pts


def _optimize_from_start(pts0, deadline):
    """Run scipy optimization from a starting point, respecting deadline."""
    best_pts = pts0.copy()
    best_energy = _log_energy_and_grad(pts0)[0]
    
    # First pass: L-BFGS-B with reasonable tolerances
    t_start = time.time()
    try:
        res = minimize(
            fun=lambda x: _log_energy_and_grad(x)[0],
            x0=pts0.ravel(),
            jac=lambda x: _log_energy_and_grad(x)[1].ravel(),
            method='L-BFGS-B',
            options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
        )
        if res.x is not None and np.all(np.isfinite(res.x)):
            cand = _project_to_sphere(res.x.reshape(N, 3))
            e = _log_energy_and_grad(cand)[0]
            if e < best_energy and np.all(np.isfinite(cand)):
                best_pts = cand
                best_energy = e
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_pts, best_energy
    
    # Second pass: tighter tolerances for polish
    t_start2 = time.time()
    try:
        res2 = minimize(
            fun=lambda x: _log_energy_and_grad(x)[0],
            x0=best_pts.ravel(),
            jac=lambda x: _log_energy_and_grad(x)[1].ravel(),
            method='L-BFGS-B',
            options={'maxiter': 10000, 'ftol': 1e-18, 'gtol': 1e-14, 'maxls': 50}
        )
        if res2.x is not None and np.all(np.isfinite(res2.x)):
            cand = _project_to_sphere(res2.x.reshape(N, 3))
            e = _log_energy_and_grad(cand)[0]
            if e < best_energy and np.all(np.isfinite(cand)):
                best_pts = cand
                best_energy = e
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_pts, best_energy
    
    # Third pass: even tighter polish with very small steps
    try:
        res3 = minimize(
            fun=lambda x: _log_energy_and_grad(x)[0],
            x0=best_pts.ravel(),
            jac=lambda x: _log_energy_and_grad(x)[1].ravel(),
            method='L-BFGS-B',
            options={'maxiter': 20000, 'ftol': 1e-20, 'gtol': 1e-16, 'maxls': 30}
        )
        if res3.x is not None and np.all(np.isfinite(res3.x)):
            cand = _project_to_sphere(res3.x.reshape(N, 3))
            e = _log_energy_and_grad(cand)[0]
            if e < best_energy and np.all(np.isfinite(cand)):
                best_pts = cand
                best_energy = e
    except Exception:
        pass
    
    return best_pts, best_energy


def _refine_with_momentum(pts0, deadline):
    """Custom momentum-based gradient descent with Armijo line search on sphere."""
    best_pts = pts0.copy()
    best_energy = _log_energy_and_grad(pts0)[0]
    
    # Flatten for optimization
    x = best_pts.ravel().copy()
    v = np.zeros_like(x)  # momentum
    lr = 0.01
    beta = 0.9
    
    for iteration in range(50000):
        if time.time() > deadline:
            break
        
        pts = _project_to_sphere(x.reshape(N, 3))
        energy, grad = _log_energy_and_grad(pts)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
        
        # Project gradient onto tangent space
        x_flat = pts.ravel()
        # For each point i, the constraint is |x_i|^2 = 1
        # Tangent projection: g_tan = g - (g . x_i) * x_i for each i
        grad_3d = grad.reshape(N, 3)
        x_3d = pts.reshape(N, 3)
        # g . x_i for each i
        dot = np.sum(grad_3d * x_3d, axis=1, keepdims=True)  # (N, 1)
        grad_tan = grad_3d - dot * x_3d  # (N, 3)
        grad_tan_flat = grad_tan.ravel()
        
        # Momentum update
        v = beta * v - lr * grad_tan_flat
        x_new = x + v
        
        # Project back to sphere
        x_new = _project_to_sphere(x_new.reshape(N, 3)).ravel()
        
        # Armijo line search: check if new energy is acceptable
        new_energy = _log_energy_and_grad(_project_to_sphere(x_new.reshape(N, 3)))[0]
        
        if new_energy < energy - 1e-4 * lr * np.sum(grad_tan_flat * grad_tan_flat):
            x = x_new
        else:
            # Halve step size
            v = 0.5 * v
            x = x + v
            x = _project_to_sphere(x.reshape(N, 3)).ravel()
        
        # Decay learning rate slowly
        lr = max(0.001, lr * 0.9999)
    
    final_pts = _project_to_sphere(best_pts)
    final_energy = _log_energy_and_grad(final_pts)[0]
    return final_pts, final_energy


def construct():
    """Construct 100 points on the unit sphere minimizing logarithmic energy."""
    global DEADLINE
    deadline = DEADLINE
    
    # Start with golden spiral
    best_pts = _golden_spiral()
    best_energy = _log_energy_and_grad(best_pts)[0]
    
    # Try multiple random starts and keep the best
    t_start = time.time()
    n_random_starts = 5
    
    for seed in range(n_random_starts):
        if time.time() > deadline - 10:
            break
        
        pts0 = _random_start(seed)
        pts, energy = _optimize_from_start(pts0, deadline - 5)
        
        if energy < best_energy:
            best_pts = pts
            best_energy = energy
    
    # Also try optimizing from the golden spiral with tighter settings
    if time.time() < deadline - 5:
        pts, energy = _optimize_from_start(_golden_spiral(), deadline - 5)
        if energy < best_energy:
            best_pts = pts
            best_energy = energy
    
    # Final refinement with momentum method if time permits
    if time.time() < deadline - 3:
        pts, energy = _refine_with_momentum(best_pts, deadline - 1)
        if energy < best_energy:
            best_pts = pts
            best_energy = energy
    
    # Ensure all points are on unit sphere and finite
    best_pts = _project_to_sphere(best_pts)
    
    # Verify no points are too close
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                # Perturb slightly
                best_pts[j] += np.random.randn(3) * 1e-10
                best_pts[j] = best_pts[j] / np.linalg.norm(best_pts[j])
    
    return best_pts
# EVOLVE-BLOCK-END
