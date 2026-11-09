"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize, L_BFGS_B
from scipy.spatial.distance import cdist
import sys


def _log_energy(pts):
    """Compute logarithmic energy sum_{i<j} log(1/d_ij) using math.fsum."""
    n = len(pts)
    # Compute all pairwise distances
    # Use efficient vectorized distance computation
    # dist^2 = |a|^2 + |b|^2 - 2 a.b ; for unit sphere |a|=|b|=1 so dist^2 = 2 - 2 a.b
    # But let's just compute directly for clarity and correctness
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-15:
                return -1e18  # invalid
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)


def _log_energy_grad(pts):
    """Compute logarithmic energy and its gradient."""
    n = len(pts)
    energy = 0.0
    grad = np.zeros_like(pts)
    
    # For unit sphere points, d_ij = |x_i - x_j|
    # d_ij^2 = 2 - 2 * x_i . x_j
    # E = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    # dE/dx_i = -sum_{j != i} d/dx_i log(d_ij) = -sum_{j != i} (1/d_ij) * (x_i - x_j)/d_ij
    #          = -sum_{j != i} (x_i - x_j) / d_ij^2
    #          = -sum_{j != i} (x_i - x_j) / (2 - 2 * x_i . x_j)
    
    for i in range(n):
        for j in range(i + 1, n):
            diff = pts[i] - pts[j]
            d2 = np.dot(diff, diff)
            if d2 < 1e-18:
                return -1e18, grad
            d = math.sqrt(d2)
            energy -= math.log(d)
            # Gradient contribution for point i: -(x_i - x_j)/d^2
            # Gradient contribution for point j: +(x_i - x_j)/d^2  (since d/dx_j log(d_ij) = (x_j-x_i)/d^2, so -d/dx_j = (x_i-x_j)/d^2)
            factor = 1.0 / d2
            grad[i] -= diff * factor
            grad[j] += diff * factor
    
    return energy, grad


def _project_to_sphere(pts):
    """Project points back onto the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin
    
    # Initial guess: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # First, do a quick local optimization with L-BFGS-B
    best_pts = pts.copy()
    best_energy = _log_energy(pts)
    
    # Use scipy's minimize with gradient
    def objective_and_grad(x):
        pts_2d = x.reshape(-1, 3)
        e, g = _log_energy_grad(pts_2d)
        return e, g.flatten()
    
    try:
        res = minimize(
            objective_and_grad,
            pts.flatten(),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 200, 'gtol': 1e-10, 'ftol': 1e-12}
        )
        if res.x is not None and np.all(np.isfinite(res.x)):
            candidate = _project_to_sphere(res.x.reshape(-1, 3))
            cand_energy = _log_energy(candidate)
            if cand_energy < best_energy:
                best_pts = candidate
                best_energy = cand_energy
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline:
        return best_pts
    
    # Now do multi-start with random perturbations and local optimization
    rng = np.random.default_rng(42)
    
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Random perturbation of the best point
        noise_scale = 0.1 / (1.0 + restart * 0.5)
        perturbed = best_pts + rng.normal(0, noise_scale, size=(N, 3))
        perturbed = _project_to_sphere(perturbed)
        
        try:
            res = minimize(
                objective_and_grad,
                perturbed.flatten(),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 100, 'gtol': 1e-10, 'ftol': 1e-12}
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                candidate = _project_to_sphere(res.x.reshape(-1, 3))
                cand_energy = _log_energy(candidate)
                if cand_energy < best_energy:
                    best_pts = candidate
                    best_energy = cand_energy
        except Exception:
            continue
    
    # Final projection to ensure unit sphere
    best_pts = _project_to_sphere(best_pts)
    
    # Verify no points are too close
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                # Add small perturbation
                best_pts[j] += rng.normal(0, 1e-8, size=3)
                best_pts[j] /= np.linalg.norm(best_pts[j])
    
    return best_pts
# EVOLVE-BLOCK-END
