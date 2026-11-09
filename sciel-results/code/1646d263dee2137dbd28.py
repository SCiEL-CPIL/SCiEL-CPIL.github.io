"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math
import time

N = 100

# EVOLVE-BLOCK-START
from scipy.optimize import minimize


def _log_energy(pts):
    """Compute log(1/d) energy sum over all pairs using chordal distances."""
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    iu = np.triu_indices(N, k=1)
    d = dist[iu]
    d = np.clip(d, 1e-15, None)
    terms = [-math.log(di) for di in d]
    return math.fsum(terms)


def _log_energy_and_tangent_grad(pts):
    """Compute log(1/d) energy and tangential gradient on the sphere."""
    diff = pts[:, None, :] - pts[None, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    dist = np.sqrt(dist_sq)
    
    dist = np.clip(dist, 1e-15, None)
    dist_sq = np.clip(dist_sq, 1e-30, None)
    
    iu = np.triu_indices(N, k=1)
    d = dist[iu]
    energy = -math.fsum([math.log(di) for di in d])
    
    # Gradient in R^3: grad_i = sum_{j != i} (p_j - p_i) / d_ij^2
    # = sum_{j != i} -diff[i,j] / d_ij^2
    inv_dist_sq = np.zeros_like(dist_sq)
    mask = ~np.eye(N, dtype=bool)
    inv_dist_sq[mask] = 1.0 / dist_sq[mask]
    
    # grad_i = -sum_j inv_dist_sq[i,j] * diff[i,j]
    # diff[i,j] = p_i - p_j, so -diff[i,j] = p_j - p_i
    # grad_i = sum_j inv_dist_sq[i,j] * (p_j - p_i)
    # = (inv_dist_sq @ pts) - p_i * sum_j inv_dist_sq[i,j]
    
    weighted_sum = inv_dist_sq @ pts
    row_sums = inv_dist_sq.sum(axis=1)
    grad_r3 = weighted_sum - pts * row_sums[:, None]
    
    # Project to tangent space: grad_tan = grad_r3 - (grad_r3 · p_i) p_i
    # Since pts are on unit sphere, p_i · p_i = 1
    dot = np.sum(grad_r3 * pts, axis=1, keepdims=True)
    grad_tan = grad_r3 - dot * pts
    
    return energy, grad_tan


def _project_to_sphere(pts):
    """Project points to the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-15, None)
    return pts / norms


def _initial_fibonacci():
    """Golden-angle (Fibonacci) spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _optimize_sphere(x0_flat, maxiter=1000, ftol=1e-15, gtol=1e-12):
    """Optimize on sphere using L-BFGS-B with tangential gradient.
    
    We parameterize by the 3N coordinates but project to sphere after each step.
    The gradient passed to the optimizer is the tangential gradient.
    Tighter tolerances for high-precision polish.
    """
    def objective(x):
        pts = x.reshape(N, 3)
        pts = _project_to_sphere(pts)
        e, g = _log_energy_and_tangent_grad(pts)
        return e, g.flatten()
    
    try:
        result = minimize(
            objective,
            x0_flat,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol, 'maxfun': maxiter * 10}
        )
        if result.fun is not None and np.isfinite(result.fun):
            result_pts = _project_to_sphere(result.x.reshape(N, 3))
            return result_pts, result.fun
    except Exception:
        pass
    
    return None, None


def construct():
    """Construct 100 points on the unit sphere to minimize logarithmic energy."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    # Start with Fibonacci spiral
    pts = _initial_fibonacci()
    best_pts = pts.copy()
    best_energy = _log_energy(pts)
    
    # Optimization from Fibonacci start with tight tolerances
    if time.time() < deadline - 10:
        x0 = pts.flatten()
        result_pts, result_energy = _optimize_sphere(x0, maxiter=1000)
        if result_pts is not None:
            actual_energy = _log_energy(result_pts)
            if actual_energy < best_energy:
                best_pts = result_pts
                best_energy = actual_energy
    
    # Try a few random restarts if time permits
    rng = np.random.default_rng(42)
    for restart in range(5):
        if time.time() >= deadline - 10:
            break
        
        random_pts = rng.standard_normal((N, 3))
        random_pts = random_pts / np.linalg.norm(random_pts, axis=1, keepdims=True)
        
        x0 = random_pts.flatten()
        # Use slightly fewer iterations for restarts to save time for polish
        result_pts, result_energy = _optimize_sphere(x0, maxiter=500)
        if result_pts is not None:
            actual_energy = _log_energy(result_pts)
            if actual_energy < best_energy:
                best_pts = result_pts
                best_energy = actual_energy
                
    # Final high-precision polish on the best point found
    if time.time() < deadline - 5:
        x0 = best_pts.flatten()
        result_pts, result_energy = _optimize_sphere(x0, maxiter=1000, ftol=1e-16, gtol=1e-13)
        if result_pts is not None:
            actual_energy = _log_energy(result_pts)
            if actual_energy < best_energy:
                best_pts = result_pts
                best_energy = actual_energy

    # Final projection to ensure unit sphere
    best_pts = _project_to_sphere(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
