"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math
import time

N = 100

# EVOLVE-BLOCK-START
from scipy.optimize import minimize


def _log_energy(pts):
    """Compute log(1/d) energy sum over all pairs using chordal distances."""
    # pts: (N, 3)
    # Compute pairwise distances
    diff = pts[:, None, :] - pts[None, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    # Get upper triangle indices
    iu = np.triu_indices(N, k=1)
    d_sq = dist_sq[iu]
    # Clip to avoid log(0) and ensure positive
    d_sq = np.clip(d_sq, 1e-30, None)
    d = np.sqrt(d_sq)
    # Sum log(1/d) = -sum(log(d))
    # Using math.fsum for precision
    terms = [-math.log(di) for di in d]
    return math.fsum(terms)


def _log_energy_and_tangent_grad(pts):
    """Compute log(1/d) energy and tangential gradient on the sphere."""
    # pts: (N, 3)
    diff = pts[:, None, :] - pts[None, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Avoid zero distances
    dist_sq = np.clip(dist_sq, 1e-30, None)
    dist = np.sqrt(dist_sq)
    
    # Energy: sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    iu = np.triu_indices(N, k=1)
    d = dist[iu]
    energy = -math.fsum([math.log(di) for di in d])
    
    # Gradient in R^3: grad_i = sum_{j != i} (p_j - p_i) / d_ij^2
    # Note: d_ij^2 = dist_sq[i,j]
    inv_dist_sq = np.zeros_like(dist_sq)
    mask = ~np.eye(N, dtype=bool)
    inv_dist_sq[mask] = 1.0 / dist_sq[mask]
    
    # grad_i = sum_j inv_dist_sq[i,j] * (p_j - p_i)
    # = sum_j inv_dist_sq[i,j] * p_j - p_i * sum_j inv_dist_sq[i,j]
    
    weighted_sum = inv_dist_sq @ pts
    row_sums = inv_dist_sq.sum(axis=1)
    
    # Euclidean gradient
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


def _optimize_sphere(x0_flat, maxiter=500, ftol=1e-12, gtol=1e-10, deadline=None):
    """Optimize on sphere using L-BFGS-B with tangential gradient.
    
    We parameterize by the 3N coordinates. The objective function projects
    to the sphere. The gradient passed is the tangential gradient.
    """
    def objective(x):
        pts = x.reshape(N, 3)
        pts = _project_to_sphere(pts)
        e, g = _log_energy_and_tangent_grad(pts)
        return e, g.flatten()
    
    try:
        # Check time if deadline is provided
        if deadline and time.time() > deadline:
            return None, None

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
    # 120s limit, leave 10s safety margin
    deadline = start_time + 110
    
    # Start with Fibonacci spiral
    pts = _initial_fibonacci()
    best_pts = pts.copy()
    best_energy = _log_energy(pts)
    
    # 1. Optimization from Fibonacci start
    if time.time() < deadline - 10:
        x0 = pts.flatten()
        result_pts, result_energy = _optimize_sphere(x0, maxiter=500, ftol=1e-14, gtol=1e-12, deadline=deadline)
        if result_pts is not None:
            actual_energy = _log_energy(result_pts)
            if actual_energy < best_energy:
                best_pts = result_pts
                best_energy = actual_energy
    
    # 2. Try random restarts if time permits
    rng = np.random.default_rng(42)
    # We can afford a few restarts. Each restart takes a few seconds.
    # Let's try 3-5 restarts.
    num_restarts = 5
    for restart in range(num_restarts):
        if time.time() >= deadline - 15:
            break
        
        # Random initial points on sphere
        random_pts = rng.standard_normal((N, 3))
        random_pts = random_pts / np.linalg.norm(random_pts, axis=1, keepdims=True)
        
        # Optimize
        x0 = random_pts.flatten()
        # Use slightly fewer iterations for restarts to allow more attempts
        result_pts, result_energy = _optimize_sphere(x0, maxiter=300, ftol=1e-12, gtol=1e-10, deadline=deadline)
        if result_pts is not None:
            actual_energy = _log_energy(result_pts)
            if actual_energy < best_energy:
                best_pts = result_pts
                best_energy = actual_energy
    
    # 3. Final polish: run optimizer again on the best point with tight tolerances
    if time.time() < deadline - 5:
        x0 = best_pts.flatten()
        # Tighter tolerances for final polish
        result_pts, result_energy = _optimize_sphere(x0, maxiter=200, ftol=1e-15, gtol=1e-14, deadline=deadline)
        if result_pts is not None:
            actual_energy = _log_energy(result_pts)
            if actual_energy < best_energy:
                best_pts = result_pts
                best_energy = actual_energy

    # Final projection to ensure unit sphere
    best_pts = _project_to_sphere(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
