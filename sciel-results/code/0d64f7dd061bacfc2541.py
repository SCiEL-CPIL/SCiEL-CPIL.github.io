"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150
TIME_LIMIT = 145.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START

def _log_energy_and_grad(points):
    """
    Compute the logarithmic energy and its gradient.
    Energy: E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    """
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, 1.0)
    
    dists_sq_inv = 1.0 / dists_sq
    grad_raw = -np.sum(diff * dists_sq_inv[:, :, np.newaxis], axis=1)
    
    iu = np.triu_indices(n, k=1)
    log_dists_sq = np.log(dists_sq[iu])
    energy = -0.5 * np.sum(log_dists_sq)
    
    return energy, grad_raw

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _fibonacci_sphere(n, seed=0.0):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * (i + seed)
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _local_minimize(points, deadline):
    """
    Perform local minimization using L-BFGS-B with a penalty method for the sphere constraint.
    Returns the optimized points.
    """
    n = len(points)
    x0 = points.flatten()
    
    def objective_and_grad(x_flat):
        x = x_flat.reshape(n, 3)
        energy, grad_euc = _log_energy_and_grad(x)
        
        # Penalty for deviation from unit sphere
        norms_sq = np.sum(x * x, axis=1)
        penalty = np.sum((norms_sq - 1.0) ** 2)
        grad_penalty = 4.0 * x * (norms_sq - 1.0)[:, np.newaxis]
        
        mu = 50.0 # Strong penalty to keep points on sphere
        total_energy = energy + mu * penalty
        total_grad = grad_euc + mu * grad_penalty
        
        return total_energy, total_grad.flatten()

    try:
        res = minimize(
            objective_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-7, 'maxfun': 500}
        )
        
        if time.time() > deadline:
            return points

        x_opt = res.x.reshape(n, 3)
        x_opt = _normalize(x_opt)
        
        # Check validity: points must be distinct
        min_dist = np.min(np.linalg.norm(x_opt[:, np.newaxis, :] - x_opt[np.newaxis, :, :], axis=-1)[np.triu_indices(n, k=1)])
        if min_dist < 1e-9:
            return points
            
        return x_opt
    except Exception:
        return points

def _kick(points, kick_size, rng):
    """
    Apply a random perturbation to the points on the sphere.
    kick_size is the approximate angular displacement.
    """
    # Generate random directions
    kick_vecs = rng.standard_normal((len(points), 3))
    # Scale by kick size
    kick_vecs *= kick_size
    # Add to current points
    new_points = points + kick_vecs
    # Normalize back to sphere
    return _normalize(new_points)

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

    # 1. Initialize with Fibonacci sphere
    best_points = _fibonacci_sphere(N)
    best_energy, _ = _log_energy_and_grad(best_points)
    
    # Local minimize the initial guess
    best_points = _local_minimize(best_points, deadline)
    best_energy, _ = _log_energy_and_grad(best_points)
    
    rng = np.random.default_rng(12345)
    
    # 2. Basin Hopping Strategy
    # We will perform a series of kicks and local minimizations.
    # The kick size will start small and increase if stuck, or we can use a fixed moderate size.
    # For N=150, the points are close together. A kick of 0.1-0.5 radians is reasonable.
    
    current_kick_size = 0.2
    no_improve_count = 0
    max_no_improve = 20
    
    # Estimate time per iteration
    # Local minimize takes ~1-2s. Kick is fast.
    # We have ~140s. So we can do ~50-70 iterations.
    
    while time.time() < deadline - 5.0:
        # Adaptive kick size: if stuck, increase kick size to escape basin
        if no_improve_count > max_no_improve:
            current_kick_size *= 1.5
            no_improve_count = 0
            # Optionally, if kick size gets too large, reset to a new random start
            if current_kick_size > 2.0:
                # Reset to a new random start to explore a completely different basin
                best_points = _random_sphere(N, rng)
                current_kick_size = 0.2
                no_improve_count = 0
                continue

        # Generate kicked points
        kicked_points = _kick(best_points, current_kick_size, rng)
        
        # Local minimize the kicked points
        new_points = _local_minimize(kicked_points, deadline)
        
        # Check validity and energy
        try:
            new_energy, _ = _log_energy_and_grad(new_points)
            # Check for close points
            min_dist = np.min(np.linalg.norm(new_points[:, np.newaxis, :] - new_points[np.newaxis, :, :], axis=-1)[np.triu_indices(N, k=1)])
            if min_dist < 1e-9:
                continue
        except Exception:
            continue

        if new_energy < best_energy:
            best_energy = new_energy
            best_points = new_points
            no_improve_count = 0
            # Decrease kick size slightly if we found a better point, to refine
            current_kick_size *= 0.9
        else:
            no_improve_count += 1

    # Final polish: one more local minimize on the best point found
    if time.time() < deadline - 2.0:
        final_points = _local_minimize(best_points, deadline)
        try:
            final_energy, _ = _log_energy_and_grad(final_points)
            min_dist = np.min(np.linalg.norm(final_points[:, np.newaxis, :] - final_points[np.newaxis, :, :], axis=-1)[np.triu_indices(N, k=1)])
            if min_dist >= 1e-9 and final_energy < best_energy:
                best_points = final_points
        except Exception:
            pass

    return _normalize(best_points)

# EVOLVE-BLOCK-END
