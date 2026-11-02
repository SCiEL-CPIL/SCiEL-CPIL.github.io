"""
Minimizing logarithmic energy of 150 points on the unit sphere.

Strategy:
1. Start from a good initial configuration (Fibonacci spiral + a few Lloyd/relaxation steps).
2. Perform gradient-based optimization using the analytic gradient of the log-energy.
3. Use adaptive step size with backtracking line search to ensure descent.
4. Periodically re-randomize if stuck in local minima (multi-start with best kept).
5. Final high-precision polish with tight tolerances.
6. Deadline guard: return best result before 145s.
"""
import numpy as np
import time
import math

N = 150
DEADLINE_SECONDS = 145.0  # safety margin before 150s kill


def _log_energy_and_grad(pts):
    """
    Compute logarithmic energy E = sum_{i<j} log(1 / |x_i - x_j|)
    and its gradient w.r.t. pts.
    
    E = -sum_{i<j} log(|x_i - x_j|)
    
    dE/dx_k = sum_{j != k} (x_k - x_j) / |x_k - x_j|^2
    (because d/dx_k [-log(|x_k - x_j|)] = -(x_k - x_j)/|x_k - x_j|^2, 
     but we have -log so it's +(x_k - x_j)/|x_k - x_j|^2... let me re-derive)
    
    Actually: E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    
    dE/dx_k = -sum_{j != k} d/dx_k log(|x_k - x_j|)
            = -sum_{j != k} (x_k - x_j) / |x_k - x_j|^2
    
    Wait: d/dx log(|x - y|) = (x-y)/|x-y|^2. So:
    dE/dx_k = -sum_{j != k} (x_k - x_j)/|x_k - x_j|^2
    """
    pts = np.asarray(pts)
    n = pts.shape[0]
    # Compute pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dist_sq = np.sum(diff * diff, axis=2)  # (n, n)
    dist = np.sqrt(dist_sq)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist, 1.0)
    np.fill_diagonal(dist_sq, 1.0)
    
    # Energy: sum over i<j of -log(dist_ij)
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient: dE/dx_k = -sum_{j != k} (x_k - x_j) / |x_k - x_j|^2
    # = sum_{j != k} (x_j - x_k) / |x_k - x_j|^2
    # diff[k,j] = x_k - x_j
    # So grad[k] = -sum_j diff[k,j] / dist_sq[k,j]  (j != k)
    # = sum_j (x_j - x_k) / dist_sq[k,j]
    
    # Weight matrix: W[k,j] = 1/dist_sq[k,j] for k != j, 0 on diagonal
    W = np.zeros_like(dist_sq)
    np.fill_diagonal(W, 0.0)
    W = 1.0 / dist_sq
    
    # grad[k] = sum_j W[k,j] * (x_j - x_k) = sum_j W[k,j]*x_j - x_k * sum_j W[k,j]
    # = W @ pts - pts * np.sum(W, axis=1)[:, np.newaxis]
    grad = W @ pts - pts * np.sum(W, axis=1)[:, np.newaxis]
    
    return energy, grad


def _project_to_sphere(pts):
    """Normalize each point to unit length."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _fibonacci_spiral(n):
    """Generate n points on unit sphere using golden-angle spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_on_sphere(n, rng):
    """Generate n random points uniformly on unit sphere."""
    pts = rng.standard_normal((n, 3))
    return _project_to_sphere(pts)


def _gradient_descent(pts, max_iter=5000, tol=1e-12, time_limit=None, start_time=None):
    """
    Gradient descent on the sphere with re-projection after each step.
    Uses adaptive step size with backtracking.
    """
    pts = _project_to_sphere(np.array(pts, dtype=np.float64))
    if start_time is None:
        start_time = time.time()
    
    best_energy, best_pts = _log_energy_and_grad(pts)
    best_pts = pts.copy()
    
    lr = 0.01
    for it in range(max_iter):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        
        energy, grad = _log_energy_and_grad(pts)
        
        # Project gradient to tangent space
        grad_tangent = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
        
        # Backtracking line search
        alpha = lr
        for _ in range(30):
            new_pts = pts + alpha * grad_tangent
            new_pts = _project_to_sphere(new_pts)
            new_energy, _ = _log_energy_and_grad(new_pts)
            if new_energy < energy:
                break
            alpha *= 0.5
        else:
            # Couldn't find a descent step; reduce learning rate
            lr *= 0.5
            if lr < 1e-10:
                break
            continue
        
        pts = new_pts
        energy = new_energy
        
        # Track best
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
        
        # Check convergence
        if it > 10 and abs(prev_energy - energy) < tol:
            break
        prev_energy = energy
        
        # Adapt learning rate
        if it % 100 == 0 and it > 0:
            # Slightly increase if we've been making good progress
            pass
    
    return best_pts, best_energy


def _lloyd_relax(pts, n_iter=50):
    """
    Simple relaxation: move each point away from its nearest neighbors.
    This helps escape bad local configurations.
    """
    pts = _project_to_sphere(np.array(pts, dtype=np.float64))
    for _ in range(n_iter):
        # Compute pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        
        # For each point, compute force away from close neighbors
        # Use inverse distance weighting
        inv_dist = np.zeros_like(dist_sq)
        mask = dist_sq < 1.0
        inv_dist[mask] = 1.0 / np.sqrt(dist_sq[mask])
        np.fill_diagonal(inv_dist, 0.0)
        
        force = inv_dist @ (-pts)  # sum_j inv_dist_ij * (-x_j) ... 
        # Actually: force on i = sum_j inv_dist_ij * (x_i - x_j)
        force = pts * np.sum(inv_dist, axis=1)[:, np.newaxis] - inv_dist @ pts
        
        # Move points
        pts = pts + 0.01 * force
        pts = _project_to_sphere(pts)
    
    return pts


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_energy = np.inf
    
    # Multi-start strategy
    n_starts = 20
    
    for start_idx in range(n_starts):
        if time.time() > deadline - 30:  # Need at least 30s for final polish
            break
        
        if start_idx == 0:
            # Start with Fibonacci spiral
            pts = _fibonacci_spiral(N)
        elif start_idx == 1:
            # Fibonacci + small random perturbation
            pts = _fibonacci_spiral(N)
            pts += 0.01 * rng.standard_normal((N, 3))
            pts = _project_to_sphere(pts)
        elif start_idx < 5:
            # Fibonacci with larger perturbation
            pts = _fibonacci_spiral(N)
            pts += 0.05 * rng.standard_normal((N, 3))
            pts = _project_to_sphere(pts)
        else:
            # Random starts
            pts = _random_on_sphere(N, rng)
        
        # Quick Lloyd relaxation to get out of obviously bad configs
        pts = _lloyd_relax(pts, n_iter=20)
        
        # Gradient descent
        remaining = deadline - time.time() - 10  # Reserve 10s for potential final polish
        if remaining < 5:
            break
        
        pts, energy = _gradient_descent(
            pts, 
            max_iter=3000, 
            tol=1e-12,
            time_limit=min(remaining, 60.0),
            start_time=time.time()
        )
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
        
        # Check if we're getting close to the known best
        if best_energy < -2364.6:
            # Do a final high-precision polish
            if time.time() < deadline - 5:
                best_pts, best_energy = _gradient_descent(
                    best_pts,
                    max_iter=10000,
                    tol=1e-14,
                    time_limit=min(deadline - time.time() - 2, 30.0),
                    start_time=time.time()
                )
            break
    
    # Final polish with very tight tolerance
    if best_pts is not None and time.time() < deadline - 2:
        remaining = deadline - time.time() - 1
        if remaining > 5:
            best_pts, best_energy = _gradient_descent(
                best_pts,
                max_iter=20000,
                tol=1e-14,
                time_limit=remaining,
                start_time=time.time()
            )
    
    if best_pts is None:
        # Fallback: Fibonacci spiral
        best_pts = _fibonacci_spiral(N)
    
    # Final normalization and validation
    best_pts = _project_to_sphere(best_pts)
    
    # Ensure no points are too close
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    min_dist = np.sqrt(np.min(dist_sq))
    if min_dist < 1e-9:
        # Add tiny perturbation to separate close points
        best_pts += 1e-8 * rng.standard_normal((N, 3))
        best_pts = _project_to_sphere(best_pts)
    
    return best_pts
