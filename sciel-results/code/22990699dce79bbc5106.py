"""Optimized logarithmic energy minimizer for 150 points on the sphere."""
N = 150
import numpy as np
import time
import math


def _energy_and_grad(pts):
    """Compute logarithmic energy and gradient for points on unit sphere.
    
    Energy: sum_{i<j} log(1/|d_ij|) = -sum_{i<j} log|d_ij|
    Gradient w.r.t. x_i: -sum_{j!=i} (x_i - x_j)/|x_i - x_j|^2
    """
    n = pts.shape[0]
    # Compute all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dist_sq = np.sum(diff * diff, axis=2)  # (n, n)
    np.fill_diagonal(dist_sq, 1.0)  # avoid division by zero
    
    dist = np.sqrt(dist_sq)
    
    # Energy: -sum_{i<j} log(dist_ij)
    # Use upper triangle only
    iu = np.triu_indices(n, k=1)
    log_dists = np.log(dist[iu])
    energy = -np.sum(log_dists)
    
    # Gradient: for each i, grad_i = -sum_{j!=i} (x_i - x_j)/|x_i - x_j|^2
    # diff[i,j] = x_i - x_j, dist_sq[i,j] = |x_i - x_j|^2
    # grad_i = -sum_j diff[i,j] / dist_sq[i,j]
    inv_dist_sq = 1.0 / dist_sq  # (n, n), diagonal is 1 but we'll zero it
    np.fill_diagonal(inv_dist_sq, 0.0)
    grad = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)  # (n, 3)
    
    return energy, grad


def _project_tangent(pts, grad):
    """Project gradient onto tangent space of sphere at each point."""
    # For unit sphere, tangent projection: g_tan = g - (g·x)x
    dot = np.sum(pts * grad, axis=1, keepdims=True)
    return grad - dot * pts


def _fibonacci_init():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _greedy_init(rng):
    """Greedy farthest-point sampling."""
    pts = np.zeros((N, 3))
    pts[0] = rng.standard_normal(3)
    pts[0] /= np.linalg.norm(pts[0])
    
    for i in range(1, N):
        # Find point that maximizes minimum distance to existing points
        best_idx = 0
        best_min_dist = -1
        # Sample candidate points
        n_candidates = 2000
        candidates = rng.standard_normal((n_candidates, 3))
        candidates /= np.linalg.norm(candidates, axis=1, keepdims=True)
        
        # For each candidate, compute min distance to existing points
        existing = pts[:i]
        # dist_sq[cand, existing]
        diff = candidates[:, np.newaxis, :] - existing[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        min_dists = np.min(dist_sq, axis=1)
        best_idx = np.argmax(min_dists)
        pts[i] = candidates[best_idx]
    
    return pts


def _random_init(rng):
    pts = rng.standard_normal((N, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def _optimize(pts, time_limit, rng):
    """Gradient-based optimization with adaptive step size."""
    n = pts.shape[0]
    best_energy = _energy_and_grad(pts)[0]
    best_pts = pts.copy()
    
    # Adaptive gradient descent
    lr = 1e-4
    min_lr = 1e-8
    max_lr = 1e-2
    patience = 0
    max_patience = 5000
    
    start = time.time()
    
    while time.time() - start < time_limit:
        energy, grad = _energy_and_grad(pts)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
            patience = 0
        else:
            patience += 1
        
        if patience > max_patience:
            break
        
        # Project gradient to tangent space
        grad_tan = _project_tangent(pts, grad)
        
        # Take step
        pts_new = pts - lr * grad_tan
        
        # Renormalize to unit sphere
        norms = np.linalg.norm(pts_new, axis=1, keepdims=True)
        pts_new = pts_new / norms
        
        # Check validity
        if not np.all(np.isfinite(pts_new)):
            lr *= 0.5
            continue
        
        # Check for too-close points
        diff = pts_new[:, np.newaxis, :] - pts_new[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, 1.0)
        if np.min(dist_sq) < 1e-18:
            lr *= 0.5
            continue
        
        pts = pts_new
        
        # Adapt learning rate based on energy change
        energy_new, _ = _energy_and_grad(pts)
        if energy_new < energy:
            # Good step, try slightly larger lr
            lr = min(lr * 1.05, max_lr)
        else:
            # Bad step, reduce lr
            lr = max(lr * 0.5, min_lr)
            if lr <= min_lr:
                break
    
    return best_pts, best_energy


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_energy = float('inf')
    
    # Multiple initialization strategies
    inits = []
    inits.append(('fib', _fibonacci_init()))
    inits.append(('greedy', _greedy_init(rng)))
    inits.append(('random1', _random_init(rng)))
    inits.append(('random2', _random_init(rng)))
    
    # Also try perturbed Fibonacci
    for scale in [0.01, 0.05, 0.1]:
        fib = _fibonacci_init()
        pert = fib + scale * rng.standard_normal((N, 3))
        pert /= np.linalg.norm(pert, axis=1, keepdims=True)
        inits.append((f'fib_pert_{scale}', pert))
    
    # Allocate time budget
    n_inits = len(inits)
    time_per_init = min(15.0, 120.0 / n_inits)
    
    for name, init_pts in inits:
        if time.time() > deadline - 5.0:
            break
        
        # Quick evaluation of initial energy
        e0, _ = _energy_and_grad(init_pts)
        
        # Optimize
        opt_pts, opt_energy = _optimize(init_pts, time_per_init, rng)
        
        if opt_energy < best_energy:
            best_energy = opt_energy
            best_pts = opt_pts.copy()
    
    # Final polish: run one more optimization with remaining time on best result
    remaining = deadline - 5.0 - time.time()
    if remaining > 2.0 and best_pts is not None:
        final_pts, final_energy = _optimize(best_pts, remaining, rng)
        if final_energy < best_energy:
            best_energy = final_energy
            best_pts = final_pts
    
    if best_pts is None:
        best_pts = _fibonacci_init()
    
    # Ensure all points are on unit sphere and valid
    best_pts /= np.linalg.norm(best_pts, axis=1, keepdims=True)
    
    return best_pts
