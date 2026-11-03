"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_radius_fast(pts):
    """
    Compute the packing radius for given points using vectorized operations.
    r = min(min_pair_dist / 2, min_boundary_dist)
    """
    if len(pts) < 2:
        return 0.0
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    min_boundary = np.min(1.0 - np.hypot(pts[:, 0], pts[:, 1]))
    return min(min_pair_dist / 2.0, min_boundary)


def _init_hex(spacing):
    """Generate hexagonal lattice points within unit disk."""
    k = int(1.0 / spacing) + 3
    points = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            r = np.hypot(x, y)
            if r <= 1.0 - spacing / 2.0:
                points.append((x, y))
    return np.array(points) if points else np.empty((0, 2))


def _soft_min_logsumexp(arr, beta):
    """
    Differentiable approximation to min(arr) using log-sum-exp.
    As beta -> inf, this approaches min(arr).
    """
    # For numerical stability, subtract max
    m = np.max(arr)
    return m - np.log(np.sum(np.exp(beta * (arr - m)))) / beta


def _objective_and_grad(x, beta):
    """
    Objective function and gradient for maximizing packing radius.
    We minimize -r where r is the packing radius (soft-min approximation).
    
    x: flattened array of shape (2*N,)
    beta: temperature parameter for soft-min
    Returns: (objective_value, gradient)
    """
    pts = x.reshape(-1, 2)
    n = len(pts)
    
    # Compute all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    dists = np.sqrt(dists_sq + 1e-15)
    np.fill_diagonal(dists, np.inf)
    
    # Pair distances / 2
    pair_dists = dists / 2.0
    
    # Boundary distances
    boundary_dists = 1.0 - np.hypot(pts[:, 0], pts[:, 1])
    
    # Combine all constraints
    all_dists = np.concatenate([pair_dists[np.triu_indices(n, k=1)], boundary_dists])
    
    # Soft-min approximation
    m = np.max(all_dists)
    logsumexp = m + np.log(np.sum(np.exp(beta * (all_dists - m)))) / beta
    obj = -logsumexp  # We want to maximize, so minimize negative
    
    # Gradient computation
    # d/dx of soft_min = -sum(exp(beta*(d_i - m)) / sum(exp(beta*(d_j - m))) * d(d_i)/dx)
    weights = np.exp(beta * (all_dists - m)) / np.sum(np.exp(beta * (all_dists - m)))
    
    grad = np.zeros_like(x)
    
    # Gradient from pair distances
    pair_weights = weights[:n*(n-1)//2]
    for i in range(n):
        for j in range(i+1, n):
            idx = i*(n-1) - i*(i-1)//2 + (j - i - 1)
            w = pair_weights[idx] * beta
            d_pts_i = (pts[i] - pts[j]) / (dists[i,j] + 1e-15)
            d_pts_j = (pts[j] - pts[i]) / (dists[i,j] + 1e-15)
            grad[2*i] += -w * d_pts_i[0]
            grad[2*i+1] += -w * d_pts_i[1]
            grad[2*j] += -w * d_pts_j[0]
            grad[2*j+1] += -w * d_pts_j[1]
    
    # Gradient from boundary distances
    boundary_weights = weights[n*(n-1)//2:]
    for i in range(n):
        w = boundary_weights[i] * beta
        r = np.hypot(pts[i, 0], pts[i, 1])
        if r > 1e-10:
            grad[2*i] += -w * pts[i, 0] / r
            grad[2*i+1] += -w * pts[i, 1] / r
    
    return obj, grad


def _objective_only(x, beta):
    """Objective only (for L-BFGS-B if needed)."""
    obj, _ = _objective_and_grad(x, beta)
    return obj


def _grad_only(x, beta):
    """Gradient only."""
    _, grad = _objective_and_grad(x, beta)
    return grad


def _gradient_ascent(pts, deadline, beta_start=50.0, beta_end=200.0, 
                     lr_start=0.001, lr_end=0.0001, max_iter=3000):
    """
    Gradient ascent to maximize packing radius using soft-min surrogate.
    """
    best_pts = pts.copy()
    best_r = _compute_radius_fast(best_pts)
    n = len(best_pts)
    x = best_pts.flatten()
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Anneal beta and learning rate
        progress = it / max_iter
        beta = beta_start + (beta_end - beta_start) * progress
        lr = lr_start + (lr_end - lr_start) * progress
        
        # Compute objective and gradient
        obj, grad = _objective_and_grad(x, beta)
        
        # Take gradient step
        x_new = x + lr * grad
        
        # Reshape and project back to feasible region
        pts_new = x_new.reshape(-1, 2)
        
        # Project points inside unit circle with margin
        radii = np.hypot(pts_new[:, 0], pts_new[:, 1])
        max_r = 1.0 - 0.005  # Small margin
        for i in range(n):
            if radii[i] > max_r:
                scale = max_r / radii[i]
                pts_new[i] *= scale
        
        # Compute actual radius
        r_new = _compute_radius_fast(pts_new)
        
        if r_new > best_r:
            best_pts = pts_new.copy()
            best_r = r_new
            x = x_new.copy()
        else:
            # If no improvement, reduce step size
            lr *= 0.9
            x = x_new.copy()  # Still accept the step to explore
    
    return best_pts, best_r


def construct():
    deadline = time.time() + 115.0  # Safety margin
    
    # Search over hexagonal spacings for initial configuration
    best_pts = None
    best_r = 0.0
    
    # Try a range of spacings
    for spacing in np.linspace(0.09, 0.13, 40):
        if time.time() > deadline:
            break
        pts = _init_hex(spacing)
        if len(pts) < N:
            continue
        # Take N closest to center
        dists = np.hypot(pts[:, 0], pts[:, 1])
        idx = np.argsort(dists, kind='stable')[:N]
        candidate = pts[idx]
        r = _compute_radius_fast(candidate)
        if r > best_r:
            best_r = r
            best_pts = candidate.copy()
    
    if best_pts is None:
        # Fallback: use original heuristic
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
            if len(points) >= N:
                best_pts = points[np.argsort(radius, kind="stable")[:N]]
                best_r = _compute_radius_fast(best_pts)
                break
            spacing *= 0.98
    
    # Gradient-based optimization
    if time.time() < deadline - 10.0:
        best_pts, best_r = _gradient_ascent(
            best_pts, 
            deadline=deadline - 5.0,
            beta_start=50.0,
            beta_end=300.0,
            lr_start=0.002,
            lr_end=0.0001,
            max_iter=5000
        )
    
    return best_pts
# EVOLVE-BLOCK-END
