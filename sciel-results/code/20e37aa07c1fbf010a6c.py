"""Packing 50 equal circles in a unit circle using non-convex optimization."""
import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import minimize
import warnings

N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay, cKDTree
from scipy.optimize import minimize
import warnings
import time

def _objective_and_grad(x):
    """
    Objective: minimize negative of the minimum clearance.
    We want to maximize r = min(min_{i<j} |c_i - c_j|/2, min_i (1 - |c_i|))
    So we minimize -r, or equivalently maximize r.
    
    For gradient-based methods, we use a soft version:
    f(x) = -min( min_{i<j} d_{ij}/2, min_i (1 - |c_i|) )
    
    But min is non-differentiable. We use a log-sum-exp style or just 
    compute gradients based on the active constraints.
    
    Actually, let's use a simpler approach: 
    We'll use a penalty-based approach where we maximize the minimum distance
    using a smooth approximation.
    
    Let's use: f(x) = -logsumexp(-d_i / tau) where d_i are all the clearance values
    This is smooth and its maximum approaches the maximum of min(d_i) as tau -> 0.
    
    Alternatively, we can just use the raw min and handle non-differentiability 
    by using subgradients or by using a different optimization strategy.
    
    Let me try a different approach: use scipy.optimize.minimize with 
    method='L-BFGS-B' or 'Powell' on a smoothed objective.
    
    Actually, for packing problems, a common approach is:
    1. Start with a good initial configuration
    2. Use gradient ascent on the minimum distance
    
    Let me implement a simple gradient ascent where at each step:
    - Compute all pairwise distances and boundary clearances
    - Find the minimum clearance value r*
    - Compute gradient: for the pair(s) achieving min distance, push them apart
    - For the point(s) achieving min boundary clearance, push them toward center
    
    But this is essentially what the seed does poorly. Let me think of a better way.
    
    Better approach: Use a two-stage method
    Stage 1: Get a decent initial packing (hexagonal or random + relaxation)
    Stage 2: Use gradient-based optimization to improve
    
    For the gradient, I'll use the following:
    Let r = min(all clearances)
    For each clearance c_k (either d_ij/2 for a pair, or 1-|c_i| for boundary):
    If c_k is close to r (within some epsilon), contribute to the gradient.
    
    For pair (i,j) with distance d_ij:
    Gradient wrt c_i: -(c_i - c_j) / d_ij * (1/2)  [since d_ij/2 decreases when c_i moves toward c_j]
    Wait, let me be careful.
    
    d_ij = |c_i - c_j|
    clearance_ij = d_ij / 2
    d(clearance_ij)/dc_i = (c_i - c_j) / (2 * d_ij)
    
    So to INCREASE clearance_ij, we should move c_i in direction (c_i - c_j)/d_ij
    and c_j in direction (c_j - c_i)/d_ij
    
    For boundary: clearance_i = 1 - |c_i|
    d(clearance_i)/dc_i = -c_i / |c_i|
    To increase clearance_i, move c_i in direction -c_i/|c_i| (toward origin)
    
    So the gradient of the min clearance with respect to x is:
    Sum over active constraints of their individual gradients.
    
    Let me implement this properly.
    """
    x = x.reshape(-1, 2)
    n = len(x)
    
    # Compute all pairwise distances
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    # Pairwise clearances
    pair_clear = dist / 2.0
    
    # Boundary clearances
    radii = np.sqrt(np.sum(x**2, axis=1))
    bound_clear = 1.0 - radii
    
    # All clearances
    all_clear = np.concatenate([pair_clear.flatten(), bound_clear])
    
    # Find minimum clearance
    r_min = np.min(all_clear)
    
    # Objective: we want to maximize r_min, so minimize -r_min
    obj = -r_min
    
    # Compute gradient
    grad = np.zeros_like(x)
    
    # Threshold for "active" constraints
    eps = 1e-4
    active = all_clear < (r_min + eps)
    
    # Pairwise contributions
    pair_active = active[:n*(n-1)]
    pair_active_matrix = pair_active.reshape(n, n-1) if False else None
    
    # Let's do it more carefully
    # pair_clear is (n, n) with diagonal inf
    # Flatten: row-major, so index i*n+j for i!=j
    # But we have upper triangle only? No, we have all pairs.
    # Actually dist is symmetric, so pair_clear is symmetric.
    # Let's just iterate.
    
    for i in range(n):
        for j in range(i+1, n):
            d = dist[i, j]
            if d < 1e-10:
                continue
            c = d / 2.0
            if c < r_min + eps:
                # Gradient contribution
                direction = (x[i] - x[j]) / d
                grad[i] += direction / 2.0
                grad[j] += -direction / 2.0
    
    for i in range(n):
        r = radii[i]
        if r < 1e-10:
            continue
        c = 1.0 - r
        if c < r_min + eps:
            direction = -x[i] / r
            grad[i] += direction
    
    # We're minimizing obj = -r_min, so gradient of obj is -gradient of r_min
    # But we computed gradient of r_min (to increase r_min)
    # So gradient of obj = -grad
    grad_obj = -grad
    
    return obj, grad_obj.flatten()


def _initial_hexagonal():
    """Generate initial hexagonal packing."""
    points = []
    spacing = 0.15
    for k in range(50):
        for i in range(-5, 6):
            for j in range(-5, 6):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x*x + y*y <= 0.95:
                    points.append([x, y])
    
    points = np.array(points)
    if len(points) >= N:
        # Pick N points closest to center
        radii = np.sqrt(np.sum(points**2, axis=1))
        idx = np.argsort(radii)[:N]
        return points[idx]
    else:
        # Fallback to random
        return _initial_random()


def _initial_random():
    """Generate random initial configuration inside unit circle."""
    np.random.seed(42)
    points = []
    for _ in range(N):
        while True:
            x = np.random.uniform(-1, 1)
            y = np.random.uniform(-1, 1)
            if x*x + y*y <= 0.81:  # radius 0.9
                points.append([x, y])
                break
    return np.array(points)


def _relaxation_step(x, step_size=0.01):
    """One step of gradient ascent on minimum clearance."""
    x = x.reshape(-1, 2)
    n = len(x)
    
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    pair_clear = dist / 2.0
    radii = np.sqrt(np.sum(x**2, axis=1))
    bound_clear = 1.0 - radii
    all_clear = np.concatenate([pair_clear.flatten(), bound_clear])
    r_min = np.min(all_clear)
    
    eps = 1e-3
    grad = np.zeros_like(x)
    
    for i in range(n):
        for j in range(i+1, n):
            d = dist[i, j]
            if d < 1e-10:
                continue
            c = d / 2.0
            if c < r_min + eps:
                direction = (x[i] - x[j]) / d
                grad[i] += direction
                grad[j] += -direction
    
    for i in range(n):
        r = radii[i]
        if r < 1e-10:
            continue
        c = 1.0 - r
        if c < r_min + eps:
            direction = -x[i] / r
            grad[i] += direction
    
    # Normalize gradient
    norm = np.sqrt(np.sum(grad**2))
    if norm > 1e-10:
        grad /= norm
    
    x_new = x + step_size * grad
    
    # Project back to unit circle (with margin)
    radii_new = np.sqrt(np.sum(x_new**2, axis=1))
    for i in range(n):
        if radii_new[i] > 0.99:
            x_new[i] *= 0.99 / radii_new[i]
    
    return x_new


def construct():
    """Construct packing of N equal circles in unit circle."""
    # Start with hexagonal initial configuration
    x = _initial_hexagonal()
    
    # Run relaxation
    step_size = 0.005
    for iteration in range(2000):
        x = _relaxation_step(x, step_size)
        # Gradually decrease step size
        if iteration % 500 == 499:
            step_size *= 0.5
    
    # Compute final radius
    x = x.reshape(-1, 2)
    n = len(x)
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)
    radii = np.sqrt(np.sum(x**2, axis=1))
    min_bound_clear = np.min(1.0 - radii)
    
    r = min(min_pair_dist / 2.0, min_bound_clear)
    
    # If radius is very small, try a different approach
    if r < 0.1:
        # Try random restarts
        best_x = x
        best_r = r
        for seed in range(5):
            np.random.seed(seed)
            x_try = _initial_random()
            step_size = 0.005
            for iteration in range(1000):
                x_try = _relaxation_step(x_try, step_size)
                if iteration % 300 == 299:
                    step_size *= 0.5
            diff_try = x_try[:, np.newaxis, :] - x_try[np.newaxis, :, :]
            dist_try = np.sqrt(np.sum(diff_try**2, axis=2))
            np.fill_diagonal(dist_try, np.inf)
            min_pair_dist_try = np.min(dist_try)
            radii_try = np.sqrt(np.sum(x_try**2, axis=1))
            min_bound_clear_try = np.min(1.0 - radii_try)
            r_try = min(min_pair_dist_try / 2.0, min_bound_clear_try)
            if r_try > best_r:
                best_r = r_try
                best_x = x_try
        x = best_x
    
    return x
# EVOLVE-BLOCK-END
