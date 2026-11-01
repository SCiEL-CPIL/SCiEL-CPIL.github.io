"""Rewritten program for circle packing with gradient optimization."""
import numpy as np
from scipy.optimize import minimize
import time

N = 300
ALPHA = 25.0  # Softness parameter for the exponential objective

def _generate_hex_lattice(n_points):
    """Generate a hexagonal lattice of points within the unit circle."""
    # Estimate spacing to get roughly n_points
    # Area of unit circle = pi. Area per point in hex lattice ~ sqrt(3)/2 * s^2.
    # n * sqrt(3)/2 * s^2 = pi => s = sqrt(2*pi / (n*sqrt(3)))
    s = np.sqrt(2.0 * np.pi / (n_points * np.sqrt(3.0)))
    
    points = []
    # Generate grid large enough
    k = int(1.0 / s) + 2
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3.0) / 2.0
            r = np.hypot(x, y)
            if r < 1.0:
                points.append((x, y, r))
    
    if len(points) < n_points:
        # If we don't have enough, reduce spacing
        # This is a fallback, usually s estimation is good
        while len(points) < n_points and s > 0.01:
            s *= 0.95
            points = []
            k = int(1.0 / s) + 2
            for i in range(-k, k + 1):
                for j in range(-k, k + 1):
                    x = (i + 0.5 * (j % 2)) * s
                    y = j * s * np.sqrt(3.0) / 2.0
                    r = np.hypot(x, y)
                    if r < 1.0:
                        points.append((x, y, r))
            
    # Sort by radius and take the closest n_points
    points.sort(key=lambda p: p[2])
    best_points = np.array([p[:2] for p in points[:n_points]])
    
    # Add small random noise to break symmetry and avoid local minima
    noise = np.random.normal(0, 0.01, best_points.shape)
    best_points += noise
    
    # Ensure they are strictly inside (optimization will push them, but init should be valid)
    radii = np.linalg.norm(best_points, axis=1)
    mask = radii >= 1.0
    if np.any(mask):
         best_points[mask] *= 0.99 / radii[mask, None]
         
    return best_points

def _objective_and_grad(x):
    """
    Objective function: sum of exp(-alpha * distance) for pairs and boundaries.
    Minimizing this maximizes the minimum distance (soft min).
    """
    C = x.reshape(N, 2)
    
    # Pairwise distances
    # D[i, j] = ||C[i] - C[j]||
    diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    dist = np.sqrt(dist_sq + 1e-12) # avoid div by 0
    
    # Mask out diagonal
    mask = ~np.eye(N, dtype=bool)
    
    # Term 1: Pairwise repulsion
    # exp(-alpha * dist)
    exp_dist = np.exp(-ALPHA * dist)
    exp_dist[~mask] = 0.0
    
    # Gradient of sum_{i<j} exp(-alpha * d_ij) w.r.t C_i
    # d/dC_i exp(-alpha * ||C_i - C_j||) = -alpha * exp(-alpha * d_ij) * (C_i - C_j) / d_ij
    # Note: dist is symmetric.
    # grad_pair[i] = sum_{j != i} -alpha * exp(-alpha * dist[i,j]) * (C[i] - C[j]) / dist[i,j]
    
    # Vectorized gradient calculation
    # We need (C_i - C_j) / d_ij. This is the unit vector u_ij.
    # diff is C_i - C_j.
    # u = diff / dist[..., None]
    u = diff / (dist[..., None] + 1e-12)
    u[~mask, :, :] = 0.0 # Zero out diagonal
    
    # Weight is alpha * exp(-alpha * dist)
    weights = ALPHA * exp_dist
    # grad_pair[i] = - sum_j weights[i,j] * u[i,j]
    grad_pair = -np.sum(weights[..., None] * u, axis=1)
    
    # Term 2: Boundary attraction
    # B_i = 1 - ||C_i||
    # exp(-alpha * B_i) = exp(-alpha * (1 - ||C_i||)) = exp(alpha * (||C_i|| - 1))
    radii = np.linalg.norm(C, axis=1)
    radii_safe = np.maximum(radii, 1e-12)
    
    exp_bound = np.exp(-ALPHA * (1.0 - radii))
    
    # Gradient of exp(-alpha * (1 - r)) w.r.t C_i
    # d/dC_i exp(-alpha * (1 - r)) = exp(-alpha * (1 - r)) * (-alpha) * d(1-r)/dC_i
    # d(1-r)/dC_i = - C_i / r
    # So grad = -alpha * exp(...) * (-C_i / r) = alpha * exp(...) * C_i / r
    grad_bound = (ALPHA * exp_bound / radii_safe)[:, None] * C
    
    # Total objective
    obj = np.sum(exp_dist) + np.sum(exp_bound)
    
    # Total gradient
    grad = grad_pair + grad_bound
    
    return obj, grad.flatten()

def construct():
    start_time = time.time()
    timeout = 110.0  # Safety margin for 120s limit
    
    best_x = None
    best_score = -np.inf
    
    # Try multiple restarts to avoid local minima
    n_restarts = 10
    
    for i in range(n_restarts):
        if time.time() - start_time > timeout:
            break
            
        # Initialize
        if i == 0:
            x0 = _generate_hex_lattice(N)
        else:
            # Perturb the best solution found so far or generate fresh
            if best_x is not None:
                x0 = best_x + np.random.normal(0, 0.05, best_x.shape)
                # Project back inside if necessary
                radii = np.linalg.norm(x0, axis=1)
                mask = radii >= 1.0
                if np.any(mask):
                    x0[mask] *= 0.99 / radii[mask, None]
            else:
                x0 = _generate_hex_lattice(N)
        
        # Optimize
        res = minimize(_objective_and_grad, x0, jac=True, method='L-BFGS-B',
                       options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12})
        
        if res.success or res.fun < best_score:
            # Calculate actual score (minimum distance)
            C = res.x.reshape(N, 2)
            diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dist, np.inf)
            min_pair_dist = np.min(dist) / 2.0
            
            radii = np.linalg.norm(C, axis=1)
            min_bound_dist = 1.0 - np.max(radii)
            
            score = min(min_pair_dist, min_bound_dist)
            
            if score > best_score:
                best_score = score
                best_x = res.x.copy()
                
        # Check time
        if time.time() - start_time > timeout:
            break
            
    if best_x is None:
        # Fallback to initial lattice if optimization failed completely
        best_x = _generate_hex_lattice(N)
        
    return best_x
