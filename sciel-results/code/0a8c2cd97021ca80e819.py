"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    """
    Place 130 points on the unit sphere to minimize the covering radius.
    Uses a multi-start local optimization approach with a good initial
    distribution (Fibonacci spiral) and then optimizes the minimum
    pairwise distance (which correlates with covering radius for near-uniform
    distributions).
    """
    # Start with a well-distributed initial configuration using Fibonacci spiral
    def fibonacci_sphere(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    # Objective: minimize the negative of the minimum pairwise angular distance
    # This encourages points to spread out, which helps reduce covering radius
    def objective(params):
        # Reconstruct points from parameters (using spherical coordinates)
        # We optimize in Cartesian but project back to sphere
        pts = params.reshape(-1, 3)
        # Normalize to unit sphere
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        pts = pts / norms
        
        # Compute all pairwise distances
        # Use dot products for efficiency
        dots = np.dot(pts, pts.T)
        # Clamp to [-1, 1] for numerical stability
        dots = np.clip(dots, -1.0, 1.0)
        # Angular distances
        angles = np.arccos(dots)
        # Set diagonal to 0
        np.fill_diagonal(angles, 0.0)
        # Minimum pairwise distance
        min_dist = np.min(angles)
        return -min_dist

    # Gradient of the objective
    def objective_gradient(params):
        pts = params.reshape(-1, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        pts = pts / norms
        
        dots = np.dot(pts, pts.T)
        dots = np.clip(dots, -1.0, 1.0)
        angles = np.arccos(dots)
        np.fill_diagonal(angles, 0.0)
        
        # Find the pair with minimum distance
        min_val = np.min(angles)
        # Get indices of minimum
        idx = np.argmin(angles)
        i, j = np.unravel_index(idx, angles.shape)
        
        # Gradient w.r.t. points i and j
        # d(arccos(d_i . d_j))/d d_i = -d_j / sqrt(1 - (d_i . d_j)^2)
        # But we want to maximize min distance, so minimize -min distance
        # Gradient of -min_dist = -gradient of min_dist
        
        dot_ij = dots[i, j]
        denom = np.sqrt(max(1e-10, 1.0 - dot_ij * dot_ij))
        
        grad = np.zeros_like(pts)
        # d(angles[i,j])/d(pts[i]) = -pts[j] / denom
        # d(angles[i,j])/d(pts[j]) = -pts[i] / denom
        grad[i] += pts[j] / denom
        grad[j] += pts[i] / denom
        
        # Need to account for normalization constraint
        # The actual gradient in parameter space (before normalization) is different
        # For simplicity, we'll use a finite difference approach or just return this
        # and let the optimizer handle it approximately
        
        # Reshape gradient
        return grad.flatten()

    # Multi-start optimization
    best_params = None
    best_score = np.inf

    # Try several different initial configurations
    n_starts = 5
    for start_idx in range(n_starts):
        if start_idx == 0:
            # Fibonacci spiral
            x0 = fibonacci_sphere(N)
        elif start_idx == 1:
            # Random uniform on sphere
            rng = np.random.RandomState(42 + start_idx)
            x0 = rng.randn(N, 3)
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        else:
            # Perturbed Fibonacci
            x0 = fibonacci_sphere(N)
            rng = np.random.RandomState(100 + start_idx)
            noise = rng.randn(N, 3) * 0.1
            x0 = x0 + noise
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)

        # Optimize using BFGS
        result = minimize(
            objective,
            x0.flatten(),
            method='BFGS',
            options={'maxiter': 500, 'gtol': 1e-6}
        )

        if result.success or result.fun < best_score:
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x

    # If optimization failed or we don't have best_params, use the best initial
    if best_params is None:
        best_params = fibonacci_sphere(N).flatten()

    # Final optimization from the best found point
    result = minimize(
        objective,
        best_params,
        method='BFGS',
        options={'maxiter': 1000, 'gtol': 1e-7}
    )

    if result.fun < best_score:
        best_params = result.x

    # Extract final points
    pts = best_params.reshape(-1, 3)
    # Normalize to ensure they're on the unit sphere
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    pts = pts / norms

    return pts
# EVOLVE-BLOCK-END
