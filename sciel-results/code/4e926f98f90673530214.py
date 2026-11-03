"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import random


def construct():
    """
    Construct 300 equal circles inside a unit circle.
    Uses a multi-start local optimization approach starting from a
    well-distributed initial configuration (Fibonacci sphere / spiral).
    """
    np.random.seed(42)
    
    # Initial configuration: use a Fibonacci-like spiral or random points
    # that are well-distributed, then optimize to maximize minimum distance.
    
    def objective(x):
        """Negative of the minimum pairwise distance and boundary distance."""
        x = x.reshape(-1, 2)
        # Boundary constraint: distance from origin must be <= 1 - r
        # We'll handle this via penalty
        r_boundary = 1.0 - np.linalg.norm(x, axis=1)
        
        # Pairwise distances
        diffs = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        r_pair = np.min(dists) / 2.0
        
        # The actual radius is min(r_boundary, r_pair)
        r = min(np.min(r_boundary), r_pair)
        
        # Penalty for any point outside
        penalty = 0.0
        for i in range(len(x)):
            if np.linalg.norm(x[i]) > 1.0:
                penalty += (np.linalg.norm(x[i]) - 1.0) ** 2
        
        return -r + penalty
    
    def run_optimization(x0, n_iter=2000):
        """Run local optimization from initial point x0."""
        x0 = x0.flatten()
        
        best_x = x0.copy()
        best_val = objective(x0)
        
        # Use Nelder-Mead for local optimization
        for attempt in range(3):
            res = minimize(
                objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': n_iter, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            val = res.fun
            if val < best_val:
                best_val = val
                best_x = res.x
            # Perturb for next attempt
            x0 = res.x + np.random.normal(0, 0.01, size=x0.shape)
            x0 = np.clip(x0, -1.0, 1.0)
        
        return best_x.reshape(-1, 2), -best_val
    
    def get_initial_config(method='spiral'):
        """Generate an initial well-distributed configuration."""
        if method == 'spiral':
            # Vogel's spiral
            golden_angle = np.pi * (3.0 - np.sqrt(5.0))
            x = np.zeros((N, 2))
            for i in range(N):
                r = np.sqrt((i + 0.5) / N)
                theta = i * golden_angle
                x[i, 0] = r * np.cos(theta)
                x[i, 1] = r * np.sin(theta)
            return x
        elif method == 'random':
            # Random points, reject if outside
            x = np.random.uniform(-1, 1, size=(N, 2))
            while True:
                mask = np.linalg.norm(x, axis=1) <= 1.0
                if np.sum(mask) >= N:
                    return x[mask][:N]
                x = np.random.uniform(-1, 1, size=(N, 2))
        else:
            raise ValueError("Unknown method")
    
    best_radius = 0.0
    best_points = None
    
    # Try multiple initial configurations
    for method in ['spiral']:
        x0 = get_initial_config(method)
        # Ensure all points are strictly inside
        norms = np.linalg.norm(x0, axis=1)
        x0[norms > 0.95] *= 0.95 / norms[norms > 0.95, np.newaxis]
        
        points, radius = run_optimization(x0, n_iter=3000)
        if radius > best_radius:
            best_radius = radius
            best_points = points
    
    # Additional refinement: try a few more perturbations
    for _ in range(5):
        x0 = best_points + np.random.normal(0, 0.02, size=best_points.shape)
        norms = np.linalg.norm(x0, axis=1)
        x0[norms > 0.98] *= 0.98 / norms[norms > 0.98, np.newaxis]
        points, radius = run_optimization(x0, n_iter=2000)
        if radius > best_radius:
            best_radius = radius
            best_points = points
    
    # Final check and clip
    norms = np.linalg.norm(best_points, axis=1)
    mask = norms <= 1.0
    if np.sum(mask) < N:
        # Fallback to spiral
        best_points = get_initial_config('spiral')
        norms = np.linalg.norm(best_points, axis=1)
        best_points[norms > 0.95] *= 0.95 / norms[norms > 0.95, np.newaxis]
    
    return best_points
# EVOLVE-BLOCK-END
