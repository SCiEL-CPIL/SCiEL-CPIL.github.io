"""Optimized 50-sphere packing in a unit cube."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
CUBE_HALF = 0.5


def _radius_from_centers(centers):
    """Compute the maximum common radius for given centers."""
    # Boundary clearance
    boundary = CUBE_HALF - np.max(np.abs(centers), axis=1)
    min_boundary = np.min(boundary)
    
    # Pairwise distances
    n = len(centers)
    min_pairwise = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_pairwise:
                min_pairwise = d
    
    return min(min_boundary, min_pairwise / 2.0)


def _fibonacci_lattice(n, seed=12345):
    """Generate n points using Fibonacci lattice for good initial distribution."""
    rng = np.random.RandomState(seed)
    # Use golden ratio for 3D Fibonacci lattice
    phi = (1 + np.sqrt(5)) / 2.0
    points = []
    for i in range(n):
        # Map to [0, 1) in 3D using different frequencies
        t = (i + 0.5) / n
        # Add some randomness to break symmetry
        x = (t * phi + rng.uniform(0, 0.1)) % 1.0
        y = (t * phi * phi + rng.uniform(0, 0.1)) % 1.0
        z = (t * phi * phi * phi + rng.uniform(0, 0.1)) % 1.0
        points.append([x * 2 - 1, y * 2 - 1, z * 2 - 1])
    return np.array(points)


def _optimize_packing(initial_centers, time_limit=100):
    """Optimize sphere positions to maximize minimum radius."""
    start_time = time.time()
    
    def objective(x):
        """Negative of the minimum radius (we minimize negative)."""
        centers = x.reshape(-1, 3)
        
        # Check bounds - if any center is outside, return large penalty
        if np.any(np.abs(centers) > CUBE_HALF):
            return 1e6
        
        # Compute boundary clearance
        boundary = CUBE_HALF - np.max(np.abs(centers), axis=1)
        min_boundary = np.min(boundary)
        
        # Compute minimum pairwise distance
        n = len(centers)
        min_pairwise = np.inf
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(centers[i] - centers[j])
                if d < min_pairwise:
                    min_pairwise = d
        
        radius = min(min_boundary, min_pairwise / 2.0)
        return -radius
    
    def gradient(x):
        """Approximate gradient via finite differences."""
        centers = x.reshape(-1, 3)
        n = len(centers)
        eps = 1e-6
        grad = np.zeros_like(x)
        
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            grad[i] = (objective(x_plus) - objective(x_minus)) / (2 * eps)
        
        return grad
    
    best_centers = initial_centers.copy()
    best_radius = _radius_from_centers(best_centers)
    
    # Try multiple optimization attempts with different initializations
    attempts = 0
    max_attempts = 5
    
    while attempts < max_attempts and (time.time() - start_time) < time_limit:
        if attempts == 0:
            current_centers = initial_centers.copy()
        else:
            # Perturb the best solution found so far
            current_centers = best_centers + np.random.randn(N, 3) * 0.01
        
        # Clip to stay within bounds
        current_centers = np.clip(current_centers, -CUBE_HALF + 1e-8, CUBE_HALF - 1e-8)
        
        x0 = current_centers.flatten()
        
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=[(-CUBE_HALF, CUBE_HALF)] * (N * 3),
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if result.success or result.fun < -best_radius:
            new_centers = result.x.reshape(-1, 3)
            new_centers = np.clip(new_centers, -CUBE_HALF + 1e-10, CUBE_HALF - 1e-10)
            new_radius = _radius_from_centers(new_centers)
            if new_radius > best_radius:
                best_radius = new_radius
                best_centers = new_centers
        
        attempts += 1
    
    return best_centers, best_radius


def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit with safety margin
    
    # Initial Fibonacci lattice
    initial_centers = _fibonacci_lattice(N)
    
    # Optimize
    centers, radius = _optimize_packing(initial_centers, time_limit=100)
    
    # Ensure validity
    centers = np.clip(centers, -CUBE_HALF + 1e-10, CUBE_HALF - 1e-10)
    
    return centers
