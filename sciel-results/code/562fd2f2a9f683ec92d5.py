"""Rewritten program for maximizing the min/max distance ratio for n=14, d=3."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 14
D = 3

def _pairwise_distances(points):
    """Compute all pairwise squared distances."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff ** 2, axis=-1)
    # Get upper triangle indices (excluding diagonal)
    i, j = np.triu_indices(N, k=1)
    return sq_dists[i, j]

def _score_from_sq_dists(sq_dists):
    """Compute score = (max_dist / min_dist)^2 from squared distances."""
    min_sq = np.min(sq_dists)
    max_sq = np.max(sq_dists)
    if min_sq <= 0:
        return np.inf
    return (max_sq / min_sq) ** 2

def _objective(x):
    """Objective to minimize: (max/min)^2."""
    points = x.reshape(N, D)
    sq_dists = _pairwise_distances(points)
    return _score_from_sq_dists(sq_dists)

def construct():
    """Construct 14 points in 3D to minimize (max/min pairwise distance)^2."""
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    # Fibonacci sphere initialization for good starting point
    def fibonacci_sphere(n):
        """Generate n points on a sphere using Fibonacci spiral."""
        points = np.zeros((n, 3))
        golden_angle = np.pi * (3 - np.sqrt(5))
        for i in range(n):
            y = 1 - 2 * i / (n - 1) if n > 1 else 0
            radius = np.sqrt(1 - y * y)
            theta = golden_angle * i
            points[i] = [np.cos(theta) * radius, y, np.sin(theta) * radius]
        return points
    
    # Initial guess: Fibonacci sphere scaled
    initial_points = fibonacci_sphere(N)
    # Scale to unit sphere
    norms = np.linalg.norm(initial_points, axis=1, keepdims=True)
    initial_points = initial_points / norms
    
    best_x = initial_points.flatten()
    best_score = _objective(best_x)
    
    # Try Nelder-Mead optimization
    result = minimize(
        _objective,
        best_x,
        method='Nelder-Mead',
        options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8}
    )
    
    if result.fun < best_score:
        best_x = result.x
        best_score = result.fun
    
    if time.time() > deadline - 30:
        return best_x.reshape(N, D)
    
    # Try multiple random restarts with Nelder-Mead
    np.random.seed(42)
    for restart in range(20):
        if time.time() > deadline - 20:
            break
        
        # Random initial points on sphere with some randomness
        random_points = np.random.randn(N, D)
        norms = np.linalg.norm(random_points, axis=1, keepdims=True)
        random_points = random_points / norms
        
        result = minimize(
            _objective,
            random_points.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-8}
        )
        
        if result.fun < best_score:
            best_x = result.x
            best_score = result.fun
    
    if time.time() > deadline - 10:
        return best_x.reshape(N, D)
    
    # Final polish with tight tolerance
    result = minimize(
        _objective,
        best_x,
        method='Nelder-Mead',
        options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10}
    )
    
    if result.fun < best_score:
        best_x = result.x
        best_score = result.fun
    
    return best_x.reshape(N, D)
