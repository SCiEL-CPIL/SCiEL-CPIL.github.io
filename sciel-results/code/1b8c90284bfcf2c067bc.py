# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time


def _score(points):
    """Compute (d_min / d_max)^2 for a set of points."""
    if len(points) < 2:
        return 0.0
    dists = pdist(points, metric='euclidean')
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0 or np.isnan(d_min) or np.isnan(d_max):
        return 0.0
    return (d_min / d_max) ** 2


def _negative_score(x, n=14, d=3):
    """Objective function for minimization (negative of score)."""
    points = x.reshape(n, d)
    return -_score(points)


def _initial_guess(seed):
    """Generate an initial set of points on a sphere with some randomness."""
    rng = np.random.RandomState(seed)
    n = 14
    d = 3
    
    # Start with points roughly on a sphere
    # Use a grid-like distribution on the sphere for better initial spread
    # Fibonacci sphere for uniform distribution
    points = np.zeros((n, d))
    golden_ratio = (1 + np.sqrt(5)) / 2
    for i in range(n):
        z = 1 - 2 * (i + 0.5) / n
        r = np.sqrt(1 - z**2)
        theta = 2 * np.pi * i / golden_ratio
        points[i] = [r * np.cos(theta), r * np.sin(theta), z]
    
    # Add some random perturbation
    points += rng.randn(n, d) * 0.1
    
    return points


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize (d_min / d_max)^2.
    
    Uses random restarts with local optimization via scipy.
    """
    n = 14
    d = 3
    
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s limit
    
    best_points = None
    best_score = -np.inf
    
    # Try multiple restarts with different seeds
    max_restarts = 50
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        seed = restart * 1000 + 42
        x0 = _initial_guess(seed).flatten()
        
        try:
            result = minimize(
                _negative_score,
                x0,
                args=(n, d),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            
            if result.success or result.fun < best_score:
                score = -result.fun
                if score > best_score:
                    best_score = score
                    best_points = result.x.reshape(n, d)
                    
        except Exception:
            continue
    
    # If no optimization succeeded, fall back to the initial guess
    if best_points is None:
        best_points = _initial_guess(42)
    
    return best_points


# EVOLVE-BLOCK-END
