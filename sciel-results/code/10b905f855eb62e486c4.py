import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 130

def _normalize(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return points / norms

def _compute_facet_data(points):
    """
    Computes the circumradius and circumcenter direction for each facet of the convex hull.
    """
    try:
        hull = ConvexHull(points)
        simplices = hull.simplices
    except Exception:
        return None
    
    if len(simplices) == 0:
        return None
        
    a = points[simplices[:, 0]]
    b = points[simplices[:, 1]]
    c = points[simplices[:, 2]]
    
    n = np.cross(b - a, c - a)
    norms = np.linalg.norm(n, axis=1)
    
    valid = norms > 1e-10
    if not np.all(valid):
        return None
        
    u = n / norms[:, np.newaxis]
    
    centroids = (a + b + c) / 3.0
    dots = np.sum(u * centroids, axis=1)
    u[dots < 0] *= -1
    
    cos_theta = np.sum(u * a, axis=1)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    radii = np.arccos(cos_theta)
    
    return radii, u, simplices

def _covering_radius_rad(points):
    """Compute the covering radius in radians."""
    data = _compute_facet_data(points)
    if data is None:
        return np.pi
    radii, _, _ = data
    return np.max(radii)

def _smooth_surrogate(points, beta=100.0):
    """
    Smooth surrogate for the covering radius using log-sum-exp.
    This is differentiable and can be used for optimization.
    """
    data = _compute_facet_data(points)
    if data is None:
        return np.pi, np.zeros((N, 3))
    
    radii, centers, simplices = data
    
    if len(radii) == 0:
        return np.pi, np.zeros((N, 3))
    
    # Log-sum-exp for smooth max
    max_r = np.max(radii)
    # Scale for numerical stability
    scaled = radii * beta
    max_scaled = np.max(scaled)
    exp_scaled = np.exp(scaled - max_scaled)
    log_sum_exp = max_scaled / beta + np.log(np.sum(exp_scaled)) / beta
    
    # Gradient computation: we want to move points to reduce the largest radii
    # For each facet, if it's "active" (close to max), we pull its vertices towards the circumcenter
    # Use softmax weights to determine which facets are active
    weights = np.exp(scaled - max_scaled) / np.sum(np.exp(scaled - max_scaled))
    
    grad = np.zeros((N, 3))
    for m in range(len(simplices)):
        if weights[m] < 1e-6:
            continue
        idx = simplices[m]
        center = centers[m]
        # Move vertices towards circumcenter
        for i_idx in idx:
            grad[i_idx] += weights[m] * (center - points[i_idx])
    
    return log_sum_exp, grad

def _fibonacci_spiral(n):
    """Generate points on a sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_initialization(n, seed):
    """Generate random points on a sphere."""
    np.random.seed(seed)
    points = np.random.randn(n, 3)
    return _normalize(points)

def _electrostatic_init(n, seed, iterations=1000):
    """
    Initialize points using electrostatic repulsion to get a good starting configuration.
    """
    np.random.seed(seed)
    points = _normalize(np.random.randn(n, 3))
    
    # Simple gradient descent to maximize minimum distance (electrostatic equilibrium)
    for it in range(iterations):
        # Compute pairwise distances
        diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        np.fill_diagonal(dists, 1.0)
        
        # Force: repulsive, proportional to 1/d^2
        # But we want to maximize min distance, so use a soft version
        # Use softmax-like weighting for the closest neighbors
        min_dists = np.min(dists, axis=1)
        
        # Move each point away from its closest neighbor
        for i in range(n):
            closest = np.argmin(dists[i])
            if closest == i:
                continue
            direction = points[i] - points[closest]
            direction = _normalize(direction.reshape(1, -1))[0]
            points[i] += 0.01 * direction
        
        points = _normalize(points)
    
    return points

def _polish(points, time_limit):
    """
    Final high-precision polish using local optimization.
    """
    start_time = time.time()
    
    # Use Nelder-Mead on a smooth surrogate
    def objective(x):
        pts = _normalize(x.reshape(N, 3))
        val, _ = _smooth_surrogate(pts, beta=50.0)
        return val
    
    x0 = points.flatten()
    
    result = minimize(
        objective,
        x0,
        method='Nelder-Mead',
        options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
    )
    
    if time.time() - start_time > time_limit:
        return points
    
    polished = _normalize(result.x.reshape(N, 3))
    return polished

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill
    
    best_points = None
    best_radius = np.pi
    
    # Multiple initialization strategies
    initializations = []
    
    # 1. Fibonacci spiral
    initializations.append(_fibonacci_spiral(N))
    
    # 2. Random with different seeds
    for seed in [42, 123, 456, 789, 1000]:
        initializations.append(_random_initialization(N, seed))
    
    # 3. Electrostatic initialization (slower but better)
    if time.time() - start_time < 30:
        initializations.append(_electrostatic_init(N, 42, iterations=500))
    
    # Optimize each initialization
    for init_idx, points in enumerate(initializations):
        if time.time() - start_time > time_limit:
            break
        
        # Check current radius
        current_radius = _covering_radius_rad(points)
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
        
        # Local optimization using smooth surrogate
        T = 0.1  # Initial step size
        cool_rate = 0.997
        no_improve_count = 0
        
        for step in range(3000):
            if time.time() - start_time > time_limit:
                break
            
            # Compute smooth surrogate and gradient
            val, grad = _smooth_surrogate(points, beta=50.0)
            
            # Update points
            points = points - T * grad
            points = _normalize(points)
            
            # Check actual covering radius
            actual_radius = _covering_radius_rad(points)
            
            if actual_radius < best_radius:
                best_radius = actual_radius
                best_points = points.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > 100:
                    T *= 0.95
                    if T < 1e-5:
                        break
                if no_improve_count > 500:
                    break
            
            T *= cool_rate
        
        # Final polish for this initialization
        if time.time() - start_time < time_limit - 10:
            polished = _polish(points, time_limit - 10)
            polished_radius = _covering_radius_rad(polished)
            if polished_radius < best_radius:
                best_radius = polished_radius
                best_points = polished.copy()
    
    # If we have time, do a final global polish on the best points
    if best_points is not None and time.time() - start_time < time_limit - 5:
        final_polished = _polish(best_points, time_limit - 5)
        final_radius = _covering_radius_rad(final_polished)
        if final_radius < best_radius:
            best_points = final_polished
            best_radius = final_radius
    
    if best_points is None:
        best_points = _fibonacci_spiral(N)
    
    return best_points
