"""Optimized covering of the unit sphere with 130 points."""
import time
import numpy as np

N = 130

def _covering_radius(points):
    """Compute the covering radius in degrees for points on the unit sphere."""
    from scipy.spatial import ConvexHull
    
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        
        max_circumradius = 0.0
        for simplex in facets:
            p1, p2, p3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            
            a = np.arccos(np.clip(np.dot(p2, p3), -1, 1))
            b = np.arccos(np.clip(np.dot(p1, p3), -1, 1))
            c = np.arccos(np.clip(np.dot(p1, p2), -1, 1))
            
            s = (a + b + c) / 2
            denom = 4 * np.sin(s) * np.sin(s-a) * np.sin(s-b) * np.sin(s-c)
            
            if abs(denom) > 1e-15:
                sin_R = np.sin(a) * np.sin(b) * np.sin(c) / denom
                sin_R = np.clip(sin_R, 0, 1)
                R = np.arcsin(sin_R)
            else:
                R = max(a, b, c) / 2
            
            if R > max_circumradius:
                max_circumradius = R
        
        return np.degrees(max_circumradius)
    
    except Exception:
        rng = np.random.RandomState(42)
        n_samples = 10000
        samples = rng.randn(n_samples, 3)
        samples /= np.linalg.norm(samples, axis=1, keepdims=True)
        
        min_dists = np.inf
        for i in range(0, n_samples, 1000):
            chunk = samples[i:i+1000]
            d = 1 - np.einsum('ij,kj->ik', chunk, points)
            d = np.clip(d, -1, 1)
            min_d = np.min(d, axis=1)
            min_dists = np.minimum(min_dists, min_d)
        
        max_min_dist = np.max(min_dists)
        return np.degrees(np.arccos(np.clip(max_min_dist, -1, 1)))


def _soft_max_surrogate(points, beta=50.0):
    """Compute a smooth surrogate for the covering radius using soft-max.
    
    The covering radius is max_y min_i angle(y, p_i).
    We approximate this by sampling and using soft-max over the min distances.
    """
    rng = np.random.RandomState(12345)
    n_samples = 3000
    samples = rng.randn(n_samples, 3)
    samples /= np.linalg.norm(samples, axis=1, keepdims=True)
    
    # Compute angular distances: d[i,j] = angle(samples[i], points[j])
    # Use cos for efficiency, then convert
    cos_d = np.clip(np.einsum('ij,kj->ik', samples, points), -1, 1)
    # min over j for each sample
    min_cos = np.max(cos_d, axis=1)  # max cos = min angle
    min_angle = np.arccos(min_cos)
    
    # Soft-max over the angles to get a smooth approximation of max
    # Use the log-sum-exp trick
    scaled = beta * min_angle
    max_scaled = np.max(scaled)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_max = log_sum / beta
    
    return soft_max


def _soft_max_gradient(points, beta=50.0):
    """Compute gradient of the soft-max surrogate w.r.t. point positions.
    
    Returns gradient array of shape (N, 3) on the tangent space.
    """
    rng = np.random.RandomState(12345)
    n_samples = 3000
    samples = rng.randn(n_samples, 3)
    samples /= np.linalg.norm(samples, axis=1, keepdims=True)
    
    cos_d = np.clip(np.einsum('ij,kj->ik', samples, points), -1, 1)
    min_cos = np.max(cos_d, axis=1)
    min_angle = np.arccos(min_cos)
    
    # Soft-max weights
    scaled = beta * min_angle
    max_scaled = np.max(scaled)
    weights = np.exp(scaled - max_scaled)
    weights /= np.sum(weights)
    
    # For each sample, find which point is nearest
    nearest_idx = np.argmax(cos_d, axis=1)
    
    # Gradient of min_angle w.r.t. points[nearest_idx]
    # d/dp_i arccos(cos_d) = -sin_d / (cos_d * sin_d) ... 
    # Actually d/dx arccos(x) = -1/sqrt(1-x^2)
    # d/dp arccos(samples · p) = -samples / sqrt(1 - (samples·p)^2)
    
    # But we only care about the nearest point for each sample
    # The gradient contribution to point j from sample i (where j is nearest to i):
    # d/dp_j min_angle_i = -samples[i] / sqrt(1 - cos_d[i,j]^2)
    
    grad = np.zeros_like(points)
    for i in range(n_samples):
        j = nearest_idx[i]
        sin_d = np.sqrt(np.clip(1.0 - cos_d[i, j]**2, 1e-12, 1.0))
        # Gradient of arccos(cos_d) w.r.t. p_j:
        # = -samples[i] / sin_d
        # But we need to project to tangent space at p_j
        g = -samples[i] / sin_d
        # Project to tangent space
        g -= np.dot(g, points[j]) * points[j]
        grad[j] += weights[i] * g
    
    # Normalize gradient to have reasonable magnitude
    grad_norm = np.linalg.norm(grad, axis=1, keepdims=True)
    grad_norm = np.maximum(grad_norm, 1e-10)
    
    return grad


def _repulsion_force(points, strength=1.0, cutoff=0.5):
    """Compute repulsive forces between nearby points.
    
    Points repel each other with a force that increases as they get closer.
    This helps prevent clustering and ensures good spread.
    """
    n = len(points)
    forces = np.zeros_like(points)
    
    for i in range(n):
        for j in range(i + 1, n):
            d = np.dot(points[i], points[j])
            if d > np.cos(cutoff):  # Only interact if within cutoff angle
                # Repulsive force: push apart
                # Force direction: away from each other
                # Project onto tangent space
                f_ij = points[i] - d * points[j]  # Tangential component
                f_ji = points[j] - d * points[i]
                
                # Force magnitude increases as points get closer
                # Use a smooth repulsion: 1/(1-d)^2 or similar
                dist_factor = 1.0 / (1.0 - d + 1e-8)
                
                f_ij *= strength * dist_factor
                f_ji *= strength * dist_factor
                
                forces[i] += f_ij
                forces[j] += f_ji
    
    return forces


def _project_to_sphere(points):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return points / norms


def _fibonacci_sphere(n):
    """Generate n points on the unit sphere using the Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, seed=None):
    """Generate n random points on the unit sphere."""
    rng = np.random.RandomState(seed)
    points = rng.randn(n, 3)
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    return points


def _optimize_combined(points, max_time=100, beta=50.0):
    """Combined optimization using soft-max gradient + repulsion forces.
    
    Uses gradient descent on the soft-max surrogate for the covering radius,
    combined with repulsive forces to prevent clustering.
    """
    start_time = time.time()
    points = _project_to_sphere(points.copy())
    
    best_points = points.copy()
    best_radius = _covering_radius(points)
    
    lr = 0.02
    repulsion_strength = 0.05
    beta_current = beta
    patience = 0
    max_patience = 200
    
    for iteration in range(5000):
        if time.time() - start_time > max_time:
            break
        
        # Compute gradient of soft-max surrogate
        grad = _soft_max_gradient(points, beta_current)
        
        # Compute repulsion forces
        rep_force = _repulsion_force(points, repulsion_strength, cutoff=0.4)
        
        # Combined update: gradient descent on surrogate + repulsion
        update = -lr * grad + rep_force
        
        # Project update to tangent space
        for i in range(len(points)):
            update[i] -= np.dot(update[i], points[i]) * points[i]
        
        # Apply update
        points = points + update
        points = _project_to_sphere(points)
        
        # Periodically evaluate and track best
        if iteration % 50 == 0:
            radius = _covering_radius(points)
            if radius < best_radius:
                best_radius = radius
                best_points = points.copy()
                patience = 0
            else:
                patience += 1
            
            # Adaptive learning rate and beta
            if patience > max_patience:
                lr *= 0.8
                beta_current *= 1.2  # Make soft-max sharper
                patience = 0
            
            # Check if we're making progress
            if time.time() - start_time > max_time * 0.8:
                # In the last 20% of time, use smaller steps
                lr *= 0.9
    
    return best_points, best_radius


def _multi_restart_optimize(max_time=100):
    """Try multiple initial configurations and return the best one."""
    start_time = time.time()
    deadline = start_time + max_time
    
    best_points = None
    best_radius = np.inf
    
    # Different initial configurations
    configs = []
    
    # Fibonacci spiral
    configs.append(_fibonacci_sphere(N))
    
    # Random initializations with different seeds
    for seed in [42, 123, 456, 789, 101, 202, 303]:
        configs.append(_random_sphere(N, seed=seed))
    
    # Time per config
    time_per_config = (deadline - time.time()) / len(configs)
    time_per_config = max(5.0, min(15.0, time_per_config))  # Clamp between 5 and 15s
    
    for i, init_points in enumerate(configs):
        if time.time() > deadline - 10:
            break
        
        remaining = deadline - time.time() - 10
        if remaining < 3:
            break
        
        # Optimize this configuration
        points, radius = _optimize_combined(init_points, max_time=remaining)
        
        if radius < best_radius:
            best_radius = radius
            best_points = points.copy()
        
        # If we've found a very good solution, stop early
        if best_radius < 11.32:
            break
    
    if best_points is None:
        best_points = _fibonacci_sphere(N)
        best_radius = _covering_radius(best_points)
    
    return best_points, best_radius


def construct():
    """Construct 130 points on the unit sphere minimizing the covering radius."""
    start_time = time.time()
    deadline = start_time + 120  # Safety margin before 150s kill
    
    # Multi-restart optimization
    points, radius = _multi_restart_optimize(max_time=110)
    
    # Final normalization
    points = _project_to_sphere(points)
    
    # Verify it's valid
    norms = np.linalg.norm(points, axis=1)
    if np.any(norms < 1e-10):
        # Fallback to Fibonacci if something went wrong
        points = _fibonacci_sphere(N)
    
    return points
