import numpy as np
import time
from scipy.spatial import ConvexHull

N = 50

def _normalize(v):
    """Normalize vector to unit length."""
    n = np.linalg.norm(v)
    if n < 1e-15:
        return np.zeros(3)
    return v / n

def _fib_sample(n):
    """Generate n points on unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _circumcenter(p1, p2, p3):
    """
    Compute the circumcenter of a spherical triangle defined by unit vectors p1, p2, p3.
    Returns the unit vector pointing to the circumcenter.
    """
    v1 = p1 - p2
    v2 = p2 - p3
    c = np.cross(v1, v2)
    n = np.linalg.norm(c)
    if n < 1e-10:
        # Degenerate triangle, return midpoint or average
        return _normalize(p1 + p2 + p3)
    c = c / n
    
    # Determine sign. The circumcenter should be on the same side of the triangle as the "center".
    avg = _normalize(p1 + p2 + p3)
    if np.dot(c, avg) < 0:
        c = -c
    return c

def _covering_radius_deg_hull(config):
    """
    Compute the exact covering radius in degrees using convex hull facets.
    """
    try:
        hull = ConvexHull(config)
    except Exception:
        # Fallback to sampling if hull fails
        grid = _fib_sample(20000)
        dots = grid @ config.T
        min_dots = np.min(dots, axis=1)
        max_angle = np.arccos(np.clip(min_dots, -1.0, 1.0))
        return np.degrees(np.max(max_angle))
    
    max_angle = 0.0
    
    for simplex in hull.simplices:
        pts = config[simplex]
        c = _circumcenter(pts[0], pts[1], pts[2])
        
        dots = config @ c
        nearest_idx = np.argmax(dots)
        
        angle = np.arccos(np.clip(dots[nearest_idx], -1.0, 1.0))
        
        if angle > max_angle:
            max_angle = angle
            
    return np.degrees(max_angle)

def _gradient_step(config, step_size, top_k=5):
    """
    Perform one gradient descent step on the covering radius.
    Uses the top K worst facets to compute the gradient.
    """
    try:
        hull = ConvexHull(config)
    except Exception:
        return config, False

    simplices = hull.simplices
    n_facets = len(simplices)
    
    # Compute circumcenters and angles for all facets
    circumcenters = np.zeros((n_facets, 3))
    angles = np.zeros(n_facets)
    nearest_indices = np.zeros(n_facets, dtype=int)
    
    for i, simplex in enumerate(simplices):
        pts = config[simplex]
        c = _circumcenter(pts[0], pts[1], pts[2])
        circumcenters[i] = c
        
        dots = config @ c
        nearest_idx = np.argmax(dots)
        nearest_indices[i] = nearest_idx
        
        # Angle is arccos(dot)
        cos_angle = dots[nearest_idx]
        # Clip to avoid numerical issues in arccos
        cos_angle = np.clip(cos_angle, -1.0 + 1e-12, 1.0 - 1e-12)
        angles[i] = np.arccos(cos_angle)
        
    # Select top K worst facets
    if n_facets > top_k:
        # Get indices of top K largest angles
        top_indices = np.argpartition(angles, -top_k)[-top_k:]
    else:
        top_indices = np.arange(n_facets)
        
    # Initialize gradient
    grad = np.zeros_like(config)
    
    for i in top_indices:
        c = circumcenters[i]
        p_idx = nearest_indices[i]
        p = config[p_idx]
        
        # Gradient of arccos(c . p) w.r.t p
        # d/dp arccos(c.p) = -c / sqrt(1 - (c.p)^2)
        dot_cp = np.dot(c, p)
        # Ensure numerical stability
        denom = np.sqrt(max(1e-12, 1.0 - dot_cp * dot_cp))
        
        # Gradient in R3
        g = -c / denom
        
        # Project to tangent space at p
        # g_tan = g - (p . g) p
        g_tan = g - np.dot(p, g) * p
        
        # Add to gradient
        grad[p_idx] += g_tan
        
    # Update points
    new_config = config - step_size * grad
    new_config = np.array([_normalize(p) for p in new_config])
    
    return new_config, True

def _local_optimize_gradient(config, deadline):
    """
    Optimize using Gradient Descent on the covering radius.
    """
    config = config.copy()
    best_config = config.copy()
    best_score = _covering_radius_deg_hull(config)
    
    step_size = 0.01
    no_improve_count = 0
    
    while time.time() < deadline - 2.0:
        new_config, success = _gradient_step(config, step_size, top_k=10)
        
        if not success:
            # If hull fails, perturb
            config += 1e-4 * np.random.randn(N, 3)
            config = np.array([_normalize(p) for p in config])
            continue
            
        new_score = _covering_radius_deg_hull(new_config)
        
        if new_score < best_score - 1e-10:
            best_score = new_score
            best_config = new_config.copy()
            config = new_config
            no_improve_count = 0
            # Decay step size slightly
            step_size *= 0.99
            if step_size < 1e-5:
                step_size = 1e-5
        else:
            no_improve_count += 1
            # If stuck, increase step size or random restart
            if no_improve_count > 10:
                # Random perturbation
                idx = np.random.randint(N)
                rand_dir = np.random.randn(3)
                rand_dir = _normalize(rand_dir)
                config[idx] = _normalize(config[idx] + 0.05 * rand_dir)
                no_improve_count = 0
                step_size = 0.01
            else:
                # Try a smaller step
                step_size *= 0.9
                if step_size < 1e-6:
                    step_size = 1e-6

    return best_config

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    # Start with Fibonacci spiral
    config = _fib_sample(N)
    
    # Run Gradient Descent optimization
    config = _local_optimize_gradient(config, deadline)
    
    # Final normalization
    config = np.array([_normalize(p) for p in config])
    
    # Ensure no coincident points
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(config[i] - config[j]) < 1e-6:
                # Perturb j
                config[j] = config[j] + 1e-4 * np.random.randn(3)
                config[j] = _normalize(config[j])
                
    return config
