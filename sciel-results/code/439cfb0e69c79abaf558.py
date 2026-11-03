import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def compute_facet_data(points):
    """
    Computes the convex hull and returns circumradii and facet indices.
    """
    try:
        hull = ConvexHull(points)
        simplices = hull.simplices
    except Exception:
        return None, None, None
    
    if len(simplices) == 0:
        return None, None, None

    a = points[simplices[:, 0]]
    b = points[simplices[:, 1]]
    c = points[simplices[:, 2]]
    
    # Compute normals
    n = np.cross(b - a, c - a)
    norms = np.linalg.norm(n, axis=1)
    
    valid = norms > 1e-10
    if not np.all(valid):
        return None, None, None
        
    u = n / norms[:, np.newaxis]
    
    # Orient normals outwards
    centroids = (a + b + c) / 3.0
    dots = np.sum(u * centroids, axis=1)
    u[dots < 0] *= -1
    
    # Circumradius
    cos_theta = np.sum(u * a, axis=1)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    radii = np.arccos(cos_theta)
    
    return radii, simplices, u

def optimize(points, time_limit):
    """
    Local optimization using worst-facet gradient descent with annealing.
    """
    best_points = points.copy()
    best_radius = np.inf
    current_radius = np.inf
    
    start_time = time.time()
    step = 0
    alpha = 0.05
    beta = 10.0 # Softness for selecting worst facets
    
    while time.time() - start_time < time_limit:
        radii, simplices, u = compute_facet_data(points)
        
        if radii is None:
            # Degenerate, add noise
            points += np.random.randn(*points.shape) * 0.01
            points /= np.linalg.norm(points, axis=1, keepdims=True)
            continue
        
        current_radius = np.max(radii)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
            
        # Identify worst facets
        # Use a soft selection to avoid discontinuities
        # We want to move vertices of facets with large radii
        # Weight each facet by exp(beta * (radii - max_radius)) ? 
        # Or just take top K.
        
        # Gradient accumulation
        grad = np.zeros_like(points)
        
        # To be efficient, we can just look at the top 10 worst facets
        # But a soft approach is smoother.
        # Let's use a hard top-K for speed and stability in this specific problem
        K = 10
        if len(radii) > K:
            top_k_indices = np.argpartition(radii, -K)[-K:]
        else:
            top_k_indices = np.arange(len(radii))
            
        for idx in top_k_indices:
            # Facet idx has vertices simplices[idx]
            # Circumcenter u[idx]
            # Radius radii[idx]
            # We want to reduce this radius.
            # Gradient for vertex v is -tangent(u - v) / sin(R)
            # Direction to move: towards u.
            
            u_f = u[idx]
            R_f = radii[idx]
            
            # If R is very small, gradient is unstable.
            if R_f < 1e-4:
                continue
                
            sin_R = np.sin(R_f)
            
            for v_idx in simplices[idx]:
                v = points[v_idx]
                # Tangent component of u - v
                # proj = u - (u.dot(v))v
                dot_uv = np.dot(u_f, v)
                tangent = u_f - dot_uv * v
                # Normalize tangent to get unit direction towards u
                t_norm = np.linalg.norm(tangent)
                if t_norm > 1e-8:
                    tangent /= t_norm
                    
                    # Gradient magnitude
                    # dR/dv = -tangent / sin(R)
                    # We want to minimize R, so we move in direction -grad = tangent / sin(R)
                    # Wait, grad is -tangent/sin(R). So -grad is tangent/sin(R).
                    # So we move in direction tangent / sin(R).
                    
                    grad[v_idx] += tangent / sin_R
        
        # Normalize gradient to prevent large jumps
        g_norm = np.linalg.norm(grad, axis=1, keepdims=True)
        g_norm[g_norm == 0] = 1.0
        grad = grad / g_norm
        
        # Update points
        # Use a simple step
        new_points = points + alpha * grad
        new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
        
        # Check if the new configuration is valid (non-degenerate)
        # We just accept it and let the next iteration handle degeneracy via noise
        points = new_points
        
        # Annealing: occasionally accept a random perturbation
        if step % 100 == 0 and np.random.rand() < 0.1:
            points += np.random.randn(*points.shape) * 0.01
            points /= np.linalg.norm(points, axis=1, keepdims=True)
            
        step += 1
        
        # Cool down alpha
        alpha *= 0.999
        
    return best_points, best_radius

def construct():
    start_time = time.time()
    time_limit = 140.0 # Safety margin
    
    best_points = None
    best_radius = np.pi
    
    # Initializations
    # 1. Fibonacci Sphere
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    init_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # 2. Random
    np.random.seed(42)
    init_rand = np.random.randn(N, 3)
    init_rand /= np.linalg.norm(init_rand, axis=1, keepdims=True)
    
    # 3. Perturbed Fibonacci
    np.random.seed(123)
    init_fib_noise = init_fib + np.random.randn(N, 3) * 0.1
    init_fib_noise /= np.linalg.norm(init_fib_noise, axis=1, keepdims=True)
    
    initializations = [init_fib, init_rand, init_fib_noise]
    
    # Add more random restarts
    for k in range(5):
        np.random.seed(1000 + k)
        init_rand_k = np.random.randn(N, 3)
        init_rand_k /= np.linalg.norm(init_rand_k, axis=1, keepdims=True)
        initializations.append(init_rand_k)
        
    for init in initializations:
        if time.time() - start_time > time_limit:
            break
            
        # Allocate remaining time to this optimization, minus a small buffer
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time < 5.0:
            break
            
        # Run optimization for a fraction of the remaining time
        # Or just run until time is low
        opt_time = remaining_time - 2.0 # Leave 2s for next iteration setup
        
        if opt_time < 1.0:
            continue
            
        points, radius = optimize(init.copy(), opt_time)
        
        if radius < best_radius:
            best_radius = radius
            best_points = points.copy()
            
    if best_points is None:
        best_points = init_fib
        
    return best_points
