import time
import numpy as np
import math
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130

# EVOLVE-BLOCK-START
def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _fibonacci_sphere(n, seed_noise=0.0, seed=42):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    if seed_noise > 0:
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, seed_noise, (n, 3))
        pts = pts + noise
        pts = _normalize(pts)
    return pts

def _compute_covering_radius_and_grads(pts):
    """
    Computes the covering radius and the gradient of the covering radius w.r.t each point.
    Returns:
    radius: The covering radius in radians.
    grads: Array of shape (N, 3) containing the gradient of the radius w.r.t each point.
    """
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 180.0 * math.pi / 180.0, np.zeros_like(pts)

    simplices = hull.simplices
    n_faces = len(simplices)
    
    # Precompute circumcenters and radii for all facets
    # We store:
    # centers: (n_faces, 3)
    # min_dots: (n_faces, 3) corresponding to the 3 vertices
    # radii: (n_faces,)
    
    centers = np.zeros((n_faces, 3))
    min_dots = np.zeros((n_faces, 3))
    radii = np.zeros(n_faces)
    
    for i, simplex in enumerate(simplices):
        v0, v1, v2 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        
        # Cross product for normal
        e1 = v1 - v0
        e2 = v2 - v0
        n = np.cross(e1, e2)
        
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            radii[i] = 0.0
            continue
        n = n / n_norm
        
        # Ensure outward normal
        if np.dot(n, v0) < 0:
            n = -n
            
        centers[i] = n
        
        d0 = np.dot(v0, n)
        d1 = np.dot(v1, n)
        d2 = np.dot(v2, n)
        
        min_dots[i, 0] = d0
        min_dots[i, 1] = d1
        min_dots[i, 2] = d2
        
        min_d = np.min(min_dots[i])
        # Radius is arccos(min_d)
        # Clip to avoid domain errors
        min_d = np.clip(min_d, -1.0, 1.0)
        radii[i] = np.arccos(min_d)
        
    max_radius = np.max(radii)
    
    # Identify active facets
    # Use a threshold to include facets close to the max
    # This smooths the gradient
    threshold = max_radius - 0.001 
    active_mask = radii > threshold
    
    if not np.any(active_mask):
        return max_radius, np.zeros_like(pts)
    
    grads = np.zeros_like(pts)
    
    active_indices = np.where(active_mask)[0]
    
    for i in active_indices:
        simplex = simplices[i]
        n = centers[i]
        # The gradient contribution for each vertex is the tangent component of (n - v)
        # We want to move v towards n to increase v.dot(n)
        # grad_v = (n - v) - (n.dot(v)) * v
        # Since n is unit, n.dot(v) is the projection.
        # Actually, the function to minimize is arccos(v.dot(n)).
        # d/dv arccos(v.n) = -n / sqrt(1 - (v.n)^2)
        # This is the gradient of the distance.
        # We want to minimize the distance, so we move in direction -grad_dist = n / sqrt(...)
        # But we only care about the vertex that gives the min dot product?
        # If v0 is the min, moving v0 towards n helps.
        # Moving v1 or v2 towards n might not help if they are not the min.
        # However, moving all vertices towards the circumcenter generally tightens the cap.
        # Let's distribute the force.
        
        # Simple heuristic: Add the tangent vector towards n for all 3 vertices.
        # Weight by how "active" they are (i.e. if they are close to the min).
        
        d0, d1, d2 = min_dots[i]
        # Weights: exp(k * (d - max_d)) ? No, we want to push the ones that are limiting.
        # If d0 is the min, it's the bottleneck.
        # Let's just push all 3 towards n.
        
        v0, v1, v2 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        
        g0 = n - np.dot(n, v0) * v0
        g1 = n - np.dot(n, v1) * v1
        g2 = n - np.dot(n, v2) * v2
        
        # Normalize gradients to have similar magnitude
        g0_norm = np.linalg.norm(g0)
        g1_norm = np.linalg.norm(g1)
        g2_norm = np.linalg.norm(g2)
        
        if g0_norm > 1e-9: g0 /= g0_norm
        if g1_norm > 1e-9: g1 /= g1_norm
        if g2_norm > 1e-9: g2 /= g2_norm
        
        # Accumulate
        grads[simplex[0]] += g0
        grads[simplex[1]] += g1
        grads[simplex[2]] += g2
        
    return max_radius, grads

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Initialization
    # Use a few random restarts to find a good basin
    best_P = None
    best_radius = 180.0 * math.pi / 180.0
    
    rng = np.random.default_rng(42)
    
    # Try a few different initializations
    initializations = [
        _fibonacci_sphere(N, seed_noise=0.0, seed=42),
        _fibonacci_sphere(N, seed_noise=0.05, seed=1),
        _fibonacci_sphere(N, seed_noise=0.1, seed=2),
        _fibonacci_sphere(N, seed_noise=0.2, seed=3)
    ]
    
    # Optimization parameters
    max_iterations = 2000
    step_size = 0.05
    cool_rate = 0.995
    acceptance_temp = 0.001 # Initial "temperature" for Metropolis
    
    for init_P in initializations:
        if time.time() > deadline - 30.0:
            break
            
        P = init_P.copy()
        current_radius, _ = _compute_covering_radius_and_grads(P)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_P = P.copy()
            
        # Annealing loop
        temp = acceptance_temp
        for it in range(max_iterations):
            if time.time() > deadline - 10.0:
                break
                
            radius, grads = _compute_covering_radius_and_grads(P)
            
            if radius < best_radius:
                best_radius = radius
                best_P = P.copy()
                
            # If gradient is very small, we might be in a local min
            # Add noise
            noise = rng.normal(0, 0.01, (N, 3))
            # Project noise to tangent space
            noise = noise - np.sum(noise * P, axis=1, keepdims=True) * P
            noise = _normalize(noise) # Re-normalize directions? No, just use as vector
            # Actually, just add to grads
            grads += noise * temp
            
            # Move
            new_P = P + step_size * grads
            new_P = _normalize(new_P)
            
            new_radius, _ = _compute_covering_radius_and_grads(new_P)
            
            # Metropolis acceptance
            delta_r = new_radius - radius
            if delta_r < 0:
                P = new_P
            else:
                # Accept with probability exp(-delta_r / temp)
                if rng.random() < math.exp(-delta_r / temp):
                    P = new_P
            
            # Cool down
            temp *= cool_rate
            step_size *= 0.999 # Slowly reduce step size
            
    return best_P
# EVOLVE-BLOCK-END
