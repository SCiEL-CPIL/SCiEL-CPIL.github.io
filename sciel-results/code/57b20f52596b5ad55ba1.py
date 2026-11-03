"""Gradient-based optimization with smooth covering radius approximation."""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 50

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms

def _covering_radius_exact(pts):
    """
    Compute the exact covering radius in degrees using Convex Hull facets.
    """
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 180.0
    
    max_radius_deg = 0.0
    for simplex in hull.simplices:
        p0 = pts[simplex[0]]
        p1 = pts[simplex[1]]
        p2 = pts[simplex[2]]
        
        a = p1 - p0
        b = p2 - p0
        aa = np.dot(a, a)
        bb = np.dot(b, b)
        ab = np.dot(a, b)
        
        denom = aa * bb - ab * ab
        if abs(denom) < 1e-15:
            continue
            
        s = (aa * bb - ab * ab) / (2.0 * denom)
        t = (aa * ab - aa * aa) / (2.0 * denom)
        
        circumcenter_2d = p0 + s * a + t * b
        R = np.linalg.norm(circumcenter_2d - p0)
        C_norm = np.linalg.norm(circumcenter_2d)
        
        if C_norm < 1e-15:
            theta = np.pi / 2.0
        else:
            cos_theta = (C_norm * C_norm + 1.0 - R * R) / (2.0 * C_norm)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            theta = np.arccos(cos_theta)
        
        radius_deg = np.degrees(theta)
        if radius_deg > max_radius_deg:
            max_radius_deg = radius_deg
            
    return max_radius_deg

def _smooth_covering_radius(pts, n_samples=500, temperature=0.05):
    """
    Compute a smooth approximation of the covering radius using soft-min.
    This is differentiable and suitable for gradient-based optimization.
    
    Args:
        pts: (N, 3) array of unit vectors
        n_samples: number of sample points on the sphere
        temperature: softness parameter (lower = sharper max)
    
    Returns:
        Smooth covering radius in radians
    """
    pts = _normalize(pts)
    rng = np.random.RandomState(42)
    samples = rng.randn(n_samples, 3)
    samples = _normalize(samples)
    
    # Compute angular distances from each sample to each point
    cos_dist = samples @ pts.T  # (n_samples, N)
    cos_dist = np.clip(cos_dist, -1.0, 1.0)
    dists = np.arccos(cos_dist)  # (n_samples, N)
    
    # For each sample, find the minimum distance to any point
    min_dists = np.min(dists, axis=1)  # (n_samples,)
    
    # Soft max: -log(sum(exp(-d/T))) / T
    # Use log-sum-exp trick for numerical stability
    scaled = -min_dists / temperature
    max_scaled = np.max(scaled)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_max = -log_sum * temperature
    
    return soft_max

def _smooth_covering_radius_grad(pts, n_samples=500, temperature=0.05):
    """
    Compute smooth covering radius and its gradient.
    
    Args:
        pts: (N, 3) array of unit vectors
        n_samples: number of sample points
        temperature: softness parameter
    
    Returns:
        (value, gradient) where gradient is (N, 3)
    """
    pts = _normalize(pts)
    rng = np.random.RandomState(42)
    samples = rng.randn(n_samples, 3)
    samples = _normalize(samples)
    
    # Compute angular distances
    cos_dist = samples @ pts.T  # (n_samples, N)
    cos_dist = np.clip(cos_dist, -1.0, 1.0)
    dists = np.arccos(cos_dist)  # (n_samples, N)
    
    # For each sample, find the index of the nearest point
    min_dists = np.min(dists, axis=1)  # (n_samples,)
    nearest_idx = np.argmin(dists, axis=1)  # (n_samples,)
    
    # Soft max value
    scaled = -min_dists / temperature
    max_scaled = np.max(scaled)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_max = -log_sum * temperature
    
    # Gradient computation
    # d(soft_max)/d(pts[i]) = sum_s weight_s * d(dist_s)/d(pts[i])
    # where weight_s = exp(-dist_s/T) / sum_j exp(-dist_j/T)
    
    # Compute weights
    exp_scaled = np.exp(scaled - max_scaled)
    weights = exp_scaled / np.sum(exp_scaled)  # (n_samples,)
    
    # Gradient of distance w.r.t. points
    # d(arccos(cos_dist))/d(cos_dist) = -1/sqrt(1-cos_dist^2)
    # d(cos_dist)/d(pts[i]) = -samples / sqrt(1-cos_dist^2)  (when cos_dist != ±1)
    
    grad = np.zeros_like(pts)
    for s in range(n_samples):
        i = nearest_idx[s]
        w = weights[s]
        if w < 1e-10:
            continue
        
        cos_val = cos_dist[s, i]
        sin_val = np.sqrt(np.maximum(0.0, 1.0 - cos_val * cos_val))
        if sin_val < 1e-10:
            continue
        
        # d(dist)/d(pts[i]) = -samples[s] / sin_val
        grad[i] += w * (-samples[s] / sin_val)
    
    # Normalize gradient to tangent space
    for i in range(N):
        grad[i] -= np.dot(grad[i], pts[i]) * pts[i]
    
    return soft_max, grad

def _gradient_descent(pts, n_steps=2000, lr_init=0.01, lr_decay=0.999, 
                      temperature_init=0.1, temperature_decay=0.999,
                      seed=0):
    """
    Gradient descent on the smooth covering radius.
    """
    pts = _normalize(pts.copy())
    rng = np.random.RandomState(seed)
    
    lr = lr_init
    temperature = temperature_init
    
    for step in range(n_steps):
        value, grad = _smooth_covering_radius_grad(pts, temperature=temperature)
        
        # Add small random perturbation to escape local minima
        noise = 0.001 * rng.randn(*pts.shape)
        
        # Update
        pts = pts - lr * (grad + noise)
        pts = _normalize(pts)
        
        # Decay learning rate and temperature
        lr *= lr_decay
        temperature *= temperature_decay
        
        # Periodically reheat to escape local minima
        if step % 500 == 499:
            lr = max(lr, 0.005)
            temperature = max(temperature, 0.05)
    
    return pts

def _fibonacci_sphere(n, offset=0.5):
    """Generate points on a sphere using Fibonacci spiral."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return _normalize(np.column_stack([r * np.cos(phi), r * np.sin(phi), z]))

def _icosahedron_based_init(rng):
    """Generate initial configuration based on icosahedron vertices."""
    phi_golden = (1.0 + np.sqrt(5.0)) / 2.0
    ico_verts = np.array([
        [-1,  phi_golden, 0], [ 1,  phi_golden, 0], [-1, -phi_golden, 0], [ 1, -phi_golden, 0],
        [0, -1,  phi_golden], [0,  1,  phi_golden], [0, -1, -phi_golden], [0,  1, -phi_golden],
        [ phi_golden, 0, -1], [ phi_golden, 0,  1], [-phi_golden, 0, -1], [-phi_golden, 0,  1]
    ], dtype=float)
    ico_verts = _normalize(ico_verts)
    
    # Add random points to fill up to N
    remaining = N - len(ico_verts)
    extra = rng.randn(remaining, 3)
    extra = _normalize(extra)
    
    return _normalize(np.vstack([ico_verts, extra]))

def construct():
    """
    Multi-start gradient-based optimization for minimizing the covering radius.
    """
    best_pts = None
    best_radius = np.inf
    
    rng = np.random.RandomState(42)
    
    # Generate diverse initial configurations
    inits = []
    
    # 1. Fibonacci spiral
    inits.append(_fibonacci_sphere(N))
    
    # 2. Icosahedron-based
    inits.append(_icosahedron_based_init(rng))
    
    # 3. Random starts
    for _ in range(3):
        pts = rng.randn(N, 3)
        inits.append(_normalize(pts))
    
    # 4. Clustered starts (points grouped in regions)
    for _ in range(2):
        centers = rng.randn(5, 3)
        centers = _normalize(centers)
        pts = []
        for c in centers:
            local = c + rng.randn(10, 3) * 0.1
            pts.append(_normalize(local))
        inits.append(np.vstack(pts))
    
    for init_idx, init in enumerate(inits):
        try:
            # Phase 1: Gradient descent on smooth approximation
            pts_smooth = _gradient_descent(
                init,
                n_steps=1500,
                lr_init=0.02,
                lr_decay=0.9995,
                temperature_init=0.15,
                temperature_decay=0.999,
                seed=init_idx
            )
            
            # Evaluate current covering radius
            current_radius_deg = _covering_radius_exact(pts_smooth)
            
            # Phase 2: Fine-tune with Nelder-Mead on exact objective
            def objective_exact(params):
                pts = _normalize(params.reshape(N, 3))
                return _covering_radius_exact(pts)
            
            result_exact = minimize(
                objective_exact,
                pts_smooth.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            
            pts_final = _normalize(result_exact.x.reshape(N, 3))
            final_radius_deg = _covering_radius_exact(pts_final)
            
            if final_radius_deg < best_radius:
                best_radius = final_radius_deg
                best_pts = pts_final
                
        except Exception:
            continue
    
    if best_pts is None:
        best_pts = inits[0]
    
    return best_pts
