"""Spherical Riemannian gradient flow for minimizing covering radius of 50 points."""
import numpy as np
from scipy.spatial import ConvexHull
import time

N = 50

# EVOLVE-BLOCK-START

def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return pts / norms

def _spherical_circumcenter(tri):
    """
    Compute the spherical circumcenter of a triangle on the unit sphere.
    Returns the circumcenter and the circumradius (angular distance).
    """
    a, b, c = tri[0], tri[1], tri[2]
    
    # The circumcenter p satisfies p·a = p·b = p·c = cos(R)
    # So p is orthogonal to (a-b) and (a-c)
    u = a - b
    v = a - c
    w = np.cross(u, v)
    norm_w = np.linalg.norm(w)
    
    if norm_w < 1e-12:
        # Degenerate: collinear on a great circle
        d_ab = np.arccos(np.clip(np.dot(a, b), -1, 1))
        d_ac = np.arccos(np.clip(np.dot(a, c), -1, 1))
        d_bc = np.arccos(np.clip(np.dot(b, c), -1, 1))
        
        if d_ab >= d_ac and d_ab >= d_bc:
            mid = a + b
        elif d_ac >= d_ab and d_ac >= d_bc:
            mid = a + c
        else:
            mid = b + c
            
        norm_mid = np.linalg.norm(mid)
        if norm_mid < 1e-12:
            return a, np.pi/2
        p = mid / norm_mid
        r = np.arccos(np.clip(np.dot(p, a), -1, 1))
        return p, r

    # p is parallel to w; choose sign so that p is on the same side as centroid
    centroid = a + b + c
    norm_centroid = np.linalg.norm(centroid)
    
    if norm_centroid < 1e-12:
        # Triangle spans a hemisphere; use the direction that gives smaller radius
        p1 = w / norm_w
        p2 = -p1
        r1 = np.arccos(np.clip(np.dot(p1, a), -1, 1))
        r2 = np.arccos(np.clip(np.dot(p2, a), -1, 1))
        if r1 <= r2:
            p = p1
            r = r1
        else:
            p = p2
            r = r2
        return p, r
    
    centroid_norm = centroid / norm_centroid
    p = w / norm_w
    # Choose sign: circumcenter should be on the same side as centroid for small facets
    if np.dot(p, centroid_norm) < 0:
        p = -p
        
    # Compute radius from all three vertices (should be equal)
    d1 = np.arccos(np.clip(np.dot(p, a), -1, 1))
    d2 = np.arccos(np.clip(np.dot(p, b), -1, 1))
    d3 = np.arccos(np.clip(np.dot(p, c), -1, 1))
    r = np.max([d1, d2, d3])  # Use max for safety
    
    return p, r

def _compute_covering_radius(pts):
    """
    Compute the covering radius by finding the convex hull and checking each facet's circumradius.
    Returns the max radius and the indices of the facets achieving it (or close to it).
    """
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi, []
    
    radii = []
    circumcenters = []
    for simplex in hull.simplices:
        tri = pts[simplex]
        cc, r = _spherical_circumcenter(tri)
        radii.append(r)
        circumcenters.append(cc)
        
    radii = np.array(radii)
    max_radius = np.max(radii)
    
    # Find facets with radii close to max (within 1e-3 for subgradient)
    active_indices = np.where(radii > max_radius - 1e-3)[0]
    
    return max_radius, active_indices, circumcenters

def _compute_subgradient(pts, active_indices, circumcenters, hull):
    """
    Compute the subgradient of the max circumradius function.
    The subgradient of max(f_i) is the convex hull of gradients of active f_i.
    We use the average of the active gradients.
    """
    grad = np.zeros_like(pts)
    n_active = len(active_indices)
    if n_active == 0:
        return grad
    
    for k, i in enumerate(active_indices):
        simplex = hull.simplices[i]
        cc = circumcenters[i]
        
        for idx in simplex:
            v = pts[idx]
            dot_vc = np.dot(v, cc)
            # Tangent vector from v towards cc
            t = cc - dot_vc * v
            norm_t = np.linalg.norm(t)
            
            if norm_t > 1e-12:
                # Unit tangent vector from v towards cc
                u = t / norm_t
                # The gradient of d(v, cc) w.r.t. v in tangent space is -u
                # To minimize, we move in direction +u
                grad[idx] += u
                
    # Average over active facets (subgradient of max is average of active gradients)
    grad /= n_active
    
    return grad

def _optimize_single_start(pts, max_iters=3000, timeout=100.0):
    """
    Optimize a single starting configuration using Riemannian gradient flow.
    """
    start_time = time.time()
    current_radius, active_indices, circumcenters = _compute_covering_radius(pts)
    best_pts = pts.copy()
    best_radius = current_radius
    
    step_size = 0.1
    min_step = 1e-6
    no_improve_count = 0
    max_no_improve = 50
    
    for iteration in range(max_iters):
        if time.time() - start_time > timeout:
            break
            
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
            
        # Recompute radii and active set
        radii = []
        ccs = []
        for simplex in hull.simplices:
            tri = pts[simplex]
            cc, r = _spherical_circumcenter(tri)
            radii.append(r)
            ccs.append(cc)
        radii = np.array(radii)
        max_radius = np.max(radii)
        active_indices = np.where(radii > max_radius - 1e-3)[0]
        
        if max_radius < best_radius - 1e-8:
            best_radius = max_radius
            best_pts = pts.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        if len(active_indices) == 0:
            break
            
        if no_improve_count > max_no_improve:
            # Perturb to escape local minimum
            noise_scale = 0.005 * (1 + no_improve_count / max_no_improve)
            pts = pts + noise_scale * np.random.randn(N, 3)
            pts = _normalize(pts)
            no_improve_count = 0
            step_size = 0.1
            continue
            
        # Compute subgradient
        grad = _compute_subgradient(pts, active_indices, ccs, hull)
        
        # Check if gradient is negligible
        grad_norms = np.linalg.norm(grad, axis=1)
        if np.max(grad_norms) < 1e-10:
            break
            
        # Line search with backtracking
        improved = False
        curr_step = step_size
        for shrink in range(25):
            new_pts = pts.copy()
            for i in range(N):
                g = grad[i]
                gn = np.linalg.norm(g)
                if gn > 1e-12:
                    # Riemannian gradient step: move in direction g, normalize
                    new_pts[i] = pts[i] + curr_step * g
                    new_pts[i] = _normalize(new_pts[i].reshape(1, -1))[0]
            
            new_radius, _, _ = _compute_covering_radius(new_pts)
            
            if new_radius < best_radius - 1e-10:
                pts = new_pts
                best_radius = new_radius
                best_pts = pts.copy()
                improved = True
                # Slightly increase step if we improved significantly
                if new_radius < best_radius - 0.001:
                    step_size = min(0.3, step_size * 1.2)
                break
            else:
                curr_step *= 0.5
                
        if not improved:
            step_size = max(min_step, step_size * 0.5)
            if step_size < min_step:
                # Try perturbation
                pts = pts + 0.003 * np.random.randn(N, 3)
                pts = _normalize(pts)
                step_size = 0.1
                no_improve_count = 0
                
    return best_pts, best_radius

def _generate_fibonacci_spiral():
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)

def _random_rotation():
    """Generate a random 3x3 rotation matrix."""
    M = np.random.randn(3, 3)
    Q, R = np.linalg.qr(M)
    d = np.sign(np.linalg.det(Q))
    Q[:, 0] *= d
    return Q

def construct():
    """
    Construct 50 points on the unit sphere minimizing the covering radius.
    Uses spherical Riemannian gradient flow with multiple diverse starts.
    """
    best_pts = None
    best_radius = np.pi
    
    start_time = time.time()
    time_budget = 115.0
    
    # Strategy: Use several diverse starting configurations
    # 1. Fibonacci spiral with random rotations
    # 2. Random points (few)
    # 3. Fibonacci with different offsets
    
    configs = []
    
    # Fibonacci spiral with different random rotations
    base_pts = _generate_fibonacci_spiral()
    for _ in range(4):
        Q = _random_rotation()
        configs.append(base_pts @ Q.T)
    
    # Fibonacci with offset
    i = np.arange(N)
    z = 1.0 - 2.0 * (i + 0.5) / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * (i + 0.3)
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    offset_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    offset_pts = _normalize(offset_pts)
    Q = _random_rotation()
    configs.append(offset_pts @ Q.T)
    
    # A few random configurations
    for _ in range(2):
        random_pts = np.random.randn(N, 3)
        random_pts = _normalize(random_pts)
        configs.append(random_pts)
    
    # Optimize each configuration
    time_per_config = time_budget / len(configs)
    
    for k, pts in enumerate(configs):
        if time.time() - start_time > time_budget:
            break
            
        remaining = time_budget - (time.time() - start_time)
        timeout = min(time_per_config, remaining * 0.8)
        
        if timeout < 5.0:
            break
            
        current_pts, current_radius = _optimize_single_start(pts, max_iters=2000, timeout=timeout)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_pts = current_pts
            
    if best_pts is None:
        best_pts = base_pts
        
    return best_pts

# EVOLVE-BLOCK-END
