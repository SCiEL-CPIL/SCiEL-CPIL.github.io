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
    Returns the circumcenter (unit vector) and the circumradius (angular distance in radians).
    The circumcenter is chosen to be on the same side as the outward normal of the facet.
    """
    a, b, c = tri[0], tri[1], tri[2]
    
    # Compute the outward normal of the triangle (for convex hull facets)
    # The outward normal points away from the origin
    n = np.cross(b - a, c - a)
    norm_n = np.linalg.norm(n)
    
    if norm_n < 1e-12:
        # Degenerate: points are collinear or antipodal
        # Use the midpoint of the longest edge as a fallback
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
            return a, np.pi / 2
        p = mid / norm_mid
        r = np.arccos(np.clip(np.dot(p, a), -1, 1))
        return p, r
    
    # The circumcenter lies along the direction orthogonal to (a-b) and (a-c)
    # which is parallel to cross(a-b, a-c)
    u = a - b
    v = a - c
    w = np.cross(u, v)
    norm_w = np.linalg.norm(w)
    
    if norm_w < 1e-12:
        # Degenerate case
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
            return a, np.pi / 2
        p = mid / norm_mid
        r = np.arccos(np.clip(np.dot(p, a), -1, 1))
        return p, r
    
    p = w / norm_w
    
    # Choose the correct sign: the circumcenter should be on the same side as the outward normal
    # For a convex hull facet, the outward normal points away from the interior
    # The circumcenter of the spherical triangle (which is a facet of the convex hull)
    # should be in the direction of the outward normal
    if np.dot(p, n) < 0:
        p = -p
    
    # Compute the circumradius
    # It should be the same for all three vertices, but use the mean for numerical stability
    d1 = np.arccos(np.clip(np.dot(p, a), -1, 1))
    d2 = np.arccos(np.clip(np.dot(p, b), -1, 1))
    d3 = np.arccos(np.clip(np.dot(p, c), -1, 1))
    r = np.mean([d1, d2, d3])
    
    return p, r

def _compute_covering_radius(pts):
    """
    Compute the covering radius by finding the convex hull and checking each facet's circumradius.
    Returns the max radius (in radians) and the indices of the facets achieving it (or close to it).
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
    
    # Find facets with radii close to max (within 1e-3)
    active_indices = np.where(radii > max_radius - 1e-3)[0]
    
    return max_radius, active_indices, circumcenters

def _manifold_gradient(pts, active_indices, circumcenters, hull):
    """
    Compute the Riemannian gradient for minimizing the max circumradius.
    For each active facet, compute the gradient of the circumradius w.r.t. each vertex.
    The gradient of d(v, cc) = arccos(v · cc) w.r.t. v on the sphere is:
      grad = -cc/sin(d) + cot(d) * v  (in ambient space)
    Projected to tangent space: grad_tan = -u, where u is the unit tangent vector from v towards cc.
    To minimize, we move in direction +u.
    """
    grad = np.zeros_like(pts)
    
    for i in active_indices:
        simplex = hull.simplices[i]
        cc = circumcenters[i]
        
        for idx in simplex:
            v = pts[idx]
            dot_vc = np.dot(v, cc)
            dot_vc = np.clip(dot_vc, -1.0 + 1e-15, 1.0 - 1e-15)
            
            # Tangent vector from v towards cc
            t = cc - dot_vc * v
            norm_t = np.linalg.norm(t)
            
            if norm_t > 1e-12:
                u = t / norm_t
                grad[idx] += u
    
    # For each point, the gradient is already in the tangent space
    # (since u is orthogonal to v, and we're summing such vectors)
    # But we need to ensure the sum is also in the tangent space
    # Actually, each u is in T_v S^2, and the sum of vectors in T_v S^2 is also in T_v S^2
    # So no additional projection needed
    
    return grad

def _exp_map(pts, grad, step_size):
    """
    Apply the exponential map on the sphere: move each point along the geodesic
    in the direction of the gradient.
    exp_v(w) = cos(|w|) * v + sin(|w|) * (w/|w|)
    """
    new_pts = np.zeros_like(pts)
    for i in range(N):
        v = pts[i]
        w = step_size * grad[i]
        norm_w = np.linalg.norm(w)
        
        if norm_w < 1e-15:
            new_pts[i] = v
        else:
            new_pts[i] = np.cos(norm_w) * v + np.sin(norm_w) * (w / norm_w)
            new_pts[i] = _normalize(new_pts[i].reshape(1, -1))[0]
    
    return new_pts

def _optimize_single_start(pts, max_iters=3000, timeout=100.0):
    """
    Optimize a single starting configuration using Riemannian gradient descent
    with backtracking line search on the sphere.
    """
    start_time = time.time()
    
    # Initial evaluation
    current_radius, active_indices, circumcenters = _compute_covering_radius(pts)
    best_pts = pts.copy()
    best_radius = current_radius
    
    step_size = 0.1
    no_improve_count = 0
    
    for iteration in range(max_iters):
        if time.time() - start_time > timeout:
            break
        
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        # Recompute radii and active indices
        radii = []
        circumcenters = []
        for simplex in hull.simplices:
            tri = pts[simplex]
            cc, r = _spherical_circumcenter(tri)
            radii.append(r)
            circumcenters.append(cc)
        radii = np.array(radii)
        max_radius = np.max(radii)
        active_indices = np.where(radii > max_radius - 1e-3)[0]
        
        if max_radius < best_radius:
            best_radius = max_radius
            best_pts = pts.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        if len(active_indices) == 0:
            break
        
        # Compute Riemannian gradient
        grad = _manifold_gradient(pts, active_indices, circumcenters, hull)
        
        # Check if gradient is zero
        grad_norms = np.linalg.norm(grad, axis=1)
        if np.max(grad_norms) < 1e-10:
            break
        
        # Backtracking line search
        improved = False
        for shrink in range(30):
            new_pts = _exp_map(pts, grad, step_size)
            new_radius, _, _ = _compute_covering_radius(new_pts)
            
            # Accept if we improve over the current point (not just best)
            if new_radius < max_radius - 1e-8:
                pts = new_pts
                no_improve_count = 0
                improved = True
                break
            else:
                step_size *= 0.5
        
        if not improved:
            if step_size < 1e-8:
                # Stuck: add small random perturbation to escape local minimum
                noise = 0.02 * np.random.randn(N, 3)
                pts = pts + noise
                pts = _normalize(pts)
                step_size = 0.1
                no_improve_count = 0
            else:
                break
    
    return best_pts, best_radius

def construct():
    """
    Main function: use multiple random rotations of the Fibonacci spiral,
    optimize each with Riemannian gradient flow, and return the best configuration.
    """
    best_pts = None
    best_radius = np.pi
    
    start_time = time.time()
    time_budget = 112.0
    
    # Try several starts with different random rotations
    num_starts = 6
    time_per_start = time_budget / num_starts
    
    for k in range(num_starts):
        # Generate Fibonacci spiral
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        pts = _normalize(pts)
        
        # Apply a random rotation
        M = np.random.randn(3, 3)
        Q, R = np.linalg.qr(M)
        d = np.sign(np.linalg.det(Q))
        Q[:, 0] *= d
        
        pts = pts @ Q.T
        
        # Optimize
        current_pts, current_radius = _optimize_single_start(
            pts, max_iters=2000, timeout=time_per_start
        )
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_pts = current_pts
        
        # Check time
        if time.time() - start_time > time_budget:
            break
    
    if best_pts is None:
        # Fallback to unoptimized Fibonacci
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        best_pts = _normalize(best_pts)
    
    return best_pts

# EVOLVE-BLOCK-END
