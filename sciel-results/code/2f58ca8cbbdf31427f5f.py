"""Spherical Riemannian Gradient Descent with Momentum for minimizing covering radius."""
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
    Returns the circumcenter (unit vector) and the circumradius (angular distance).
    """
    a, b, c = tri[0], tri[1], tri[2]
    
    # The circumcenter p satisfies p dot a = p dot b = p dot c.
    # This implies p is orthogonal to a-b and a-c.
    # So p is parallel to cross(a-b, a-c).
    u = a - b
    v = a - c
    w = np.cross(u, v)
    norm_w = np.linalg.norm(w)
    
    if norm_w < 1e-12:
        # Degenerate case: points are collinear on a great circle or identical.
        # Find the point on the great circle equidistant from the three.
        # This is the midpoint of the longest arc?
        # Actually, if they are collinear, the "circumcircle" is the great circle.
        # The "circumradius" is the max distance to the nearest of the three?
        # No, the covering radius of a facet is the max distance from the facet to the nearest point.
        # For a degenerate triangle, the "facet" is an edge or point.
        # If it's an edge, the circumcenter is the midpoint of the edge?
        # Let's just return the midpoint of the longest pair as a heuristic.
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
            # Antipodal points
            return a, np.pi/2
        p = mid / norm_mid
        r = np.arccos(np.clip(np.dot(p, a), -1, 1))
        return p, r

    p = w / norm_w
    
    # Determine orientation: p should be on the same side as the centroid for small triangles.
    centroid = a + b + c
    norm_centroid = np.linalg.norm(centroid)
    if norm_centroid > 1e-12:
        centroid_norm = centroid / norm_centroid
        if np.dot(p, centroid_norm) < 0:
            p = -p
            
    # Calculate radius
    d1 = np.arccos(np.clip(np.dot(p, a), -1, 1))
    d2 = np.arccos(np.clip(np.dot(p, b), -1, 1))
    d3 = np.arccos(np.clip(np.dot(p, c), -1, 1))
    r = np.mean([d1, d2, d3])
    
    return p, r

def _compute_covering_radius(pts):
    """
    Compute the covering radius by finding the convex hull and checking each facet's circumradius.
    Returns the max radius, the active indices (facets close to max), and the hull.
    """
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi, np.array([]), None
    
    radii = []
    for simplex in hull.simplices:
        tri = pts[simplex]
        cc, r = _spherical_circumcenter(tri)
        radii.append(r)
        
    radii = np.array(radii)
    max_radius = np.max(radii)
    
    # Active facets are those within a small tolerance of the max
    # Use a tolerance that is small but accounts for numerical noise
    tol = 1e-4 
    active_indices = np.where(radii > max_radius - tol)[0]
    
    return max_radius, active_indices, hull

def _gradient_step(pts, active_indices, hull, step_size):
    """
    Compute the Riemannian gradient step for the active facets.
    Returns the gradient vector (tangent vectors) for each point.
    """
    grad = np.zeros_like(pts)
    
    # Count how many active facets each point is in, for averaging
    counts = np.zeros(N)
    
    for i in active_indices:
        simplex = hull.simplices[i]
        tri = pts[simplex]
        cc, r = _spherical_circumcenter(tri)
        
        # Skip if radius is 0 or very small
        if r < 1e-6:
            continue
            
        for idx in simplex:
            v = pts[idx]
            dot_vc = np.dot(v, cc)
            # Tangent vector from v to cc
            t = cc - dot_vc * v
            norm_t = np.linalg.norm(t)
            
            if norm_t > 1e-12:
                u = t / norm_t
                # The gradient of the distance is -u (pointing away from cc? No, towards cc is descent).
                # We want to minimize distance, so we move towards cc.
                # So the direction is u.
                grad[idx] += u
                counts[idx] += 1
                
    # Average the gradients
    for i in range(N):
        if counts[i] > 0:
            grad[i] /= counts[i]
            
    return grad

def _optimize_single_start(pts, max_iters=3000, timeout=100.0):
    """
    Optimize a single starting configuration using Riemannian Gradient Descent with Momentum.
    """
    start_time = time.time()
    current_radius, active_indices, hull = _compute_covering_radius(pts)
    best_pts = pts.copy()
    best_radius = current_radius
    
    # Momentum
    velocity = np.zeros_like(pts)
    beta = 0.9
    
    # Initial step size
    alpha = 0.1
    
    for iteration in range(max_iters):
        if time.time() - start_time > timeout:
            break
            
        if hull is None:
            break
            
        # Compute gradient
        grad = _gradient_step(pts, active_indices, hull, alpha)
        
        # Update momentum
        # Project velocity onto tangent space?
        # v is a vector in R^3. We want it to be tangent to pts.
        # Actually, we can just add the gradient to the point and normalize.
        # But for momentum, we should keep the velocity in the tangent space.
        # Simple approach: just use the gradient in Euclidean space, then project.
        
        # Update velocity
        velocity = beta * velocity + (1 - beta) * grad
        
        # Propose new points
        new_pts = pts + alpha * velocity
        
        # Normalize
        new_pts = _normalize(new_pts)
        
        # Line Search: Backtracking
        improved = False
        # Check if the new radius is better
        new_radius, new_active, new_hull = _compute_covering_radius(new_pts)
        
        # We want a decrease.
        # Simple check: if new_radius < current_radius - 1e-7
        # If not, shrink alpha.
        
        # To avoid infinite loops, limit shrink steps
        shrink_count = 0
        max_shrinks = 10
        
        while new_radius >= current_radius - 1e-7 and shrink_count < max_shrinks:
            alpha *= 0.5
            new_pts = pts + alpha * velocity
            new_pts = _normalize(new_pts)
            new_radius, new_active, new_hull = _compute_covering_radius(new_pts)
            shrink_count += 1
            
        if new_radius < current_radius - 1e-7:
            pts = new_pts
            current_radius = new_radius
            active_indices = new_active
            hull = new_hull
            
            if current_radius < best_radius:
                best_radius = current_radius
                best_pts = pts.copy()
            improved = True
        else:
            # If we couldn't improve even with small steps, try a random perturbation
            # to escape local minimum
            if alpha < 1e-6:
                pts = pts + 0.01 * np.random.randn(N, 3)
                pts = _normalize(pts)
                velocity = np.zeros_like(pts)
                alpha = 0.1
                # Recalculate
                current_radius, active_indices, hull = _compute_covering_radius(pts)
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_pts = pts.copy()
            else:
                # Stuck, break
                break
                
    return best_pts, best_radius

def construct():
    # Use multiple random rotations of the Fibonacci spiral to escape local minima
    best_pts = None
    best_radius = np.pi
    
    start_time = time.time()
    time_budget = 110.0
    
    # Try several starts
    num_starts = 8
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
        current_pts, current_radius = _optimize_single_start(pts, max_iters=2000, timeout=time_per_start)
        
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
