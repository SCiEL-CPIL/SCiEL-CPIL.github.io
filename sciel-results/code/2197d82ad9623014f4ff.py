import numpy as np
from scipy.spatial import Delaunay
import time

N = 130

def _normalize(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    return points / norms

def _compute_circumradius(p1, p2, p3):
    # p1, p2, p3 are unit vectors
    # Compute the normal vector
    n = np.cross(p2 - p1, p3 - p1)
    norm_n = np.linalg.norm(n)
    if norm_n < 1e-12:
        return np.pi / 2.0 # Degenerate
    
    # The circumcenter on the sphere is the normalized normal
    # The angular distance from the circumcenter to any vertex is the circumradius
    c = n / norm_n
    
    # Ensure c is on the same side as the vertices (dot product should be positive)
    # Actually, the circumcenter is equidistant. The angular distance is arccos(c . p1)
    # But we need to be careful with the sign of n.
    # The circumcenter is the point on the sphere equidistant from p1, p2, p3.
    # There are two such points: c and -c. We pick the one closer to the triangle.
    # The triangle lies in a plane. The origin is not in the plane (usually).
    # The circumcenter is the intersection of the sphere and the line through the origin normal to the plane.
    
    d1 = np.dot(c, p1)
    d2 = np.dot(c, p2)
    d3 = np.dot(c, p3)
    
    # Due to numerical errors, d1, d2, d3 should be equal.
    # The circumradius R is arccos(d)
    # If d is negative, the triangle is "large" (contains the pole opposite to c).
    # We want the angular radius of the cap covering the triangle.
    # The cap center is c if the triangle is small, or -c if large.
    # Actually, the covering radius is the max over all points.
    # The circumcircle of a Delaunay triangle is empty.
    # The radius is the angular distance from the circumcenter to the vertices.
    
    # To determine which side, check if the origin is inside the tetrahedron formed by the triangle and the circumcenter?
    # Simpler: The circumradius R is in [0, pi/2] if the triangle is small.
    # If the triangle is large (contains the antipode of c), the radius is pi - arccos(d)?
    # No, the angular distance is always arccos(|d|)? No.
    
    # Let's just use the angle.
    # If the triangle is small, c is close to the vertices. d ~ 1. R ~ 0.
    # If the triangle is large, c is far. d ~ -1. R ~ pi.
    # But a Delaunay triangle on a sphere usually has circumradius < pi/2 if points are well distributed.
    # We take R = arccos(np.clip(d1, -1, 1)).
    
    # However, we must ensure we pick the correct circumcenter.
    # The circumcenter must be such that the triangle is contained in the cap of radius R centered at C.
    # This is true for the center closer to the triangle.
    
    # A robust way:
    # The circumradius is the angle subtended by the chord?
    # Let's just compute the angle.
    R = np.arccos(np.clip(d1, -1.0, 1.0))
    
    # If the triangle is "large" (area > pi/2?), the circumcenter might be the other one.
    # For now, assume well-distributed points, so R < pi/2.
    return R

def _get_worst_facets(points, tri, k=1):
    # Compute circumradius for all facets
    radii = []
    centers = []
    for simplex in tri.simplices:
        p1 = points[simplex[0]]
        p2 = points[simplex[1]]
        p3 = points[simplex[2]]
        R = _compute_circumradius(p1, p2, p3)
        radii.append(R)
        
        # Compute center
        n = np.cross(p2 - p1, p3 - p1)
        norm_n = np.linalg.norm(n)
        if norm_n > 1e-12:
            c = n / norm_n
            # Check orientation
            if np.dot(c, p1) < 0:
                c = -c
            centers.append(c)
        else:
            centers.append(np.zeros(3))
            
    radii = np.array(radii)
    centers = np.array(centers)
    
    # Get indices of top k largest radii
    idx = np.argsort(radii)[::-1][:k]
    return idx, radii[idx], centers[idx]

def _gradient_step(points, tri, k=5, step_size=0.01):
    # Get worst facets
    idx, radii, centers = _get_worst_facets(points, tri, k)
    
    # For each worst facet, compute gradient direction for its vertices
    # We want to move vertices towards the circumcenter to reduce radius
    # Gradient of R wrt p_i is roughly - (c - p_i) projected to tangent
    # So we move p_i in direction (c - p_i)
    
    grad = np.zeros_like(points)
    
    for i in range(len(idx)):
        simplex = tri.simplices[idx[i]]
        c = centers[i]
        
        for v_idx in simplex:
            p = points[v_idx]
            # Vector from p to c
            diff = c - p
            # Project diff onto tangent plane at p: diff - (p . diff) p
            proj = diff - np.dot(p, diff) * p
            # Normalize projection to get unit direction
            norm_proj = np.linalg.norm(proj)
            if norm_proj > 1e-12:
                grad[v_idx] += proj / norm_proj
                
    # Normalize gradient
    norm_grad = np.linalg.norm(grad, axis=1, keepdims=True)
    norm_grad[norm_grad == 0] = 1.0
    grad = grad / norm_grad
    
    # Update points
    new_points = points + step_size * grad
    new_points = _normalize(new_points)
    
    return new_points

def _thomson_init(n):
    # Simple Thomson initialization
    points = np.random.randn(n, 3)
    points = _normalize(points)
    
    for _ in range(100):
        # Compute forces
        forces = np.zeros_like(points)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                diff = points[i] - points[j]
                dist_sq = np.dot(diff, diff)
                if dist_sq < 1e-12:
                    continue
                # Repulsion ~ 1/r^2
                force = diff / (dist_sq ** 1.5)
                forces[i] += force
        
        # Update
        points += 0.01 * forces
        points = _normalize(points)
        
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 140 # 10s margin
    
    # Initialize
    points = _thomson_init(N)
    
    # Initial Delaunay
    tri = Delaunay(points)
    
    best_points = points.copy()
    best_radius = np.inf
    
    # Optimize
    step_size = 0.05
    k_worst = 10
    
    for iter in range(5000):
        if time.time() > deadline:
            break
            
        # Get worst facets
        idx, radii, centers = _get_worst_facets(points, tri, k_worst)
        
        if len(radii) == 0:
            break
            
        current_max_radius = radii[0]
        
        if current_max_radius < best_radius:
            best_radius = current_max_radius
            best_points = points.copy()
            
        # Gradient step
        grad = np.zeros_like(points)
        for i in range(len(idx)):
            simplex = tri.simplices[idx[i]]
            c = centers[i]
            
            for v_idx in simplex:
                p = points[v_idx]
                diff = c - p
                proj = diff - np.dot(p, diff) * p
                norm_proj = np.linalg.norm(proj)
                if norm_proj > 1e-12:
                    grad[v_idx] += proj / norm_proj
                    
        norm_grad = np.linalg.norm(grad, axis=1, keepdims=True)
        norm_grad[norm_grad == 0] = 1.0
        grad = grad / norm_grad
        
        new_points = points + step_size * grad
        new_points = _normalize(new_points)
        
        # Check if new points are valid (non-zero, etc.)
        if not np.all(np.isfinite(new_points)):
            break
            
        points = new_points
        
        # Recompute Delaunay every 10 steps to save time
        if iter % 10 == 0:
            try:
                tri = Delaunay(points)
            except Exception:
                # If Delaunay fails, break
                break
                
    return best_points
