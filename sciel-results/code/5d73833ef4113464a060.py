import numpy as np
from scipy.spatial import ConvexHull
import time

N = 50

def _normalize(v):
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return v
    return v / norm

def _compute_covering_radius(points):
    # Compute convex hull
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi # invalid or degenerate
        
    max_radius = 0.0
    # hull.simplices contains the indices of the vertices of each face
    for simplex in hull.simplices:
        a = points[simplex[0]]
        b = points[simplex[1]]
        c = points[simplex[2]]
        
        # Circumcenter direction
        v = np.cross(a - b, b - c)
        m = _normalize(v)
        
        # Check if m is on the correct side (dot product with a should be positive for acute, but for obtuse it might be negative? 
        # Actually, if the triangle contains the origin in its projection, the circumcenter is on the opposite side.
        # But for points on a sphere, the triangle is on the surface.
        # The circumcenter is the pole of the great circle.
        # We want the circumcenter on the same side as the triangle normal?
        # The normal of the triangle is n = np.cross(b-a, c-a).
        # If np.dot(m, n) < 0, then m is on the other side.
        # In that case, the "circumcenter" for covering purposes is actually the point antipodal to m? 
        # No, the covering radius is determined by the farthest point in the Voronoi cell.
        # If the triangle is obtuse, the Voronoi vertex (circumcenter) is not the farthest point from the vertices in the cell?
        # Actually, the Voronoi cell is bounded by the perpendicular bisectors.
        # The circumcenter is the intersection.
        # If the triangle is obtuse, the circumcenter is outside the triangle.
        # The distance from the circumcenter to the vertices is still the circumradius.
        # However, if the circumcenter is "behind" the triangle (dot(m, n) < 0), the actual farthest point in the Voronoi cell might be one of the vertices or an edge midpoint?
        # No, the Voronoi cell is the intersection of half-spaces.
        # The circumcenter is the unique point equidistant from a, b, c.
        # If the triangle is obtuse, the circumcenter is outside.
        # The distance is $\arccos(|m \cdot a|)$? No, $\arccos(m \cdot a)$ if $m \cdot a > 0$.
        # If $m \cdot a < 0$, the distance is $\pi - \arccos(|m \cdot a|)$?
        # Actually, the angular distance is $\arccos(m \cdot a)$.
        # If $m \cdot a$ is negative, the angle is $> 90$ degrees.
        
        # Let's just compute the distance to the nearest vertex.
        d_a = np.arccos(np.clip(m @ a, -1, 1))
        d_b = np.arccos(np.clip(m @ b, -1, 1))
        d_c = np.arccos(np.clip(m @ c, -1, 1))
        
        # The circumradius is the max of these? No, they should be equal.
        # Due to numerical errors, they might differ slightly.
        r = max(d_a, d_b, d_c)
        
        if r > max_radius:
            max_radius = r
            
    return max_radius

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_radius = _compute_covering_radius(points)
    
    # Lloyd's Algorithm
    for iteration in range(1000):
        if time.time() > deadline:
            break
            
        try:
            hull = ConvexHull(points)
        except Exception:
            break
            
        # For each point, find adjacent triangles
        # hull.simplices is (F, 3)
        # We need a mapping from vertex index to list of triangle indices
        vertex_to_triangles = [[] for _ in range(N)]
        for idx, simplex in enumerate(hull.simplices):
            for v in simplex:
                vertex_to_triangles[v].append(idx)
                
        new_points = np.zeros_like(points)
        
        for i in range(N):
            triangles = vertex_to_triangles[i]
            if not triangles:
                new_points[i] = points[i]
                continue
                
            # Get circumcenters of adjacent triangles
            circumcenters = []
            for t_idx in triangles:
                simplex = hull.simplices[t_idx]
                a = points[simplex[0]]
                b = points[simplex[1]]
                c = points[simplex[2]]
                
                v = np.cross(a - b, b - c)
                m = _normalize(v)
                
                # Ensure m is on the same side as the triangle normal
                # Normal of triangle
                n = np.cross(b - a, c - a)
                if np.dot(m, n) < 0:
                    m = -m
                    
                circumcenters.append(m)
                
            circumcenters = np.array(circumcenters)
            
            # Compute spherical centroid
            # Simple average and normalize?
            # This is an approximation. A better way is to integrate over the spherical polygon.
            # But simple average is a good first step.
            centroid = np.sum(circumcenters, axis=0)
            centroid = _normalize(centroid)
            
            new_points[i] = centroid
            
        # Check if new points are valid (not coincident, etc)
        # And if they improve the radius
        new_radius = _compute_covering_radius(new_points)
        
        if new_radius < best_radius:
            best_radius = new_radius
            best_points = new_points.copy()
            points = new_points
        else:
            # If no improvement, maybe reduce step size or stop
            # For now, just break or try a smaller step
            # Let's try to interpolate
            # Actually, if Lloyd's doesn't improve, it might be stuck.
            # We can try a random restart or perturbation.
            # But for now, let's just break if no improvement for several iterations.
            # To keep it simple, we'll just break.
            # Or we can try to move halfway.
            half_points = (points + new_points) / 2.0
            half_points = np.array([_normalize(p) for p in half_points])
            half_radius = _compute_covering_radius(half_points)
            if half_radius < best_radius:
                best_radius = half_radius
                best_points = half_points.copy()
                points = half_points
            else:
                break
                
    return best_points
