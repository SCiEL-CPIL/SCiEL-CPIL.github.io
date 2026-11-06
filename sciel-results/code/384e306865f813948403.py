"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial import cKDTree
import time

def _covering_radius(points):
    """Compute the covering radius in radians for points on the unit sphere.
    Uses the convex hull facets: for each facet, find the farthest point on the
    corresponding spherical cap (the circumradius of the facet's spherical triangle)."""
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi  # invalid, return large value
    
    max_radius = 0.0
    for simplex in hull.simplices:
        # Get the vertices of this facet
        verts = points[simplex]
        
        # For a spherical triangle, the farthest point from all vertices is the
        # circumcenter of the spherical triangle (if it lies inside) or a vertex.
        # We compute the circumradius of the spherical triangle.
        
        # Compute the circumcenter of the triangle on the sphere.
        # The circumcenter is the intersection of the sphere with the plane
        # equidistant from all three vertices.
        
        a, b, c = verts[0], verts[1], verts[2]
        
        # Compute side lengths (angular distances)
        def ang_dist(u, v):
            dot = np.clip(np.dot(u, v), -1.0, 1.0)
            return np.arccos(dot)
        
        A = ang_dist(b, c)  # opposite to a
        B = ang_dist(a, c)  # opposite to b
        C = ang_dist(a, b)  # opposite to c
        
        # Spherical circumradius: R = arccos((cos A * cos B * cos C + 1) / (2 * cos A * cos B + 2 * cos B * cos C + 2 * cos C * cos A - 1))
        # Actually, use the formula: for a spherical triangle with sides a,b,c (angular),
        # the circumradius R satisfies:
        # cos R = (cos a * cos b * cos c + 1) / (2*(cos a*cos b + cos b*cos c + cos c*cos a) - 1)
        # Wait, let me use a more reliable approach.
        
        # Alternative: find the circumcenter by solving for a point x on the sphere
        # such that |x-a| = |x-b| = |x-c|.
        # This means x is equidistant from a, b, c, so x is along the normal to the plane
        # containing a, b, c... no, that's for planar.
        
        # For spherical: the circumcenter is the unit vector in the direction of
        # a*(sin B * sin C) + b*(sin C * sin A) + c*(sin A * sin B) ... not quite.
        
        # Let's use the formula from spherical trigonometry.
        # Given a spherical triangle with angular side lengths a, b, c (opposite to vertices A, B, C),
        # the circumradius R is given by:
        # cos R = (cos a * cos b * cos c + 1) / (2 * cos a * cos b + 2 * cos b * cos c + 2 * cos c * cos a - 1)
        
        cos_a = np.cos(A)
        cos_b = np.cos(B)
        cos_c = np.cos(C)
        
        numerator = cos_a * cos_b * cos_c + 1.0
        denominator = 2.0 * (cos_a * cos_b + cos_b * cos_c + cos_c * cos_a) - 1.0
        
        if abs(denominator) < 1e-12:
            # Degenerate case, use max of angular distances to vertices
            max_d = max(A, B, C)
            max_radius = max(max_radius, max_d)
            continue
        
        cos_R = numerator / denominator
        cos_R = np.clip(cos_R, -1.0, 1.0)
        R = np.arccos(cos_R)
        
        # Check if the circumcenter lies inside the spherical triangle.
        # If not, the farthest point is one of the vertices, so the covering radius
        # for this facet is the max angular distance from the vertex to opposite side... 
        # Actually, if circumcenter is outside, the farthest point on the cap is a vertex.
        # For now, use R as an upper bound.
        
        max_radius = max(max_radius, R)
    
    return max_radius


def _lloyd_step(points, iters=50):
    """Perform Lloyd's algorithm iterations to minimize mean distance."""
    points = points.copy()
    for _ in range(iters):
        # Compute Voronoi regions using convex hull
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        
        # For each point, find the centroid of its Voronoi cell on the sphere
        # Approximation: use the average of neighboring points weighted by proximity
        # Better: use the circumcenters of adjacent facets
        
        new_points = np.zeros_like(points)
        
        # Build adjacency: for each point, find which facets it belongs to
        point_facets = [[] for _ in range(len(points))]
        for i, simplex in enumerate(hull.simplices):
            for v in simplex:
                point_facets[v].append(i)
        
        for i in range(len(points)):
            if not point_facets[i]:
                new_points[i] = points[i]
                continue
            
            # Compute the centroid of the Voronoi cell as the average of the 
            # circumcenters of the facets sharing this point
            circumcenters = []
            for fi in point_facets[i]:
                simplex = hull.simplices[fi]
                verts = points[simplex]
                a, b, c = verts[0], verts[1], verts[2]
                
                # Compute circumcenter of spherical triangle
                # Use the formula: circumcenter direction = a*sin(B)*sin(C) + b*sin(C)*sin(A) + c*sin(A)*sin(B)
                # where A, B, C are the angles at vertices a, b, c
                # Actually, let's compute it differently.
                
                # The circumcenter is the unit vector n such that n·a = n·b = n·c
                # This means n is perpendicular to (a-b) and (a-c), so n is along (a-b) × (a-c)
                # But that gives the normal to the plane, not the circumcenter on the sphere.
                
                # Correct approach: the circumcenter is the point on the sphere equidistant from a, b, c.
                # It lies in the plane of a, b, c? No, it's on the sphere.
                
                # Let's solve: find unit vector x such that x·a = x·b = x·c = t
                # This is a linear system. x·(a-b) = 0, x·(a-c) = 0, and ||x||=1.
                # So x is parallel to (a-b) × (a-c), normalized.
                
                n = np.cross(a - b, a - c)
                n_norm = np.linalg.norm(n)
                if n_norm < 1e-12:
                    continue
                n = n / n_norm
                
                # The circumcenter is either n or -n. Choose the one that is "inside" the triangle.
                # Check which one is closer to the centroid of the triangle.
                centroid = (a + b + c) / 3.0
                if np.dot(n, centroid) < 0:
                    n = -n
                
                circumcenters.append(n)
            
            if circumcenters:
                new_points[i] = np.mean(circumcenters, axis=0)
                new_points[i] /= np.linalg.norm(new_points[i])
            else:
                new_points[i] = points[i]
        
        points = new_points
    
    return points


def _refine_covering(points, max_time=100.0):
    """Refine points to minimize covering radius using hull-based optimization."""
    start_time = time.time()
    points = points.copy()
    
    best_points = points.copy()
    best_radius = _covering_radius(points)
    
    # Iteratively move points to reduce the largest facet's circumradius
    for iteration in range(200):
        if time.time() - start_time > max_time:
            break
        
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        
        # Find the facet with the largest circumradius
        max_r = 0.0
        max_facet = None
        for i, simplex in enumerate(hull.simplices):
            verts = points[simplex]
            a, b, c = verts[0], verts[1], verts[2]
            
            def ang_dist(u, v):
                dot = np.clip(np.dot(u, v), -1.0, 1.0)
                return np.arccos(dot)
            
            A = ang_dist(b, c)
            B = ang_dist(a, c)
            C = ang_dist(a, b)
            
            cos_a = np.cos(A)
            cos_b = np.cos(B)
            cos_c = np.cos(C)
            
            numerator = cos_a * cos_b * cos_c + 1.0
            denominator = 2.0 * (cos_a * cos_b + cos_b * cos_c + cos_c * cos_a) - 1.0
            
            if abs(denominator) < 1e-12:
                r = max(A, B, C)
            else:
                cos_R = numerator / denominator
                cos_R = np.clip(cos_R, -1.0, 1.0)
                r = np.arccos(cos_R)
            
            if r > max_r:
                max_r = r
                max_facet = simplex
        
        if max_facet is None:
            break
        
        # Move the points of the largest facet toward the circumcenter
        verts = points[max_facet]
        a, b, c = verts[0], verts[1], verts[2]
        
        # Compute circumcenter
        n = np.cross(a - b, a - c)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            break
        n = n / n_norm
        centroid = (a + b + c) / 3.0
        if np.dot(n, centroid) < 0:
            n = -n
        
        # Move each vertex toward the circumcenter
        for idx, v in zip(max_facet, verts):
            new_v = v + 0.1 * (n - v)
            new_v /= np.linalg.norm(new_v)
            points[idx] = new_v
        
        # Check if this improved the covering radius
        new_radius = _covering_radius(points)
        if new_radius < best_radius:
            best_radius = new_radius
            best_points = points.copy()
    
    return best_points


def construct():
    start_time = time.time()
    deadline = 120.0  # seconds, with safety margin
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_radius = _covering_radius(points)
    
    # Try multiple random restarts with Lloyd + hull refinement
    np.random.seed(42)
    for restart in range(5):
        if time.time() - start_time > deadline - 10:
            break
        
        # Generate random initial points
        if restart == 0:
            pts = points.copy()
        else:
            pts = np.random.randn(N, 3)
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        
        # Lloyd's algorithm for better distribution
        if time.time() - start_time < deadline - 20:
            pts = _lloyd_step(pts, iters=30)
        
        # Hull-based refinement
        if time.time() - start_time < deadline - 5:
            refined = _refine_covering(pts, max_time=deadline - time.time() - 5)
            rad = _covering_radius(refined)
            if rad < best_radius:
                best_radius = rad
                best_points = refined
    
    # Ensure all points are on the unit sphere
    best_points /= np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points
# EVOLVE-BLOCK-END
