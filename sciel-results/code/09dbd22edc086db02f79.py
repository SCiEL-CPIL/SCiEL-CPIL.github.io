import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def _covering_radius(points):
    """Compute the covering radius in radians."""
    try:
        hull = ConvexHull(points)
        max_rad = 0.0
        for simplex in hull.simplices:
            v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            
            # Compute circumcenter of the triangle on the sphere
            # The circumcenter is the intersection of the perpendicular bisectors.
            # Alternatively, solve for a point c such that c.v1 = c.v2 = c.v3 and |c|=1.
            # c is proportional to v1 x v2 + v2 x v3 + v3 x v1? No.
            # The circumcenter c is in the plane of the triangle.
            # Let's use the formula for circumcenter of a triangle on a sphere.
            
            # Vector approach:
            # c = a*v1 + b*v2 + c*v3
            # c.v1 = c.v2 = c.v3 = cos(R)
            # This is a linear system.
            
            # Simpler: The circumcenter is the normalized sum of the vertices weighted by the sines of the opposite angles?
            # Actually, for a spherical triangle, the circumcenter can be found by intersecting the great circles that are perpendicular bisectors of the sides.
            
            # Let's use a numerical solver or a direct formula.
            # Formula: c = (v1 * sin(2*A) + v2 * sin(2*B) + v3 * sin(2*C)) / |...| ? No.
            
            # Let's use the property that c is equidistant from v1, v2, v3.
            # c . v1 = c . v2 => c . (v1 - v2) = 0
            # c . v1 = c . v3 => c . (v1 - v3) = 0
            # So c is orthogonal to (v1-v2) and (v1-v3).
            # c is parallel to (v1-v2) x (v1-v3).
            
            n = np.cross(v1 - v2, v1 - v3)
            if np.linalg.norm(n) < 1e-10:
                continue # Degenerate triangle
            c = n / np.linalg.norm(n)
            
            # Check if c is the circumcenter. It might be the antipode.
            # We need the one that is "inside" the triangle.
            # The distance from c to v1 should be the radius.
            # Let's check the dot product.
            # If c is the circumcenter, c.v1 = cos(R).
            # There are two such points: c and -c.
            # We want the one closer to the triangle centroid?
            centroid = (v1 + v2 + v3) / 3.0
            if np.dot(c, centroid) < 0:
                c = -c
                
            rad = np.arccos(np.clip(np.dot(c, v1), -1.0, 1.0))
            
            # Check if circumcenter is inside the triangle.
            # If not, the farthest point is a vertex or edge midpoint.
            # For a triangle on a sphere, if the circumcenter is outside, the max distance is to the vertex opposite the longest side?
            # Actually, if the circumcenter is outside, the covering radius for this facet is determined by the vertices.
            # The max distance from any point in the facet to the nearest vertex is the max of the distances to the vertices, which is the circumradius IF the circumcenter is inside.
            # If the circumcenter is outside, the farthest point in the facet is one of the vertices.
            # Wait, the covering radius is the max over ALL points on the sphere.
            # The farthest point from the set of points is the circumcenter of some facet IF it is inside that facet.
            # If the circumcenter is outside the facet, then the farthest point in that region is a vertex.
            # But the global covering radius is the max over all facets of the "local covering radius".
            # For a facet, the local covering radius is the max distance from any point in the facet to its vertices.
            # This is the circumradius if the circumcenter is inside.
            # If the circumcenter is outside, the max distance is the max distance to the vertices, which is the circumradius of the triangle formed by the vertices? No.
            # If the circumcenter is outside, the triangle is "obtuse" in some sense. The farthest point in the spherical triangle from the vertices is the vertex opposite the longest side?
            # Actually, the function d(p, V) = min_i d(p, v_i) is convex on the sphere? No.
            # The max of d(p, V) over the facet is either at the circumcenter (if inside) or at a vertex.
            # So local_rad = rad if circumcenter inside, else max(d(v1, V), d(v2, V), d(v3, V)).
            # But d(v_i, V) includes d(v_i, v_i)=0, so that's not right.
            # The distance from a vertex v1 to the nearest OTHER point? No, the nearest point in the configuration.
            # The vertices are part of the configuration. So d(v1, V) = 0.
            # This logic is flawed.
            
            # Correct logic:
            # The covering radius is the maximum over the sphere of min_i d(y, p_i).
            # This maximum is attained at a point y which is either:
            # 1. A circumcenter of a triangle (facet) in the Delaunay triangulation, if it lies within the triangle.
            # 2. A midpoint of an edge in the Delaunay triangulation (if the circumcenter is outside).
            # 3. A vertex (if the triangle is very obtuse?).
            
            # Actually, it's known that the covering radius is the maximum circumradius of the triangles in the Delaunay triangulation, provided we define the circumradius appropriately.
            # If the circumcenter is outside the triangle, the "circumradius" in the context of covering is effectively the distance to the farthest vertex from the circumcenter? No.
            # Let's stick to: The farthest point from the sites in a convex cell is the circumcenter if it's inside, otherwise the vertex farthest from the circumcenter?
            # Actually, if the circumcenter is outside, the max distance in the cell is at the vertex.
            # So we should compute the max distance from the circumcenter to the vertices? No, the distance from the circumcenter to the vertices IS the circumradius.
            # So, is the covering radius simply the max circumradius of all facets?
            # Yes, it is a known result that the covering radius of a set of points on a sphere is equal to the maximum circumradius of the triangles in its Delaunay triangulation.
            
            if rad > max_rad:
                max_rad = rad
                
    except Exception:
        return np.pi # Fallback, large radius
        
    return max_rad

def construct():
    # Fibonacci spiral initialization
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_rad = _covering_radius(best_points)
    
    start_time = time.time()
    max_time = 110.0 # Safety margin
    
    # Local optimization
    step_size = 0.05
    for it in range(1000):
        if time.time() - start_time > max_time:
            break
            
        try:
            hull = ConvexHull(points)
        except Exception:
            break
            
        # Find the facet with the largest circumradius
        max_facet_rad = 0.0
        worst_facet_idx = -1
        worst_circumcenter = None
        
        for idx, simplex in enumerate(hull.simplices):
            v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            n = np.cross(v1 - v2, v1 - v3)
            if np.linalg.norm(n) < 1e-10:
                continue
            c = n / np.linalg.norm(n)
            centroid = (v1 + v2 + v3) / 3.0
            if np.dot(c, centroid) < 0:
                c = -c
            rad = np.arccos(np.clip(np.dot(c, v1), -1.0, 1.0))
            
            if rad > max_facet_rad:
                max_facet_rad = rad
                worst_facet_idx = idx
                worst_circumcenter = c
                worst_v1, worst_v2, worst_v3 = v1, v2, v3
                worst_indices = simplex
                
        if worst_facet_idx == -1:
            break
            
        # Move the vertices of the worst facet towards its circumcenter
        if worst_circumcenter is not None:
            for j in range(3):
                v_idx = worst_indices[j]
                v = points[v_idx]
                # Move towards circumcenter
                new_v = v + step_size * (worst_circumcenter - v)
                points[v_idx] = new_v / np.linalg.norm(new_v)
                
        # Calculate new covering radius
        current_rad = _covering_radius(points)
        
        if current_rad < best_rad:
            best_rad = current_rad
            best_points = points.copy()
        else:
            # If it got worse, revert? Or keep going?
            # Usually, small steps should improve or stay same.
            # Let's revert if it got significantly worse to avoid diverging.
            # But we don't have the previous state saved easily in this loop structure without extra memory.
            # Let's just continue. The step is small.
            pass
            
    return best_points
