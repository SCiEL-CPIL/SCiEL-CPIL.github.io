"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
import itertools

def _get_voronoi_centroids(points):
    """
    Computes the centroids of the spherical Voronoi cells for the given points.
    Each point is moved to the centroid of its Voronoi region.
    """
    n = len(points)
    new_points = np.zeros_like(points)
    
    # Precompute pairwise distances/angles to identify neighbors if needed, 
    # but for N=50, we can use a convex hull approach on the sphere or 
    # direct half-space intersection.
    
    # A simpler and robust approach for small N:
    # For each point i, find the region of the sphere closer to i than to any other point.
    # This region is a spherical polygon. Its vertices are the intersections of the 
    # great circle arcs that are perpendicular bisectors of the edges connecting i to its neighbors.
    
    # However, calculating spherical Voronoi vertices directly can be tricky.
    # An alternative is to use the fact that the centroid of a spherical polygon 
    # can be approximated or calculated exactly.
    
    # Let's use a more direct optimization approach: 
    # Minimize the maximum distance from any point on the sphere to the nearest site.
    # This is equivalent to finding the maximum circumradius of the Voronoi cells.
    
    # For Lloyd's, we move to the centroid.
    # To get the centroid, we can sample the Voronoi region densely and average, 
    # but that's slow.
    
    # Better: Use the property that the gradient of the covering radius is related 
    # to the Voronoi vertices.
    
    # Let's implement a standard Lloyd iteration using a convex hull trick.
    # Project points to 3D. The Voronoi cell of a point on the sphere corresponds 
    # to the normal cone of the face of the convex hull of the points? No, that's 
    # for Delaunay.
    
    # Let's use a simpler, effective heuristic: 
    # 1. Compute the Delaunay triangulation of the points on the sphere.
    # 2. The circumcenters of the Delaunay triangles are the vertices of the Voronoi cells.
    # 3. The centroid of a Voronoi cell is the average of its vertices (normalized).
    
    # SciPy's ConvexHull gives Delaunay triangulation of points in R3.
    # Since points are on the unit sphere, the convex hull of the points is a polyhedron.
    # The faces of this polyhedron correspond to the Delaunay triangles on the sphere.
    # The circumcenter of each face (triangle) is a vertex of the Voronoi diagram.
    
    try:
        hull = ConvexHull(points)
        simplices = hull.simplices
    except Exception:
        # If hull fails (e.g., points are coplanar or degenerate), return original points
        return points

    # For each point, collect the circumcenters of the faces (triangles) that contain it.
    # These circumcenters form the vertices of the Voronoi cell for that point.
    
    # First, compute circumcenters for all faces.
    circumcenters = []
    face_to_circumcenter = []
    
    for simplex in simplices:
        v1, v2, v3 = points[simplex]
        
        # Calculate circumcenter of triangle v1, v2, v3 on the sphere.
        # The circumcenter is the point equidistant from v1, v2, v3.
        # It lies in the plane defined by v1, v2, v3.
        # The vector from the origin to the circumcenter is perpendicular to the plane? No.
        # The circumcenter C satisfies |C - v1|^2 = |C - v2|^2 = |C - v3|^2 = R^2 (constant for the sphere? No, distance to vertices is chordal distance).
        # Actually, we want the point on the sphere that is the circumcenter of the spherical triangle.
        # This is the normalization of the cross product of (v2-v1) and (v3-v1)? No.
        
        # The circumcenter of a spherical triangle is the intersection of the great circles 
        # that are the perpendicular bisectors of the sides.
        # Alternatively, it is the direction of the vector:
        # c = (v1 + v2 + v3) is not correct.
        
        # The circumcenter vector is proportional to:
        # c = (v1 * |v2|^2 * |v3|^2 ... ) 
        # Since |v1|=|v2|=|v3|=1:
        # The circumcenter direction is given by the cross product of the edges?
        # Let's use the formula for the circumcenter of a triangle on a sphere.
        # The circumcenter is the normalized vector:
        # c = (v1 * sin(2*angle_1) + v2 * sin(2*angle_2) + v3 * sin(2*angle_3)) ? No.
        
        # Simpler: The circumcenter is the point where the three great circle bisectors meet.
        # It is also the pole of the great circle passing through the midpoints? No.
        
        # Let's calculate it via linear algebra.
        # The circumcenter C is in the span of v1, v2, v3.
        # C = a*v1 + b*v2 + c*v3
        # |C-v1|^2 = |C-v2|^2 => 2*C.(v2-v1) = |v2|^2 - |v1|^2 = 0 => C.(v2-v1) = 0
        # |C-v1|^2 = |C-v3|^2 => 2*C.(v3-v1) = 0 => C.(v3-v1) = 0
        # So C is orthogonal to (v2-v1) and (v3-v1).
        # Thus, C is parallel to the cross product (v2-v1) x (v3-v1).
        
        edge1 = v2 - v1
        edge2 = v3 - v1
        cross = np.cross(edge1, edge2)
        
        if np.linalg.norm(cross) < 1e-10:
            # Degenerate triangle
            cc = v1
        else:
            cc = cross / np.linalg.norm(cross)
            
        # We need to ensure cc is on the same side of the plane as the sphere center?
        # The circumcenter should be on the sphere. The cross product gives the normal to the plane.
        # Is it the correct direction?
        # The circumcenter is the point on the sphere equidistant from v1, v2, v3.
        # The vector (v2-v1) x (v3-v1) is perpendicular to the plane containing v1,v2,v3.
        # The circumcenter lies on the line passing through the origin and perpendicular to the plane?
        # No, the circumcenter lies in the plane of the triangle? No, on the sphere.
        # The chordal circumcenter is the intersection of the perpendicular bisectors in 3D space.
        # For points on a sphere centered at origin, the circumcenter on the sphere is indeed 
        # the normalization of the cross product of the edges?
        # Let's check. If v1, v2, v3 are on the unit sphere, the plane is at distance d from origin.
        # The circumcenter is the point on the sphere closest to the plane? No.
        # The circumcenter is the point where the three bisector great circles meet.
        # The vector (v2-v1) x (v3-v1) is normal to the plane.
        # The circumcenter C satisfies C . (v2-v1) = 0 and C . (v3-v1) = 0.
        # So C is parallel to (v2-v1) x (v3-v1).
        # Yes, this is correct.
        
        # Check if it's the correct pole. We might need to flip the sign.
        # The circumcenter should be "inside" the spherical triangle.
        # We can check the dot product with the centroid of the triangle (v1+v2+v3)/3.
        # If dot < 0, flip.
        centroid_dir = (v1 + v2 + v3) / 3.0
        if np.dot(cc, centroid_dir) < 0:
            cc = -cc
            
        circumcenters.append(cc)
        face_to_circumcenter.append(cc)

    # Now, for each point i, find the faces that contain i.
    # The Voronoi cell of i is the polygon formed by the circumcenters of these faces.
    # The centroid of this polygon is the target for Lloyd's iteration.
    
    # Map point index to list of circumcenters
    point_voronoi_vertices = [[] for _ in range(n)]
    
    for idx, simplex in enumerate(simplices):
        cc = face_to_circumcenter[idx]
        for p_idx in simplex:
            point_voronoi_vertices[p_idx].append(cc)

    # Calculate centroids
    for i in range(n):
        vertices = point_voronoi_vertices[i]
        if not vertices:
            # Should not happen for a valid hull
            new_points[i] = points[i]
        else:
            # Sum the vertices and normalize
            sum_vec = sum(vertices)
            norm = np.linalg.norm(sum_vec)
            if norm < 1e-10:
                new_points[i] = points[i]
            else:
                new_points[i] = sum_vec / norm

    return new_points

def _compute_covering_radius(points):
    """
    Computes the covering radius of the given points on the sphere.
    The covering radius is the maximum distance from any point on the sphere 
    to its nearest site. This is equal to the maximum circumradius of the 
    Voronoi cells (or Delaunay triangles).
    """
    try:
        hull = ConvexHull(points)
        simplices = hull.simplices
    except Exception:
        return np.inf

    max_radius = 0.0
    for simplex in simplices:
        v1, v2, v3 = points[simplex]
        
        # Calculate the circumradius (angular distance from circumcenter to vertices)
        # The circumcenter is the normalized cross product as before.
        edge1 = v2 - v1
        edge2 = v3 - v1
        cross = np.cross(edge1, edge2)
        
        if np.linalg.norm(cross) < 1e-10:
            continue
            
        cc = cross / np.linalg.norm(cross)
        
        # Determine correct sign
        centroid_dir = (v1 + v2 + v3) / 3.0
        if np.dot(cc, centroid_dir) < 0:
            cc = -cc
            
        # Angular distance from cc to v1
        # cos(theta) = dot(cc, v1)
        cos_theta = np.dot(cc, v1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        
        if theta > max_radius:
            max_radius = theta
            
    return max_radius

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # Initial Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_radius = _compute_covering_radius(best_points)
    
    # Lloyd's Iteration
    # We will run iterations until the radius stops improving or time runs out.
    # To avoid getting stuck in local minima, we can add small perturbations 
    # or use multiple restarts if time permits.
    
    # For now, single run with Lloyd's.
    # We can also try to optimize the step size or use a gradient descent on the radius.
    
    # Let's add a simple gradient descent step if Lloyd's stalls.
    # But Lloyd's is usually enough for N=50.
    
    for iteration in range(200):
        if time.time() > deadline:
            break
            
        new_points = _get_voronoi_centroids(points)
        
        # Check for improvement
        new_radius = _compute_covering_radius(new_points)
        
        if new_radius < best_radius:
            best_radius = new_radius
            best_points = new_points.copy()
            points = new_points
        else:
            # If no improvement, we might be stuck. 
            # We can try to perturb the points slightly.
            # Or break if we've stagnated for a while.
            # Let's add a small random perturbation to escape local minima.
            noise = np.random.normal(0, 0.01, size=points.shape)
            perturbed = points + noise
            perturbed /= np.linalg.norm(perturbed, axis=1, keepdims=True)
            
            perturbed_radius = _compute_covering_radius(perturbed)
            if perturbed_radius < best_radius:
                best_radius = perturbed_radius
                best_points = perturbed.copy()
                points = perturbed
            else:
                # Stagnation. Let's try a few more iterations with smaller noise or break.
                # For simplicity, let's break if we don't improve for a few steps.
                # But we don't track stagnation count here.
                # Let's just break to save time for potential other strategies if we had them.
                # Since we only have Lloyd's, breaking early is fine if we're stuck.
                # However, Lloyd's can sometimes oscillate.
                # Let's continue for a few more steps to see if it recovers.
                pass
                
    # Final check
    final_radius = _compute_covering_radius(best_points)
    if final_radius < best_radius:
        best_radius = final_radius
        
    return best_points

# EVOLVE-BLOCK-END
