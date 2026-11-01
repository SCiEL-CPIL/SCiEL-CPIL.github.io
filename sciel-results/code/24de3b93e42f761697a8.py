"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull, Delaunay

def _covering_radius(pts):
    """Compute the covering radius in degrees for a set of points on the unit sphere."""
    # Delaunay triangulation on the sphere
    try:
        tri = Delaunay(pts)
    except Exception:
        return np.pi  # fallback: worst case
    
    # For each simplex (tetrahedron in 3D), compute the circumradius
    # The covering radius is the maximum circumradius of all facets
    # Actually, we need to find the maximum distance from any point on the sphere
    # to the nearest point in pts. This equals the maximum circumradius of the
    # spherical Voronoi cells, which corresponds to the circumcenters of the
    # Delaunay simplices projected onto the sphere.
    
    # A simpler approach: the covering radius is the maximum over all points y on the sphere
    # of min_i angle(y, pts[i]). This is achieved at a circumcenter of a Delaunay simplex
    # (or at a vertex if the cell contains the center).
    
    # For each simplex in the Delaunay triangulation, compute the circumcenter
    # and check if it's valid (i.e., the cap is empty).
    
    max_radius = 0.0
    n = len(pts)
    
    for simplex in tri.simplices:
        # simplex is an array of 4 indices (tetrahedron)
        simplex = np.array(simplex)
        if len(simplex) < 2:
            continue
        
        # For a spherical Delaunay, the relevant facets are triangles (3 points)
        # But Delaunay in 3D gives tetrahedra. We need the boundary triangles.
        # Alternative: use the fact that the covering radius is determined by
        # the farthest point in each Voronoi cell, which is at the circumcenter
        # of a Delaunay simplex.
        
        # Let's compute the circumcenter of the tetrahedron formed by the 4 points
        # and project it onto the sphere.
        
        P = pts[simplex]  # 4x3 array
        
        # Circumcenter of a tetrahedron: solve for c such that |c - P[i]|^2 = R^2 for all i
        # This gives a linear system.
        # |c|^2 - 2*c·P[i] + |P[i]|^2 = R^2
        # Since |P[i]| = 1: |c|^2 - 2*c·P[i] + 1 = R^2
        # For i=0,1,2,3:
        # 2*c·(P[i] - P[0]) = |P[i]|^2 - |P[0]|^2 = 0 (since all on unit sphere)
        # So c·(P[i] - P[0]) = 0 for i=1,2,3
        
        # This means c is perpendicular to P[1]-P[0], P[2]-P[0], P[3]-P[0]
        # So c is parallel to the cross product of (P[1]-P[0]) and (P[2]-P[0])
        # But we also need it to be equidistant from all 4 points.
        
        # Actually, for points on the unit sphere, the circumcenter of the tetrahedron
        # that lies on the sphere (or closest to it) gives us the direction.
        
        # Let's use a different approach: the circumcenter c satisfies:
        # c · (P[i] - P[0]) = 0 for i = 1, 2, 3
        # And we want to find the point on the sphere in this direction that maximizes
        # the minimum distance to the 4 points.
        
        # The direction is given by the null space of the 3x3 matrix [P[1]-P[0]; P[2]-P[0]; P[3]-P[0]]
        # i.e., the cross product of two of these vectors.
        
        v1 = P[1] - P[0]
        v2 = P[2] - P[0]
        v3 = P[3] - P[0]
        
        # Cross product to get a vector perpendicular to v1 and v2
        cross = np.cross(v1, v2)
        if np.linalg.norm(cross) < 1e-12:
            continue
        
        # Check if this is also perpendicular to v3
        # If not, the 4 points are not co-spherical in a way that gives a unique circumcenter
        # on the sphere. But they are on the unit sphere, so there should be a solution.
        
        # Actually, for 4 points on the unit sphere, the circumcenter of the tetrahedron
        # is the origin if they form a regular tetrahedron, otherwise it's somewhere else.
        # The point on the sphere closest to the circumcenter (in terms of angular distance)
        # is what we want.
        
        # Let's solve the linear system properly.
        # We want c such that c · (P[i] - P[0]) = 0 for i=1,2,3
        # This is a homogeneous system. The solution space is 1-dimensional (if the
        # 3 vectors are linearly independent), so c is proportional to the cross product.
        
        # But we need the specific point on the unit sphere. The circumcenter of the
        # spherical triangle (or tetrahedron) is the point equidistant from all vertices.
        
        # Let's try a different approach: for each face of the Delaunay triangulation
        # (triangle), compute the circumcenter of that triangle on the sphere.
        
        # Actually, let's use a more direct method: sample many points and find the
        # maximum minimum distance. But this is expensive.
        
        # Better: the covering radius is the maximum circumradius of the spherical
        # Voronoi cells. For each Delaunay simplex, the circumcenter (if it lies
        # within the simplex) gives a candidate.
        
        # For a triangle on the sphere, the circumcenter is the point equidistant
        # from the 3 vertices. It lies in the plane of the triangle, at the intersection
        # of the perpendicular bisectors.
        
        # Let's compute for each triangle (face) of the convex hull.
        break  # This approach is getting complicated. Let's use a simpler method.
    
    # Simpler method: use the fact that the covering radius equals the maximum
    # circumradius of the facets of the convex hull. For each facet (triangle),
    # compute the circumradius and check if the circumcenter (projected onto the
    # sphere) is within the cap.
    
    # Even simpler: use a grid-based approximation.
    # Generate a fine grid of points on the sphere and compute the minimum distance
    # to the nearest point in pts for each grid point. The maximum of these is
    # an upper bound on the covering radius.
    
    # For a more accurate estimate, we can use the Delaunay triangulation and
    # compute the circumcenters of the triangles.
    
    # Let's use the convex hull approach.
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi
    
    max_radius = 0.0
    for simplex in hull.simplices:
        # Each simplex is a triangle (3 points) on the convex hull
        P = pts[simplex]  # 3x3 array
        
        # Compute the circumcenter of the triangle on the sphere
        # The circumcenter is the point on the sphere equidistant from all 3 vertices.
        # It lies in the plane of the triangle.
        
        # For a triangle on the unit sphere, the circumcenter can be found by:
        # 1. Finding the plane of the triangle
        # 2. Finding the circumcenter of the planar triangle
        # 3. Projecting onto the sphere
        
        # Normal of the plane
        n = np.cross(P[1] - P[0], P[2] - P[0])
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-12:
            continue
        n = n / norm_n
        
        # The circumcenter of the planar triangle:
        # Solve for c such that |c - P[i]|^2 = R^2 for i=0,1,2
        # This gives: c·(P[i] - P[0]) = (|P[i]|^2 - |P[0]|^2) / 2 = 0 (since all on unit sphere)
        # So c·(P[1]-P[0]) = 0 and c·(P[2]-P[0]) = 0
        # c is parallel to n.
        
        # The circumcenter in the plane is at distance d from the origin along n.
        # |c|^2 - 2*c·P[0] + |P[0]|^2 = R^2
        # |c|^2 - 2*(c·n)*n·P[0] + 1 = R^2
        # If c = t*n, then t^2 - 2*t*(n·P[0]) + 1 = R^2
        
        # We want the point on the unit sphere, so |c| = 1.
        # t^2 = 1, so t = ±1.
        # The circumcenter on the sphere is ±n, but we need the one that is
        # equidistant from all 3 points.
        
        # Actually, for a triangle on the unit sphere, the circumcenter is the
        # point on the sphere in the direction of the normal (or its negative)
        # that is equidistant from the 3 vertices.
        
        # The angular distance from n to P[i] is the same for all i if n is the
        # circumcenter. Let's check both n and -n.
        
        for sign in [1, -1]:
            c = sign * n
            # Check if c is within the cap (i.e., the cap centered at c with
            # radius = max angular distance to the 3 vertices contains no other points)
            # For now, just compute the circumradius.
            angles = np.arccos(np.clip(np.abs(pts @ c), -1, 1))
            # The circumradius is the maximum angle from c to the 3 vertices
            circumradius = np.max(angles[simplex])
            
            # Check if c is a valid circumcenter (empty cap)
            # The cap is empty if no other point is closer to c than the 3 vertices
            min_angle_to_c = np.min(angles)
            # If min_angle_to_c > circumradius, then the cap is not empty
            # (some other point is inside the cap)
            # Actually, we want the cap centered at c with radius circumradius to
            # contain only the 3 vertices. So we need to check that no other point
            # is within circumradius of c.
            if np.min(angles) < circumradius - 1e-9:
                # Some other point is inside the cap, so this is not a valid circumcenter
                continue
            
            max_radius = max(max_radius, circumradius)
    
    return np.degrees(max_radius)


def _initial_fib():
    """Golden-angle (Fibonacci) spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _improve(pts, max_time=120.0):
    """Improve the configuration using local optimization."""
    start_time = time.time()
    best_pts = pts.copy()
    best_radius = _covering_radius(best_pts)
    
    # Use a simple gradient-free optimization: for each point, try moving it
    # slightly in random directions and keep the move if it improves the radius.
    
    for iteration in range(10000):
        if time.time() - start_time > max_time - 5.0:
            break
        
        # Pick a random point
        idx = np.random.randint(N)
        
        # Try a small random perturbation
        for attempt in range(10):
            delta = np.random.randn(3) * 0.01
            new_pts = best_pts.copy()
            new_pts[idx] += delta
            # Normalize
            new_pts[idx] /= np.linalg.norm(new_pts[idx])
            
            # Check if the move improves the covering radius
            new_radius = _covering_radius(new_pts)
            if new_radius < best_radius - 1e-10:
                best_pts = new_pts
                best_radius = new_radius
                break
    
    return best_pts, best_radius


def construct():
    """Construct 130 points on the unit sphere to minimize the covering radius."""
    start_time = time.time()
    max_time = 120.0
    
    # Start with Fibonacci spiral
    pts = _initial_fib()
    current_radius = _covering_radius(pts)
    
    # Try multiple random restarts with local optimization
    best_pts = pts
    best_radius = current_radius
    
    while time.time() - start_time < max_time - 10.0:
        # Generate a new random configuration
        pts = np.random.randn(N, 3)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        
        # Local optimization
        improved_pts, improved_radius = _improve(pts, max_time=max_time - (time.time() - start_time) - 5.0)
        
        if improved_radius < best_radius:
            best_pts = improved_pts
            best_radius = improved_radius
    
    # Ensure the result is valid
    # Check that no two points are coincident
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(best_pts[i] - best_pts[j]) < 1e-10:
                # Perturb one of the points
                best_pts[j] += np.random.randn(3) * 0.001
                best_pts[j] /= np.linalg.norm(best_pts[j])
    
    # Normalize all points
    best_pts /= np.linalg.norm(best_pts, axis=1, keepdims=True)
    
    return best_pts
# EVOLVE-BLOCK-END
