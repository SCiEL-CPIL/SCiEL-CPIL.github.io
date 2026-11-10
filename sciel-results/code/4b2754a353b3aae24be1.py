"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time

def _normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-15:
        return v
    return v / n

def _covering_radius(points):
    """Compute the covering radius in degrees for points on the unit sphere.
    
    The covering radius is the maximum over all points y on the sphere of the
    angular distance from y to the nearest configuration point. This is equivalent
    to the largest circumradius of the facets of the convex hull.
    """
    points = np.array(points)
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    # Check for coincident points
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            if np.linalg.norm(points[i] - points[j]) < 1e-10:
                # Coincident points - return a large value
                return 180.0
    
    # Check if all points are in a closed hemisphere
    # This is equivalent to checking if the origin is inside the convex hull
    # We'll use the convex hull approach
    
    try:
        hull = ConvexHull(points)
    except Exception:
        # If hull computation fails, fall back to brute force
        return _covering_radius_brute(points)
    
    # For each facet (simplex) of the convex hull, compute the circumradius
    # The covering radius is the maximum circumradius over all facets
    
    max_circumradius = 0.0
    
    for simplex in hull.simplices:
        simplex_points = points[simplex]
        
        # Compute the circumradius of this simplex
        # For a simplex in 3D (tetrahedron), we need to find the circumsphere
        # The circumcenter is equidistant from all vertices
        
        # Method: solve for the center c such that |c - p_i|^2 = R^2 for all i
        # This gives a linear system
        
        n = len(simplex_points)
        
        if n < 3:
            # Degenerate case
            continue
        
        # Set up the linear system:
        # |c|^2 - 2 c·p_i + |p_i|^2 = R^2
        # Since |p_i| = 1: |c|^2 - 2 c·p_i + 1 = R^2
        # So: 2 c·p_i = |c|^2 + 1 - R^2
        # Subtracting equations for i and j:
        # 2 c·(p_i - p_j) = 0 ... wait, that's not right.
        
        # Let me redo this. For points on the unit sphere:
        # |c - p_i|^2 = R^2
        # |c|^2 - 2 c·p_i + |p_i|^2 = R^2
        # |c|^2 - 2 c·p_i + 1 = R^2
        # So: 2 c·p_i = |c|^2 + 1 - R^2 = K (constant)
        
        # Therefore: 2 c·(p_i - p_j) = 0 for all i, j
        # This means c is orthogonal to all (p_i - p_j), so c is parallel to the 
        # normal of the plane containing the points... but that's only for coplanar points.
        
        # Actually, for a simplex in 3D, the vertices are not coplanar (unless degenerate).
        # Let me use a different approach.
        
        # The circumcenter c satisfies:
        # |c - p_0|^2 = |c - p_i|^2 for i = 1, ..., n-1
        # This gives: -2c·p_0 + |p_0|^2 = -2c·p_i + |p_i|^2
        # Since |p_i| = 1: 2c·(p_i - p_0) = 0 for i = 1, ..., n-1
        
        # Wait, that's still not right. Let me be more careful.
        # |c - p_0|^2 = |c|^2 - 2c·p_0 + |p_0|^2
        # |c - p_i|^2 = |c|^2 - 2c·p_i + |p_i|^2
        # Setting equal: -2c·p_0 + |p_0|^2 = -2c·p_i + |p_i|^2
        # 2c·(p_i - p_0) = |p_i|^2 - |p_0|^2 = 0 (since all on unit sphere)
        
        # So c is orthogonal to all (p_i - p_0) for i = 1, ..., n-1.
        # This means c is parallel to the cross product of two of these vectors,
        # or more generally, c lies along the line perpendicular to the affine hull.
        
        # For a tetrahedron (n=4), we have 3 vectors (p_1-p_0, p_2-p_0, p_3-p_0)
        # and c must be orthogonal to all three, so c = λ * (p_1-p_0) × (p_2-p_0)
        # (or any vector orthogonal to all three).
        
        # Actually, for n points in 3D forming a simplex, we have n-1 independent vectors
        # from p_0 to p_i. If n=4, we have 3 vectors, and c is orthogonal to all, so
        # c is along the normal to the plane... but wait, 3 vectors in 3D span the space,
        # so the only vector orthogonal to all three is the zero vector. That can't be right.
        
        # I think I'm making an error. Let me reconsider.
        
        # The condition is that c is equidistant from all p_i. This is a system of
        # n-1 equations (comparing each p_i to p_0):
        # 2c·(p_i - p_0) = |p_i|^2 - |p_0|^2 = 0
        
        # So we have n-1 linear equations in the 3 unknowns c_x, c_y, c_z.
        # For n=4, we have 3 equations, which should determine c uniquely (if consistent).
        
        # Let's set up the matrix A where row i is (p_i - p_0) for i = 1, ..., n-1.
        # Then A @ c = 0.
        
        # But this gives a homogeneous system, so the only solution is c = 0 (if A is full rank).
        # That's not right either.
        
        # I think the issue is that I'm working with vectors from p_0, but the circumcenter
        # is not necessarily at the origin. Let me use a different formulation.
        
        # The circumcenter c satisfies |c - p_i|^2 = R^2 for all i.
        # Expanding: |c|^2 - 2c·p_i + |p_i|^2 = R^2
        # Since |p_i| = 1: |c|^2 - 2c·p_i + 1 = R^2
        # So: 2c·p_i = |c|^2 + 1 - R^2
        
        # Let K = |c|^2 + 1 - R^2. Then 2c·p_i = K for all i.
        # This means c·p_i = K/2 for all i.
        
        # So c is a vector such that its dot product with each p_i is the same constant.
        # This is a system of n equations in 3 unknowns (c_x, c_y, c_z) plus the unknown K.
        
        # Actually, we can eliminate K by subtracting equations:
        # c·(p_i - p_j) = 0 for all i, j.
        
        # So c is orthogonal to all differences p_i - p_j.
        
        # For a simplex in 3D, the vectors p_i - p_0 for i = 1, ..., n-1 span a subspace.
        # If n=4, they span all of R^3, so c must be 0. But that's only if the simplex
        # is "full rank" in some sense.
        
        # I think the correct approach is to solve the linear system directly.
        
        # Let's use the following method:
        # The circumcenter c minimizes the maximum distance to the vertices, but for a simplex,
        # it's the unique point equidistant from all vertices.
        
        # Set up: |c - p_i|^2 = R^2 for i = 0, ..., n-1
        # Subtract the equation for i=0 from each other:
        # |c|^2 - 2c·p_i + |p_i|^2 - |c|^2 + 2c·p_0 - |p_0|^2 = 0
        # -2c·(p_i - p_0) + (|p_i|^2 - |p_0|^2) = 0
        # 2c·(p_i - p_0) = |p_i|^2 - |p_0|^2
        
        # Since all |p_i| = 1: 2c·(p_i - p_0) = 0
        
        # So we have the system:
        # [p_1 - p_0; p_2 - p_0; ...; p_{n-1} - p_0] @ c = 0
        
        # This is a homogeneous system. For n=4, we have 3 equations in 3 unknowns.
        # The only solution is c = 0 if the matrix is invertible.
        
        # But that would mean the circumcenter is at the origin, which is only true if the
        # simplex is "centered" at the origin.
        
        # I think I need to be more careful. The circumcenter of a simplex inscribed in a
        # sphere is NOT generally at the center of that sphere. The circumradius is the radius
        # of the smallest sphere containing the simplex, which for points on the unit sphere
        # is related to the angular size of the facet.
        
        # Let me use a different approach: compute the circumradius directly.
        
        # For a tetrahedron with vertices p_0, p_1, p_2, p_3 on the unit sphere,
        # the circumradius R satisfies:
        # R = (a * b * c) / (6 * V)
        # where a, b, c are the edge lengths of the base triangle, and V is the volume.
        # Wait, that's not quite right either.
        
        # Actually, for a simplex in n dimensions, the circumradius can be computed as:
        # R^2 = |c|^2 where c is the circumcenter.
        
        # Let me just solve the linear system numerically.
        
        # The circumcenter c satisfies:
        # |c - p_i|^2 = |c - p_0|^2 for i = 1, ..., n-1
        # Expanding and simplifying (since |p_i| = |p_0| = 1):
        # c·(p_i - p_0) = 0 for i = 1, ..., n-1
        
        # So c is in the null space of the matrix with rows (p_i - p_0).
        
        # For n=4 in 3D, this matrix is 3x3, and if it's invertible, c = 0.
        # But that's not generally true.
        
        # I think the issue is that I'm confusing the circumcenter of the simplex with
        # the center of the sphere on which the vertices lie.
        
        # Let me use a more robust method: use the fact that the covering radius is the
        # maximum, over all points on the sphere, of the distance to the nearest point.
        # This is equivalent to the inradius of the Voronoi diagram on the sphere, or
        # the maximum circumradius of the convex hull facets.
        
        # For each facet (triangle or tetrahedron) of the convex hull, the relevant
        # circumradius is the radius of the smallest cap (spherical cap) that contains
        # the facet's vertices.
        
        # Actually, I think the correct interpretation is:
        # The covering radius is the maximum, over all facets F of the convex hull,
        # of the circumradius of F, where the circumradius of a facet is the radius
        # of the smallest spherical cap containing the facet's vertices.
        
        # For a triangle on the unit sphere, the circumradius is the angular radius
        # of the smallest cap containing the three vertices.
        
        # Let me compute this directly for each facet.
        
        # For a triangle with vertices a, b, c on the unit sphere, the circumradius
        # (angular) can be computed as follows:
        # The center of the minimal cap is the point on the sphere closest to the
        # centroid of the triangle (in some sense), or more precisely, the point
        # that maximizes the minimum angular distance to a, b, c.
        
        # Actually, for a spherical triangle, the circumradius is the angular radius
        # of the circumscribed circle, which is the circle on the sphere passing
        # through all three vertices.
        
        # The center of this circle is the point c on the sphere such that
        # c·a = c·b = c·c' (where c' is the center of the cap).
        
        # This is getting complicated. Let me use a simpler approach:
        # For each facet, compute the circumradius by finding the point on the sphere
        # that is equidistant (angularly) from all vertices of the facet.
        
        # For a triangle with vertices a, b, c, the circumcenter (on the sphere) is
        # the point p such that p·a = p·b = p·c.
        
        # This means p is in the null space of [a-b; b-c], or equivalently, p is
        # parallel to (a-b) × (b-c).
        
        # So p = normalize((a-b) × (b-c)).
        
        # Then the circumradius is the angular distance from p to a (or b, or c).
        
        # Let's implement this.
        
        if n == 3:
            # Triangle
            a, b, c = simplex_points
            # Circumcenter is parallel to (a-b) × (b-c)
            v = np.cross(a - b, b - c)
            v_norm = np.linalg.norm(v)
            if v_norm < 1e-10:
                # Degenerate triangle
                continue
            p = v / v_norm
            # Circumradius is the angular distance from p to a
            cos_theta = np.dot(p, a)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            theta = np.arccos(cos_theta)
            max_circumradius = max(max_circumradius, theta)
        
        elif n == 4:
            # Tetrahedron - this is more complex
            # For a tetrahedron, the "circumradius" in the context of covering
            # is the maximum, over points in the tetrahedron's interior (on the sphere),
            # of the distance to the nearest vertex.
            
            # Actually, for the convex hull of points on the sphere, the facets are
            # triangles (2D faces), not tetrahedra. The convex hull in 3D has triangular
            # facets.
            
            # Wait, ConvexHull.simplices gives the 2D faces (triangles) for a 3D hull.
            # So each simplex should have 3 vertices.
            
            # Let me check: for a 3D convex hull, hull.simplices are the facets,
            # which are triangles (3 vertices each).
            
            # So n should always be 3. Let me handle the case where n != 3.
            
            # For robustness, if n == 4, it might be a 3D simplex (tetrahedron),
            # but that shouldn't happen for a 3D convex hull's facets.
            
            # Let's just skip it for now.
            continue
    
    # Convert to degrees
    return np.degrees(max_circumradius)


def _covering_radius_brute(points):
    """Brute force computation of the covering radius by sampling the sphere."""
    points = np.array(points)
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    # Sample many points on the sphere
    n_samples = 10000
    # Use Fibonacci spiral for uniform sampling
    i = np.arange(n_samples) + 0.5
    z = 1.0 - 2.0 * i / n_samples
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    samples = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # For each sample, find the nearest point
    max_dist = 0.0
    for s in samples:
        dists = np.arccos(np.clip(points @ s, -1.0, 1.0))
        min_dist = np.min(dists)
        max_dist = max(max_dist, min_dist)
    
    return np.degrees(max_dist)


def _local_optimize(points, max_iters=1000, step_size=0.1):
    """Local optimization using gradient descent on the covering radius."""
    points = np.array(points, dtype=float)
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    best_points = points.copy()
    best_radius = _covering_radius(points)
    
    for it in range(max_iters):
        # Compute the covering radius and the point that achieves it
        # For simplicity, use a random perturbation approach
        
        # Pick a random point to move
        idx = np.random.randint(len(points))
        
        # Compute the gradient of the covering radius with respect to this point
        # This is complex, so use a simple perturbation
        
        # Try moving the point in a random direction
        for _ in range(10):
            # Random direction on the sphere
            direction = np.random.randn(3)
            direction = direction / np.linalg.norm(direction)
            
            # Move the point slightly
            new_point = points[idx] + step_size * direction
            new_point = new_point / np.linalg.norm(new_point)
            
            # Create a new configuration
            new_points = points.copy()
            new_points[idx] = new_point
            
            # Compute the new covering radius
            new_radius = _covering_radius(new_points)
            
            if new_radius < best_radius:
                best_radius = new_radius
                best_points = new_points.copy()
                points = new_points.copy()
                # Reduce step size slightly
                step_size *= 0.95
    
    return best_points, best_radius


def _fibonacci_spiral(n):
    """Generate n points on the unit sphere using the Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def construct():
    """Construct 50 points on the unit sphere to minimize the covering radius."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin
    
    # Start with Fibonacci spiral
    points = _fibonacci_spiral(N)
    best_points = points.copy()
    best_radius = _covering_radius(points)
    
    # Local optimization
    points, radius = _local_optimize(points, max_iters=500, step_size=0.05)
    if radius < best_radius:
        best_radius = radius
        best_points = points.copy()
    
    # Check time
    if time.time() > deadline:
        return best_points
    
    # Try multiple restarts with different initializations
    for restart in range(5):
        if time.time() > deadline:
            break
        
        # Random initialization
        np.random.seed(restart * 12345 + 678)
        points = np.random.randn(N, 3)
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        
        # Local optimization
        points, radius = _local_optimize(points, max_iters=300, step_size=0.05)
        if radius < best_radius:
            best_radius = radius
            best_points = points.copy()
    
    return best_points
# EVOLVE-BLOCK-END
