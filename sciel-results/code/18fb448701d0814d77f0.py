"""Covering the sphere with 50 equal caps: local optimization of the covering radius."""
import numpy as np
import time
import itertools

N = 50

# Deadline guard: finish well before 150s
DEADLINE = 140.0  # seconds


def _covering_radius(points):
    """
    Compute the covering radius (in radians) of a set of points on the unit sphere.
    The covering radius is the maximum angular distance from any point on the sphere
    to its nearest chosen point.
    
    For points on the sphere, the covering radius equals the maximum circumradius
    of the facets of the convex hull (for facets whose circumcenter is on the sphere
    and the cap is empty). A practical computation: the covering radius is the max
    over all points y on the sphere of min_i angle(y, p_i).
    
    An efficient way: the covering radius is determined by the vertices of the
    Voronoi cells on the sphere. The worst point is either:
    1. A point equidistant to 3 or more chosen points (a vertex of the spherical
       Voronoi diagram), or
    2. A point equidistant to 2 chosen points with the great-circle arc between them
       being a facet edge (but this is captured by the 3-point case in general).
    
    For N=50, we can check all triples of points to find the spherical Voronoi vertices.
    However, a simpler and robust approach: the covering radius is the maximum, over
    all points, of the distance to the nearest neighbor. The critical points are the
    circumcenters of the triangular facets of the convex hull.
    
    Alternative: Use the fact that the covering radius equals the max over all pairs
    of the angular distance to the midpoint, if the midpoint is not closer to any other
    point. But the most reliable method for small N is to check all triples.
    
    For efficiency with N=50, checking all C(50,3)=19600 triples is feasible.
    For each triple, compute the circumcenter of the spherical triangle (if it exists
    and is within the triangle), and check if it's a valid Voronoi vertex (i.e., no
    other point is closer). The covering radius is the max distance from such a vertex
    to its nearest point.
    
    Actually, a simpler characterization: the covering radius is the smallest r such that
    caps of angular radius r around the points cover the sphere. This is equivalent to
    the max over all points y of min_i d(y, p_i).
    
    The maximum is attained at a point that is equidistant to at least 3 points (a
    vertex of the Delaunay/Voronoi structure) or at a point equidistant to 2 points
    on the boundary of a Voronoi cell.
    
    For practical purposes with N=50, I'll use the following approach:
    - Compute all pairwise distances.
    - For each triple (i, j, k), find the point on the sphere equidistant to all three
      (the circumcenter of the spherical triangle). If this point is "valid" (i.e., it's
      the farthest from the nearest of the three, and no other point is closer), then
      the distance to the nearest of the three is a candidate for the covering radius.
    - Also consider midpoints of pairs as candidates (in case the worst point is on an
      edge of a Voronoi cell).
    
    Actually, the cleanest method: the covering radius is the max over all points y on
    the sphere of f(y) = min_i d(y, p_i). The maxima of f occur at points where f is
    not differentiable, i.e., where at least two (typically three) distances are equal
    and minimal. So we check all triples.
    
    For a triple (i,j,k), the point y equidistant to p_i, p_j, p_k on the sphere is the
    circumcenter of the spherical triangle. We can compute it as follows:
    - Solve for y such that y·p_i = y·p_j = y·p_k = c (same cosine of distance).
    - This means y is in the null space of [p_i - p_j; p_j - p_k] (approximately), 
      or more precisely, y is parallel to the cross product (p_i × p_j) × (p_j × p_k) 
      or similar. Actually, y must satisfy (p_i - p_j)·y = 0 and (p_j - p_k)·y = 0,
      so y is parallel to (p_i - p_j) × (p_j - p_k).
    
    Let's compute:
    v1 = p_i - p_j
    v2 = p_j - p_k  
    y = v1 × v2
    Normalize y. Then check d(y, p_i) = d(y, p_j) = d(y, p_k).
    The distance is arccos(y·p_i).
    
    But we also need to verify that this y is a valid Voronoi vertex, i.e., that for
    all other points m, d(y, p_m) >= d(y, p_i). If not, then the true nearest point
    is some p_m, and y is not a vertex of the Voronoi cell for {i,j,k}.
    
    For the covering radius, we take the max over all valid Voronoi vertices of
    d(y, p_nearest).
    
    Additionally, the covering radius could be determined by a point on the "edge"
    of a Voronoi cell (equidistant to exactly 2 points). But in generic position,
    the max is at a vertex (3+ points). For safety, I'll also check midpoints of
    pairs, but this is less critical.
    
    For N=50, checking all 19600 triples and for each checking 47 other points is
    19600 * 47 ≈ 921,200 operations, which is very fast.
    """
    points = np.array(points)
    n = len(points)
    max_dist = 0.0
    
    # Precompute pairwise dot products
    # dot[i,j] = points[i] · points[j]
    dot = points @ points.T
    
    # Check all triples
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                # Compute circumcenter of spherical triangle (i, j, k)
                pi = points[i]
                pj = points[j]
                pk = points[k]
                
                # y is parallel to (pi - pj) × (pj - pk)
                v1 = pi - pj
                v2 = pj - pk
                y = np.cross(v1, v2)
                y_norm = np.linalg.norm(y)
                if y_norm < 1e-12:
                    continue
                y = y / y_norm
                
                # Distance to each of the three points
                d_i = np.arccos(np.clip(np.dot(y, pi), -1.0, 1.0))
                d_j = np.arccos(np.clip(np.dot(y, pj), -1.0, 1.0))
                d_k = np.arccos(np.clip(np.dot(y, pk), -1.0, 1.0))
                
                # The distance should be approximately equal; use the max as the
                # distance to the nearest of the three (they should be equal)
                d_min_three = min(d_i, d_j, d_k)
                
                # Check if any other point is closer
                is_valid = True
                for m in range(n):
                    if m == i or m == j or m == k:
                        continue
                    d_m = np.arccos(np.clip(np.dot(y, points[m]), -1.0, 1.0))
                    if d_m < d_min_three - 1e-10:
                        is_valid = False
                        break
                
                if is_valid:
                    if d_min_three > max_dist:
                        max_dist = d_min_three
    
    # Also check midpoints of pairs (for robustness)
    for i in range(n):
        for j in range(i+1, n):
            mid = points[i] + points[j]
            mid_norm = np.linalg.norm(mid)
            if mid_norm < 1e-12:
                continue
            mid = mid / mid_norm
            
            d_i = np.arccos(np.clip(np.dot(mid, points[i]), -1.0, 1.0))
            d_j = np.arccos(np.clip(np.dot(mid, points[j]), -1.0, 1.0))
            d_min_pair = min(d_i, d_j)
            
            is_valid = True
            for m in range(n):
                if m == i or m == j:
                    continue
                d_m = np.arccos(np.clip(np.dot(mid, points[m]), -1.0, 1.0))
                if d_m < d_min_pair - 1e-10:
                    is_valid = False
                    break
            
            if is_valid:
                if d_min_pair > max_dist:
                    max_dist = d_min_pair
    
    return max_dist


def _covering_radius_fast(points):
    """
    Vectorized version of covering radius computation.
    """
    points = np.array(points)
    n = len(points)
    
    # Compute all pairwise dot products
    dot = points @ points.T
    
    max_dist = 0.0
    
    # Check all triples
    for i in range(n):
        for j in range(i+1, n):
            pi = points[i]
            pj = points[j]
            v1 = pi - pj
            
            for k in range(j+1, n):
                pk = points[k]
                v2 = pj - pk
                y = np.cross(v1, v2)
                y_norm = np.linalg.norm(y)
                if y_norm < 1e-12:
                    continue
                y = y / y_norm
                
                # Distance to the three points
                d_i = np.arccos(np.clip(y[0]*pi[0] + y[1]*pi[1] + y[2]*pi[2], -1.0, 1.0))
                d_j = np.arccos(np.clip(y[0]*pj[0] + y[1]*pj[1] + y[2]*pj[2], -1.0, 1.0))
                d_k = np.arccos(np.clip(y[0]*pk[0] + y[1]*pk[1] + y[2]*pk[2], -1.0, 1.0))
                d_min_three = min(d_i, d_j, d_k)
                
                # Check all other points
                # Compute distances from y to all points
                d_all = np.arccos(np.clip(points @ y, -1.0, 1.0))
                d_all[i] = np.inf
                d_all[j] = np.inf
                d_all[k] = np.inf
                min_other = np.min(d_all)
                
                if min_other >= d_min_three - 1e-10:
                    if d_min_three > max_dist:
                        max_dist = d_min_three
    
    return max_dist


def _gradient_descent(points, max_iter=500, lr=0.01, tol=1e-10):
    """
    Gradient descent on the sphere to minimize the covering radius.
    Uses a smooth approximation: f(p) = -log-sum-exp(-d_i / tau) where d_i are
    distances from a test point... Actually, this is complex.
    
    Alternative: Use the fact that the covering radius is determined by the worst
    Voronoi vertex. We can do a local optimization by moving points to reduce the
    max circumradius.
    
    Simpler approach: Use a repulsion-based energy that approximates the covering
    radius minimization. The covering radius is related to the max over pairs of
    the angular distance, but more precisely it's the max circumradius.
    
    Let's use a different approach: minimize the sum of exp(alpha * d_i) where d_i
    are the distances from the worst point... This is getting complicated.
    
    Instead, let's use a simple local search: for each point, try moving it in
    small random directions and keep the move if it reduces the covering radius.
    """
    points = np.array(points, dtype=float)
    n = len(points)
    best_points = points.copy()
    best_cr = _covering_radius_fast(best_points)
    
    for it in range(max_iter):
        if time.time() > DEADLINE - 10:
            break
            
        # Pick a random point to move
        idx = np.random.randint(n)
        original = points[idx].copy()
        
        # Generate a random direction
        direction = np.random.randn(3)
        direction = direction / np.linalg.norm(direction)
        
        # Try several step sizes
        improved = False
        for lr in [0.1, 0.05, 0.02, 0.01, 0.005, 0.002]:
            # Move point in direction, then project back to sphere
            new_point = points[idx] + lr * direction
            new_point = new_point / np.linalg.norm(new_point)
            
            points[idx] = new_point
            cr = _covering_radius_fast(points)
            
            if cr < best_cr - 1e-12:
                best_cr = cr
                best_points = points.copy()
                improved = True
                break
            else:
                points[idx] = original
        
        if not improved:
            # Try opposite direction
            direction = -direction
            for lr in [0.1, 0.05, 0.02, 0.01, 0.005, 0.002]:
                new_point = points[idx] + lr * direction
                new_point = new_point / np.linalg.norm(new_point)
                
                points[idx] = new_point
                cr = _covering_radius_fast(points)
                
                if cr < best_cr - 1e-12:
                    best_cr = cr
                    best_points = points.copy()
                    improved = True
                    break
                else:
                    points[idx] = original
        
        if not improved:
            points[idx] = original
    
    return best_points, best_cr


def _fibonacci_sphere(n):
    """Generate Fibonacci sphere points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _icosahedron_and_subdivisions(n_target=50):
    """
    Generate points by starting with an icosahedron and subdividing faces.
    An icosahedron has 12 vertices. Subdividing each face into 4 gives 48 vertices,
    which is close to 50.
    """
    # Icosahedron vertices
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Icosahedron faces
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]
    
    # Subdivide each face into 4 triangles by adding midpoints
    new_verts = [tuple(v) for v in verts]
    new_faces = []
    midpoint_cache = {}
    
    def get_midpoint(p1, p2):
        key = (min(p1, p2), max(p1, p2))
        if key in midpoint_cache:
            return midpoint_cache[key]
        mid = (np.array(p1) + np.array(p2)) / 2.0
        mid = mid / np.linalg.norm(mid)
        idx = len(new_verts)
        new_verts.append(tuple(mid))
        midpoint_cache[key] = idx
        return idx
    
    for face in faces:
        a, b, c = face
        ab = get_midpoint(a, b)
        bc = get_midpoint(b, c)
        ca = get_midpoint(c, a)
        new_faces.append([a, ab, ca])
        new_faces.append([b, bc, ab])
        new_faces.append([c, ca, bc])
        new_faces.append([ab, bc, ca])
    
    new_verts = np.array(new_verts)
    # Normalize
    new_verts = new_verts / np.linalg.norm(new_verts, axis=1, keepdims=True)
    
    # We have 12 + 30 = 42 vertices from subdivision.
    # We need 50, so we have 42. Let's add more by further subdividing or using a different approach.
    
    # Actually, icosahedron has 20 faces. Subdividing each into 4 gives 20*4=80 faces,
    # and the number of vertices is 12 + 30 = 42 (since each edge is shared by 2 faces,
    # and there are 30 edges). So we get 42 points.
    
    # For 50 points, we can start with 42 and add 8 more by further subdividing 8 faces.
    # Or we can just use the 42 points and add 8 random points, then optimize.
    
    # Let's just return the 42 points and handle the rest in the main function.
    return new_verts


def construct():
    np.random.seed(42)
    start_time = time.time()
    
    # Start with Fibonacci sphere
    points = _fibonacci_sphere(N)
    best_points = points.copy()
    best_cr = _covering_radius_fast(best_points)
    
    # Try local search with multiple restarts
    num_restarts = 5
    
    for restart in range(num_restarts):
        if time.time() > DEADLINE - 30:
            break
        
        # Perturb the initial points
        if restart == 0:
            current = points.copy()
        else:
            current = points + np.random.randn(N, 3) * 0.3
            current = current / np.linalg.norm(current, axis=1, keepdims=True)
        
        # Run local search
        current, cr = _gradient_descent(current, max_iter=300)
        
        if cr < best_cr:
            best_cr = cr
            best_points = current.copy()
    
    # Final polish with smaller steps
    if time.time() < DEADLINE - 15:
        best_points, best_cr = _gradient_descent(best_points, max_iter=200, max_iter=200)
    
    # Ensure points are normalized
    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points


# EVOLVE-BLOCK-START
# (The above code is the complete program; this block is for compatibility)
# EVOLVE-BLOCK-END
