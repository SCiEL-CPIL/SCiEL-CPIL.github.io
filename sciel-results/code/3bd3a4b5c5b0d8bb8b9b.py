"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 50
SEED = 42
DEADLINE_MARGIN = 10.0  # seconds to leave before the 150s kill

# EVOLVE-BLOCK-START
def _covering_radius(points):
    """
    Compute the covering radius of points on the unit sphere.
    The covering radius is the largest angular distance from any point on the
    sphere to its nearest chosen point. This equals the largest circumradius
    of the facets of the convex hull (in degrees).
    """
    points = np.array(points)
    n = len(points)
    if n < 4:
        # Degenerate case
        return 90.0
    
    # Ensure points are on the unit sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    try:
        hull = ConvexHull(points)
    except Exception:
        # If hull fails, fall back to a direct computation
        return _covering_radius_direct(points)
    
    # For each facet, compute the circumradius
    # The circumradius of a spherical triangle (facet) is the angular distance
    # from the center of the circumscribed circle to the vertices.
    # For a planar facet with vertices v1, v2, v3, the circumcenter in 3D
    # is along the normal direction. The angular circumradius is the angle
    # between the facet center (normalized) and any vertex.
    
    max_r = 0.0
    for simplex in hull.simplices:
        # simplex has 3 vertices for a 3D hull
        tri = points[simplex]
        # Compute the centroid of the triangle (in 3D, not on the sphere)
        # Actually, we need the circumcenter of the spherical triangle.
        # The circumcenter of a spherical triangle is the point on the sphere
        # equidistant from all three vertices.
        
        # Method: The circumcenter lies on the line through the origin and
        # the circumcenter of the planar triangle.
        # Compute the planar circumcenter of the triangle.
        a, b, c = tri[0], tri[1], tri[2]
        
        # Compute the circumcenter of the planar triangle
        # Using the formula: circumcenter = (a * |b-c|^2 + b * |c-a|^2 + c * |a-b|^2) / (2 * cross product area)
        # Actually, let's use a more robust method.
        
        # Compute the normal to the plane
        ab = b - a
        ac = c - a
        nrm = np.cross(ab, ac)
        nrm_norm = np.linalg.norm(nrm)
        if nrm_norm < 1e-12:
            continue
        nrm = nrm / nrm_norm
        
        # The circumcenter in the plane satisfies:
        # |circ - a|^2 = |circ - b|^2 = |circ - c|^2
        # This gives a linear system.
        
        # Let circ = a + t1 * ab + t2 * ac
        # |a + t1*ab + t2*ac - a|^2 = |a + t1*ab + t2*ac - b|^2
        # |t1*ab + t2*ac|^2 = |t1*ab + t2*ac - ab|^2
        # 0 = -2*t1*|ab|^2 - 2*t2*(ab·ac) + |ab|^2
        # t1*|ab|^2 + t2*(ab·ac) = |ab|^2/2
        
        # Similarly for c:
        # t1*(ab·ac) + t2*|ac|^2 = |ac|^2/2
        
        A = np.array([[np.dot(ab, ab), np.dot(ab, ac)],
                       [np.dot(ab, ac), np.dot(ac, ac)]])
        rhs = np.array([0.5 * np.dot(ab, ab), 0.5 * np.dot(ac, ac)])
        
        try:
            t = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            continue
        
        circ_planar = a + t[0] * ab + t[1] * ac
        
        # The spherical circumcenter is circ_planar normalized
        circ_norm = np.linalg.norm(circ_planar)
        if circ_norm < 1e-12:
            continue
        circ_sphere = circ_planar / circ_norm
        
        # The angular circumradius is the angle between circ_sphere and any vertex
        cos_r = np.clip(np.dot(circ_sphere, a), -1.0, 1.0)
        r = np.arccos(cos_r)
        max_r = max(max_r, r)
    
    return np.degrees(max_r)


def _covering_radius_direct(points, n_samples=10000):
    """Direct computation by sampling the sphere."""
    points = np.array(points)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    # Sample points on the sphere
    rng = np.random.RandomState(0)
    samples = rng.randn(n_samples, 3)
    samples /= np.linalg.norm(samples, axis=1, keepdims=True)
    
    # Compute distances
    # Use matrix multiplication for speed
    # cos_dist = samples @ points.T
    cos_dist = np.clip(samples @ points.T, -1.0, 1.0)
    min_dist = np.arccos(cos_dist.max(axis=1))
    
    return np.degrees(min_dist.max())


def _lloyd_iteration(points, n_iters=100, tol=1e-10):
    """
    Perform Lloyd's algorithm on the sphere.
    Each point is moved to the centroid of its Voronoi cell (spherical).
    """
    points = np.array(points, dtype=np.float64)
    n = len(points)
    
    for it in range(n_iters):
        # Compute Voronoi cells using the convex hull
        # Actually, for spherical Voronoi, we can use the fact that the
        # centroid of the Voronoi cell of point i is the normalized sum of
        # the vertices of the dual cell.
        
        # Simpler approach: Use the fact that the spherical Voronoi cell of
        # point i is the set of points closer to i than to any other point.
        # The centroid can be approximated by integrating over the cell.
        
        # For efficiency, let's use a different approach:
        # Move each point toward the centroid of its nearby points.
        
        new_points = np.zeros_like(points)
        for i in range(n):
            # Find the Voronoi cell of point i
            # The cell is bounded by great circle arcs.
            # Approximate by finding points on the sphere closer to i.
            
            # Use the convex hull of all points to find the cell.
            # The dual of the convex hull gives the Voronoi diagram.
            
            # For simplicity, use a gradient-based approach:
            # The gradient of the sum of squared distances to the cell boundary
            # points gives the direction to move.
            
            # Actually, let's use a simpler heuristic:
            # Move point i toward the centroid of points j where i is the
            # nearest neighbor of the midpoint of i and j.
            
            midpoints = []
            for j in range(n):
                if i == j:
                    continue
                mid = (points[i] + points[j]) / 2.0
                mid_norm = np.linalg.norm(mid)
                if mid_norm > 1e-12:
                    mid = mid / mid_norm
                    # Check if i is the nearest to mid
                    dists = np.arccos(np.clip(points @ mid, -1.0, 1.0))
                    if np.argmin(dists) == i:
                        midpoints.append(mid)
            
            if midpoints:
                centroid = np.sum(midpoints, axis=0)
                centroid_norm = np.linalg.norm(centroid)
                if centroid_norm > 1e-12:
                    new_points[i] = centroid / centroid_norm
                else:
                    new_points[i] = points[i]
            else:
                new_points[i] = points[i]
        
        # Check convergence
        diff = np.linalg.norm(new_points - points)
        points = new_points
        
        if diff < tol:
            break
    
    return points


def _fibonacci_sphere(n, offset=0.0):
    """Generate n points on the sphere using the Fibonacci spiral."""
    i = np.arange(n) + 0.5 + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, seed=0):
    """Generate n random points on the sphere."""
    rng = np.random.RandomState(seed)
    points = rng.randn(n, 3)
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    return points


def _icosahedron():
    """Return the 12 vertices of a regular icosahedron."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=np.float64)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    return verts


def _optimize_covering(points, max_time=60.0):
    """
    Optimize the covering radius using gradient-based methods.
    We minimize the maximum angular distance, which is a nonsmooth objective.
    We use a smooth approximation via log-sum-exp.
    """
    points = np.array(points, dtype=np.float64)
    n = len(points)
    start_time = time.time()
    
    def objective_and_grad(x):
        """
        x: 3n vector of point coordinates.
        Objective: log-sum-exp of angular distances (smooth approximation to max).
        """
        pts = x.reshape(n, 3)
        # Normalize
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        
        # Compute the covering radius using the convex hull
        # For the smooth approximation, we need the angular distances from
        # each facet center to its vertices.
        
        try:
            hull = ConvexHull(pts)
        except Exception:
            # Fall back to a direct computation
            rng = np.random.RandomState(0)
            samples = rng.randn(5000, 3)
            samples /= np.linalg.norm(samples, axis=1, keepdims=True)
            cos_dist = np.clip(samples @ pts.T, -1.0, 1.0)
            min_dist = np.arccos(cos_dist.max(axis=1))
            return min_dist.max(), np.zeros_like(x)
        
        # Compute the circumradius for each facet
        # For a smooth approximation, we use the log-sum-exp of the circumradii.
        
        circumradii = []
        grads = np.zeros_like(x)
        
        for simplex in hull.simplices:
            tri = pts[simplex]
            a, b, c = tri[0], tri[1], tri[2]
            
            ab = b - a
            ac = c - a
            nrm = np.cross(ab, ac)
            nrm_norm = np.linalg.norm(nrm)
            if nrm_norm < 1e-12:
                continue
            nrm = nrm / nrm_norm
            
            A = np.array([[np.dot(ab, ab), np.dot(ab, ac)],
                           [np.dot(ab, ac), np.dot(ac, ac)]])
            rhs = np.array([0.5 * np.dot(ab, ab), 0.5 * np.dot(ac, ac)])
            
            try:
                t = np.linalg.solve(A, rhs)
            except np.linalg.LinAlgError:
                continue
            
            circ_planar = a + t[0] * ab + t[1] * ac
            circ_norm = np.linalg.norm(circ_planar)
            if circ_norm < 1e-12:
                continue
            circ_sphere = circ_planar / circ_norm
            
            cos_r = np.clip(np.dot(circ_sphere, a), -1.0, 1.0)
            r = np.arccos(cos_r)
            circumradii.append(r)
        
        if not circumradii:
            return 90.0, np.zeros_like(x)
        
        circumradii = np.array(circumradii)
        # Log-sum-exp for smooth max
        beta = 100.0  # Temperature parameter
        max_r = np.max(circumradii)
        log_sum_exp = max_r + np.log(np.sum(np.exp(beta * (circumradii - max_r)))) / beta
        
        # Gradient: approximate numerically
        eps = 1e-5
        grad = np.zeros_like(x)
        for i in range(n):
            for j in range(3):
                x_plus = x.copy()
                x_plus[i * 3 + j] += eps
                f_plus = objective_and_grad(x_plus)[0]
                
                x_minus = x.copy()
                x_minus[i * 3 + j] -= eps
                f_minus = objective_and_grad(x_minus)[0]
                
                grad[i * 3 + j] = (f_plus - f_minus) / (2 * eps)
        
        return log_sum_exp, grad
    
    # Use L-BFGS-B with projection onto the sphere
    def project_sphere(x):
        pts = x.reshape(n, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        return pts.reshape(-1)
    
    # Initial objective
    f0, g0 = objective_and_grad(points.reshape(-1))
    
    best_x = points.copy()
    best_f = f0
    
    # Use scipy's L-BFGS-B
    from scipy.optimize import minimize as sp_minimize
    
    result = sp_minimize(
        lambda x: objective_and_grad(x)[0],
        points.reshape(-1),
        jac=lambda x: objective_and_grad(x)[1],
        method='L-BFGS-B',
        options={'maxiter': 100, 'ftol': 1e-10}
    )
    
    if result.x is not None:
        best_x = project_sphere(result.x)
        best_f = objective_and_grad(best_x.reshape(-1))[0]
    
    return best_x.reshape(n, 3)


def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN
    
    best_points = None
    best_score = np.inf
    
    # Try multiple initializations
    initializations = []
    
    # Fibonacci spiral
    initializations.append(_fibonacci_sphere(N))
    
    # Fibonacci with offset
    initializations.append(_fibonacci_sphere(N, offset=0.1))
    
    # Random
    for seed in range(5):
        initializations.append(_random_sphere(N, seed=seed))
    
    # Icosahedron-based (expand to 50 points)
    ico = _icosahedron()
    # Add midpoints of edges
    ico_points = [ico]
    for i in range(12):
        for j in range(i + 1, 12):
            mid = (ico[i] + ico[j]) / 2.0
            mid_norm = np.linalg.norm(mid)
            if mid_norm > 1e-12:
                mid = mid / mid_norm
                ico_points.append(mid)
    
    # Take the first 50
    ico_points = np.array(ico_points[:50])
    initializations.append(ico_points)
    
    for init in initializations:
        if time.time() > deadline:
            break
        
        # Normalize
        init = init / np.linalg.norm(init, axis=1, keepdims=True)
        
        # Run Lloyd's algorithm
        lloyd_points = _lloyd_iteration(init, n_iters=50, tol=1e-10)
        
        # Compute score
        score = _covering_radius(lloyd_points)
        
        if score < best_score:
            best_score = score
            best_points = lloyd_points
        
        # If we have time, try further optimization
        if time.time() < deadline - 30.0:
            opt_points = _optimize_covering(lloyd_points, max_time=20.0)
            opt_score = _covering_radius(opt_points)
            
            if opt_score < best_score:
                best_score = opt_score
                best_points = opt_points
    
    # Final polish: run Lloyd's on the best points
    if best_points is not None and time.time() < deadline:
        final_points = _lloyd_iteration(best_points, n_iters=100, tol=1e-12)
        final_score = _covering_radius(final_points)
        
        if final_score < best_score:
            best_score = final_score
            best_points = final_points
    
    if best_points is None:
        # Fallback to Fibonacci
        best_points = _fibonacci_sphere(N)
    
    # Ensure points are on the unit sphere
    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points
# EVOLVE-BLOCK-END
