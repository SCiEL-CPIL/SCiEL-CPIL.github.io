import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 50
START_TIME = time.time()
DEADLINE = 110.0  # Leave 40s margin before 150s kill

def to_sphere(params):
    """Convert raw params to unit vectors."""
    v = params.reshape(-1, 3)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return v / norms

def compute_covering_radius(points):
    """Compute the covering radius (in radians) for a set of points on the unit sphere.
    
    The covering radius is the maximum angular distance from any point on the sphere
    to the nearest point in the set. This equals the maximum circumradius of the
    spherical Voronoi cells, which corresponds to the largest circumcircle of
    facets of the convex hull (checked to have empty caps).
    
    For a set of points on the sphere, the covering radius is the max over all
    "facets" (triangles in the Delaunay triangulation of the spherical points)
    of the circumradius of that triangle, but only for triangles that have
    an empty circumcircle (no other points inside).
    
    A practical approach: use the convex hull of 3D points. Each face of the
    convex hull is a triangle (in 3D). The circumradius of each triangular face
    gives a candidate for the covering radius. The covering radius is the
    maximum of these circumradii for faces that are on the "outer" hull.
    
    Actually, the correct formulation: the covering radius of a set of points
    on the sphere is the minimum r such that the union of spherical caps of
    radius r around the points covers the sphere. This is equal to the
    maximum, over all points y on the sphere, of the minimum distance to any
    chosen point.
    
    A computationally efficient way: the covering radius equals the maximum
    circumradius of the spherical Delaunay triangulation facets. In 3D, the
    convex hull approach works: for each face of the convex hull of the 3D
    points, compute the circumradius (radius of the sphere through the three
    vertices, centered at the origin). The covering radius is the
    maximum of these circumradii.
    """
    points = np.asarray(points)
    # Try to build convex hull
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi  # worst case
    
    max_circum = 0.0
    for simplex in hull.simplices:
        tri = points[simplex]
        # For a triangle on the unit sphere, the circumradius (angular) is
        # related to the 3D circumradius. The circumcenter of the triangle
        # in 3D, projected to the sphere, gives the angular circumradius.
        
        a, b, c = tri[0], tri[1], tri[2]
        # Compute circumcenter of triangle in 3D
        ab = b - a
        ac = c - a
        abab = np.dot(ab, ab)
        acac = np.dot(ac, ac)
        abac = np.dot(ab, ac)
        
        denom = abab * acac - abac * abac
        if abs(denom) < 1e-12:
            # Degenerate triangle
            continue
        u1 = (acac * abac - abab * abac) / denom
        u2 = (abab * abac - acac * abac) / denom
        circumcenter = a + u1 * ab + u2 * ac
        circumradius_3d = np.linalg.norm(circumcenter - a)
        
        # The angular circumradius: the triangle is inscribed in a circle
        # of radius circumradius_3d. The angular distance from the "pole"
        # (circumcenter direction) to each vertex is arccos(dot(vertex, pole)).
        # The circumradius in angular terms: if circumcenter has norm R_cc,
        # and vertices are on unit sphere, the angular circumradius theta
        # satisfies: cos(theta) = 1 / (R_cc) ... hmm, let me think again.
        
        # Actually, for points on the unit sphere, the spherical circumradius
        # of a spherical triangle is the angular distance from the spherical
        # circumcenter to each vertex. The spherical circumcenter is the
        # intersection of the perpendicular bisectors on the sphere.
        
        # Alternative: compute directly. For each vertex, compute angle to
        # the circumcenter direction.
        if np.linalg.norm(circumcenter) < 1e-10:
            # Circumcenter at origin, degenerate
            continue
        pole = circumcenter / np.linalg.norm(circumcenter)
        # The angular distance from pole to each vertex should be the circumradius
        theta_a = np.arccos(np.clip(np.dot(pole, a), -1, 1))
        theta_b = np.arccos(np.clip(np.dot(pole, b), -1, 1))
        theta_c = np.arccos(np.clip(np.dot(pole, c), -1, 1))
        
        # The circumradius is the max of these (should be equal if circumcenter
        # is correct, but numerical issues may cause slight differences)
        circum_angular = max(theta_a, theta_b, theta_c)
        max_circum = max(max_circum, circum_angular)
    
    # Also check that no point is "inside" the hull (all points are vertices of
    # the convex hull or on the surface). For a set of points on the sphere,
    # the convex hull includes all points if none are inside the convex hull
    # of the others. The covering radius should also consider Voronoi regions
    # that may have 4+ facets if points are not in general position.
    
    # A more robust approach: also sample the sphere and compute the max
    # nearest-neighbor distance to catch any missed cases.
    # But this is expensive. Let's also check: for each point, the Voronoi
    # cell's farthest point is the circumcenter of the Delaunay triangle
    # containing that point's Voronoi region. The convex hull approach
    # approximates this.
    
    return max_circum

def local_optimize(initial_points, max_iter=100, tol=1e-8):
    """Use gradient-free optimization (Powell) to locally optimize the
    covering radius by perturbing points on the sphere."""
    params = initial_points.flatten()
    
    def obj(p):
        points = to_sphere(p)
        return compute_covering_radius(points)
    
    best_params = params.copy()
    best_val = obj(best_params)
    
    try:
        # Powell is often faster and more robust for this type of problem
        result = minimize(obj, params, method='Powell',
                          options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-8})
        if result.fun < best_val:
            best_params = result.x
            best_val = result.fun
    except Exception:
        pass
    
    return to_sphere(best_params), best_val

def random_init(n_points, rng):
    """Generate random points on the unit sphere."""
    # Use Gaussian and normalize
    v = rng.randn(n_points, 3)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / norms

def fibonacci_spiral(n_points):
    """Fibonacci spiral on the sphere."""
    i = np.arange(n_points) + 0.5
    z = 1.0 - 2.0 * i / n_points
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def icosahedron_based_init(n_points):
    """Generate points based on icosahedron vertices, then add more points
    by subdividing and projecting."""
    # 12 vertices of icosahedron
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    if n_points <= 12:
        return verts[:n_points]
    
    # Start with icosahedron and add points by finding worst-covered areas
    points = verts.copy()
    
    rng = np.random.RandomState(123)
    while len(points) < n_points:
        # Find the point on sphere that is farthest from existing points
        # Use a grid-based approach
        best_dist = 0
        best_point = None
        for _ in range(200):
            candidate = rng.randn(3)
            candidate = candidate / np.linalg.norm(candidate)
            dists = np.arccos(np.clip(points @ candidate, -1, 1))
            min_dist = np.min(dists)
            if min_dist > best_dist:
                best_dist = min_dist
                best_point = candidate
        if best_point is not None:
            points = np.vstack([points, best_point])
    
    return points[:n_points]

def construct():
    """Construct 50 points on the unit sphere minimizing the covering radius."""
    rng = np.random.RandomState(42)
    
    # Try multiple initialization strategies
    inits = []
    
    # 1. Fibonacci spiral
    inits.append(fibonacci_spiral(N))
    
    # 2. Icosahedron-based
    inits.append(icosahedron_based_init(N))
    
    # 3. Random
    inits.append(random_init(N, rng))
    
    # 4. Slightly perturbed Fibonacci
    noise = 0.05 * rng.randn(N, 3)
    fib_perturbed = fibonacci_spiral(N) + noise
    fib_perturbed = fib_perturbed / np.linalg.norm(fib_perturbed, axis=1, keepdims=True)
    inits.append(fib_perturbed)
    
    # Initialize with the first valid init to ensure best_points is never None
    best_points = inits[0].copy()
    best_cr = compute_covering_radius(best_points)
    
    for init in inits:
        if time.time() > DEADLINE - 20:
            break
        
        # Quick local optimization
        try:
            optimized, cr = local_optimize(init, max_iter=100)
            if cr < best_cr:
                best_cr = cr
                best_points = optimized
        except Exception:
            continue
    
    # Final fine-tuning with more iterations if time allows
    if time.time() < DEADLINE - 10:
        try:
            optimized, cr = local_optimize(best_points, max_iter=200)
            if cr < best_cr:
                best_cr = cr
                best_points = optimized
        except Exception:
            pass
    
    # Ensure all points are on the unit sphere
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    # Avoid division by zero if a vector became zero (shouldn't happen with to_sphere)
    norms = np.where(norms < 1e-10, 1.0, norms)
    best_points = best_points / norms
    
    # Validate: check no coincident points
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(best_points[i] - best_points[j]) < 1e-6:
                # Separate them slightly
                best_points[j] = best_points[j] * 0.999 + 0.001 * rng.randn(3)
                best_points[j] /= np.linalg.norm(best_points[j])
    
    # Check not all in a closed hemisphere: the origin should be in the
    # interior of the convex hull
    try:
        hull = ConvexHull(best_points)
    except Exception:
        pass
    
    return best_points
