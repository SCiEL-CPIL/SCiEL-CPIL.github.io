import numpy as np
import time

N = 130

# EVOLVE-BLOCK-START

def _covering_radius_fast(points):
    """
    Fast vectorized covering radius calculation.
    Uses the property that the covering radius is the maximum over the vertices
    of the Voronoi cells, which are the points equidistant to at least 3 nearest neighbors.
    However, a simpler and robust approximation for local optimization is to
    compute the maximum distance to the nearest neighbor for a dense set of test points,
    or more efficiently, use the circumcenters of the Delaunay triangulation.
    
    For speed in an optimization loop, we use a vectorized check on a subset of
    candidate worst-case points. A common fast proxy is the maximum of the circumradii
    of the triangles formed by the 3 nearest neighbors for each point.
    
    Actually, the most robust fast method for local search is to evaluate the
    covering radius by sampling a fine grid or using the property that the
    worst point is either a vertex of the convex hull or a point equidistant
    to 3 nearest neighbors.
    
    For this implementation, we use a vectorized approach that computes the
    distance from every point to its 3 nearest neighbors and takes the maximum
    circumradius of those triangles. This is a lower bound on the true covering
    radius but is a very tight proxy for local optimization.
    
    Wait, the circumradius of the triangle formed by the 3 nearest neighbors of a point
    is NOT necessarily the covering radius. The covering radius is the maximum
    circumradius of the Delaunay triangles.
    
    Let's use a different fast proxy:
    1. Compute pairwise distances.
    2. For each point, find the 3 nearest neighbors.
    3. Compute the circumradius of the triangle formed by the point and its 2 nearest neighbors? No.
    
    The covering radius is the maximum distance from any point on the sphere to the nearest
    of the N points. This is equivalent to the maximum circumradius of the facets of the
    convex hull (if the points are in general position and the hull is well-defined).
    
    A fast and accurate enough proxy for local optimization is:
    For each point i, consider the triangle formed by i and its two nearest neighbors j, k.
    The circumcenter of this triangle (projected onto the sphere) is a candidate for the
    worst-case point. The distance from this circumcenter to i, j, k is the circumradius.
    The maximum of these circumradii is a good lower bound.
    
    However, a better fast proxy is to use the fact that the covering radius is the
    maximum over the vertices of the Voronoi diagram. The vertices of the Voronoi diagram
    on the sphere are the circumcenters of the Delaunay triangles.
    
    Computing the Delaunay triangulation on the sphere is expensive.
    
    Alternative fast proxy:
    Use a dense set of test points (e.g., 10000 points) on the sphere and compute the
    minimum distance from each test point to the N points. The maximum of these is an
    upper bound on the covering radius. This is very fast with vectorized operations.
    
    Let's use this:
    1. Generate a fixed set of M test points on the sphere (e.g., M=10000).
    2. Compute the distance from each test point to each of the N points.
    3. For each test point, find the minimum distance.
    4. The maximum of these minimum distances is the estimated covering radius.
    
    This is O(M*N) which is 10000*130 = 1.3M operations, very fast in NumPy.
    """
    # Generate test points once (static)
    if not hasattr(_covering_radius_fast, 'test_points'):
        # Use a Fibonacci spiral for test points too, with M points
        M = 20000
        i = np.arange(M) + 0.5
        z = 1.0 - 2.0 * i / M
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        _covering_radius_fast.test_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    test_points = _covering_radius_fast.test_points
    # Compute distances: (M, N)
    # Use dot product: cos(theta) = dot(p, q)
    # theta = arccos(dot(p, q))
    # To avoid arccos, we can maximize 1 - dot(p, q) as a proxy, but the score is in degrees.
    # The evaluator uses angular distance. We need to be consistent.
    # Let's compute the actual angular distance.
    dots = np.dot(test_points, points.T)  # (M, N)
    dots = np.clip(dots, -1.0, 1.0)
    min_dots = np.max(dots, axis=1)  # Max dot product = min angular distance
    # Angular distance = arccos(min_dots)
    # We want the maximum angular distance.
    # max_arccos = arccos(min(min_dots))
    # But we can just return min(min_dots) as a proxy for optimization (minimize max distance = maximize min dot product)
    # However, the score is in degrees, so we need to be careful.
    # Let's return the maximum angular distance in radians for internal use, and convert to degrees for scoring.
    min_dot = np.min(min_dots)
    # If min_dot is close to 1, the distance is small.
    # The covering radius is arccos(min_dot).
    return np.arccos(min_dot)

def _fibonacci_start(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    points = _fibonacci_start(N)
    best_points = points.copy()
    best_radius = _covering_radius_fast(best_points)
    
    # Local search: coordinate perturbation
    # For each point, perturb one coordinate at a time, normalize, and check if the covering radius improves.
    # Use a decreasing step size.
    
    step_size = 0.1
    min_step_size = 1e-6
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline:
        improved = False
        for i in range(N):
            for j in range(3):
                # Perturb coordinate j
                for sign in [1, -1]:
                    new_points = points.copy()
                    new_points[i, j] += sign * step_size
                    # Normalize
                    norm = np.linalg.norm(new_points[i])
                    if norm > 1e-10:
                        new_points[i] /= norm
                    
                    new_radius = _covering_radius_fast(new_points)
                    if new_radius < best_radius:
                        points = new_points
                        best_points = points.copy()
                        best_radius = new_radius
                        improved = True
                        no_improve_count = 0
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break
            # Reduce step size
            step_size *= 0.9
            if step_size < min_step_size:
                break
        
        # Check time
        if time.time() > deadline:
            break
    
    return best_points

# EVOLVE-BLOCK-END
