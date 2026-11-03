"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50
import time

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay


def _normalize(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    return points / norms


def _covering_radius(points):
    """
    Compute the covering radius in degrees for a set of unit points.
    The covering radius is the maximum over the sphere of the distance to the nearest point.
    This is equivalent to the maximum circumradius of the spherical Voronoi cells.
    A practical way to compute this is to find the convex hull of the points in R^3.
    The facets of the convex hull correspond to the "empty caps" of the Voronoi diagram.
    The circumradius of a facet is the distance from the origin to the plane containing the facet.
    However, the covering radius is the angular distance, which is related to the distance to the plane.
    Specifically, for a facet with normal vector n (unit) and distance d from the origin,
    the farthest point in the cap is at angular distance arccos(d).
    Wait, that's not quite right. The covering radius is the maximum distance from any point on the sphere
    to the nearest of the given points.
    The Voronoi cell of a point p_i is the set of points on the sphere closer to p_i than to any other p_j.
    The farthest point in this cell is the circumcenter of the spherical polygon formed by the Voronoi vertices.
    The Voronoi vertices on the sphere are the points equidistant to 3 (or more) of the given points.
    These correspond to the circumcenters of the triangular facets of the convex hull of the points.
    The distance from the origin to the circumcenter of a facet (in R^3) is 1 / sin(omega) where omega is the angular covering radius for that facet? No.
    Let's re-derive.
    Let the points on the sphere be p_1, ..., p_N.
    The convex hull in R^3 has facets. Each facet is defined by 3 points p_i, p_j, p_k.
    The plane containing this facet has a normal vector n. The distance from the origin to this plane is h.
    The circumcenter of the triangle p_i p_j p_k in R^3 is c = n * h / (n . n) ? No, c is the projection of the origin onto the plane? No.
    The circumcenter of a triangle on the sphere (in the sense of the point on the sphere equidistant to the 3 vertices) is the point along the normal to the facet's plane (in R^3) that intersects the sphere.
    Let the plane equation be x . n = h, where n is the unit normal.
    The circumcenter C on the sphere is C = n * h / h = n? No.
    The point on the sphere closest to the plane is at distance h from the origin? No, the distance from origin to plane is |h|.
    The circumcenter of the spherical triangle is the point on the sphere in the direction of the normal to the facet (if the facet is "facing out").
    Actually, the covering radius is the maximum over all facets of the angular distance from the facet's circumcenter to any of its vertices? No, the circumcenter IS the farthest point in the Voronoi cell.
    The angular distance from the circumcenter C to a vertex p_i is the covering radius for that cell.
    C is the point on the sphere in the direction of the normal to the facet.
    Let n be the unit normal to the facet (pointing outward).
    The distance from origin to the plane is h = n . p_i.
    The circumcenter C is at position n (if h>0) or -n (if h<0). Since points are on the unit sphere and we are looking at the convex hull, the origin is inside, so h > 0.
    So C = n.
    The angular distance from C to p_i is arccos(C . p_i) = arccos(h).
    Wait, C . p_i = n . p_i = h.
    So the covering radius is max over all facets of arccos(h).
    We want to minimize this, so we want to maximize the minimum h over all facets.
    So the objective is to maximize min_facet (n . p_i) for all facets.
    This is equivalent to minimizing max_facet (arccos(n . p_i)).
    
    Let's implement this.
    1. Compute ConvexHull of points.
    2. For each facet, get the 3 points.
    3. Compute the normal n = cross(p_i - p_k, p_j - p_k). Normalize n.
    4. Ensure n points outward (away from origin). Since origin is inside, n . p_i should be positive. If not, flip n.
    5. h = n . p_i.
    6. The angular covering radius for this facet is arccos(h).
    7. The total covering radius is max(arccos(h)) over all facets.
    8. We want to minimize this.
    """
    if len(points) < 3:
        return np.pi
    try:
        hull = ConvexHull(points)
    except:
        return np.pi
    
    max_angle = 0.0
    for simplex in hull.simplices:
        p1 = points[simplex[0]]
        p2 = points[simplex[1]]
        p3 = points[simplex[2]]
        
        # Normal to the plane
        v1 = p2 - p1
        v2 = p3 - p1
        n = np.cross(v1, v2)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-10:
            continue
        n = n / norm_n
        
        # Ensure n points away from origin
        if np.dot(n, p1) < 0:
            n = -n
        
        h = np.dot(n, p1)
        h = np.clip(h, -1.0, 1.0)
        angle = np.arccos(h)
        if angle > max_angle:
            max_angle = angle
            
    return max_angle * 180.0 / np.pi


def _objective(params):
    """Objective function: covering radius in degrees."""
    pts = params.reshape(N, 3)
    pts = _normalize(pts)
    return _covering_radius(pts)


def _spherical_lloyd_iteration(points, max_iter=200, time_limit=60, damping=0.5):
    """
    Spherical Lloyd's algorithm (Voronoi relaxation).
    Points are moved toward the centroids of their spherical Voronoi cells.
    """
    start_time = time.time()
    points = _normalize(points).copy()
    current_cost = _covering_radius(points)
    best_points = points.copy()
    best_cost = current_cost
    
    for i in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        try:
            hull = ConvexHull(points)
        except:
            break
        
        # For each point, find the Voronoi cell centroid.
        # The Voronoi cell of point i on the sphere is bounded by the great circle arcs
        # that are the perpendicular bisectors of the edges of the convex hull incident to i.
        # The centroid of the Voronoi cell can be approximated by the average of the
        # circumcenters of the facets incident to point i, weighted by facet area,
        # or more simply, by the normalized sum of the facet normals of incident facets.
        
        # Actually, a simpler approach: for each point, compute the average of the
        # outward normals of all hull facets that contain that point. This gives
        # an approximation of the Voronoi cell center direction.
        
        new_points = points.copy()
        
        # Build a mapping from point index to list of facet indices
        point_to_facets = {i: [] for i in range(N)}
        for fi, simplex in enumerate(hull.simplices):
            for pi in simplex:
                point_to_facets[pi].append(fi)
        
        for pi in range(N):
            facets = point_to_facets[pi]
            if not facets:
                continue
            
            # Compute the average of facet normals for this point
            normal_sum = np.zeros(3)
            for fi in facets:
                simplex = hull.simplices[fi]
                p1 = points[simplex[0]]
                p2 = points[simplex[1]]
                p3 = points[simplex[2]]
                
                v1 = p2 - p1
                v2 = p3 - p1
                n = np.cross(v1, v2)
                norm_n = np.linalg.norm(n)
                if norm_n < 1e-10:
                    continue
                n = n / norm_n
                
                # Ensure outward normal
                if np.dot(n, p1) < 0:
                    n = -n
                
                normal_sum += n
            
            if np.linalg.norm(normal_sum) > 1e-10:
                # Move point toward the average normal direction
                target = normal_sum / np.linalg.norm(normal_sum)
                new_points[pi] = points[pi] + damping * (target - points[pi])
                new_points[pi] = new_points[pi] / np.linalg.norm(new_points[pi])
        
        new_points = _normalize(new_points)
        new_cost = _covering_radius(new_points)
        
        if new_cost < current_cost:
            current_cost = new_cost
            points = new_points
            if current_cost < best_cost:
                best_cost = current_cost
                best_points = points.copy()
        else:
            # If cost increased, reduce damping and try again with smaller step
            break
    
    return best_points


def _local_optimize_covering(points, time_limit=20):
    """
    Local optimization using L-BFGS-B on the covering radius.
    Note: _covering_radius is non-smooth, but L-BFGS-B might still work for polishing.
    """
    start_time = time.time()
    def objective(params):
        if time.time() - start_time > time_limit:
            return 1e6
        pts = params.reshape(N, 3)
        pts = _normalize(pts)
        return _covering_radius(pts)
        
    x0 = points.flatten()
    result = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-8})
    best_points = result.x.reshape(N, 3)
    return _normalize(best_points)


def _sa_optimize_covering(points, T_start=1.0, T_end=1e-4, n_iter=5000, time_limit=100):
    """
    Simulated Annealing to minimize covering radius directly.
    """
    start_time = time.time()
    points = _normalize(points).copy()
    current_cost = _covering_radius(points)
    best_points = points.copy()
    best_cost = current_cost
    
    cooling = (T_end / T_start) ** (1.0 / n_iter)
    
    for i in range(n_iter):
        if time.time() - start_time > time_limit:
            break
        T = T_start * (cooling ** i)
        
        idx = np.random.randint(N)
        original_point = points[idx].copy()
        
        step_size = 0.05 * T / T_start
        noise = np.random.normal(0, step_size, 3)
        new_point = original_point + noise
        new_point = new_point / np.linalg.norm(new_point)
        
        points[idx] = new_point
        
        new_cost = _covering_radius(points)
        delta_C = new_cost - current_cost
        
        if delta_C < 0 or np.random.rand() < np.exp(-delta_C / T):
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_points = points.copy()
        else:
            points[idx] = original_point
            
    return best_points


def construct():
    """Construct 50 points on the unit sphere minimizing the covering radius."""
    total_time_limit = 145.0  # Safety margin before 150s kill
    start_time = time.time()
    
    # 1. Initialize with Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = x0
    best_cost = _covering_radius(best_points)
    
    # 2. Spherical Lloyd's algorithm (Voronoi relaxation) - primary optimization
    time_spent = time.time() - start_time
    remaining_time = total_time_limit - time_spent
    if remaining_time > 30:
        points_lloyd = _spherical_lloyd_iteration(best_points, max_iter=100, time_limit=remaining_time - 40, damping=0.5)
        cost_lloyd = _covering_radius(points_lloyd)
        if cost_lloyd < best_cost:
            best_points = points_lloyd
            best_cost = cost_lloyd
    
    # 3. Simulated Annealing with time limit
    time_spent = time.time() - start_time
    remaining_time = total_time_limit - time_spent
    if remaining_time > 15:
        points_sa = _sa_optimize_covering(best_points, T_start=0.5, T_end=1e-4, n_iter=10000, time_limit=remaining_time - 20)
        cost_sa = _covering_radius(points_sa)
        if cost_sa < best_cost:
            best_points = points_sa
            best_cost = cost_sa
    
    # 4. Another round of Lloyd's for fine-tuning
    time_spent = time.time() - start_time
    remaining_time = total_time_limit - time_spent
    if remaining_time > 10:
        points_lloyd2 = _spherical_lloyd_iteration(best_points, max_iter=50, time_limit=remaining_time - 15, damping=0.3)
        cost_lloyd2 = _covering_radius(points_lloyd2)
        if cost_lloyd2 < best_cost:
            best_points = points_lloyd2
            best_cost = cost_lloyd2
    
    # 5. Local Polish with time limit
    time_spent = time.time() - start_time
    remaining_time = total_time_limit - time_spent
    if remaining_time > 5:
        points_final = _local_optimize_covering(best_points, time_limit=remaining_time - 5)
        cost_final = _covering_radius(points_final)
        if cost_final < best_cost:
            best_points = points_final
            best_cost = cost_final
    
    # 6. Final check and normalization
    best_points = _normalize(best_points)
    
    return best_points
# EVOLVE-BLOCK-END
