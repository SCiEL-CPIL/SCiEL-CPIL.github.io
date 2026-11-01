"""Optimized covering of the unit sphere with 130 equal caps."""
import numpy as np
import time
import sys

N = 130

def _covering_radius(points):
    """
    Compute the covering radius in degrees.
    The covering radius is the maximum, over all points on the sphere, of the
    angular distance to the nearest point in `points`.
    This equals the largest circumradius among the facets of the convex hull
    of the points (assuming the origin is inside the convex hull).
    
    We use a grid-based approach for robustness, but also check via the convex hull
    if scipy is available. For speed and reliability in 120s, we use a dense grid.
    """
    # Use a reasonably fine grid on the sphere
    # The grid should be fine enough that the error is much smaller than the expected improvement
    # 130 points -> covering radius ~11.3 degrees ~ 0.197 radians
    # A grid with spacing ~1 degree should be adequate for accurate measurement
    # But to be safe, use ~0.5 degree spacing
    # Number of points: ~ (180/0.5)^2 * 2 = 129600, which is fine
    
    # Actually, let's use a more efficient approach:
    # The covering radius is determined by the Voronoi cells on the sphere.
    # The farthest point from any generator in a Voronoi cell is one of the vertices
    # of the Voronoi cell (the circumcenters of the Delaunay triangles).
    # So we need the convex hull, find the facets, and for each facet compute the
    # circumradius (angular distance from the facet's plane to the sphere, i.e., 
    # the angle whose cosine is the distance from origin to the facet plane).
    
    # For robustness, let's use both: compute via convex hull if possible, 
    # and verify with a grid.
    
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(points)
        # For each facet (simplex), compute the circumradius
        # A facet is defined by N+1 points (in 3D, 4 points for a tetrahedron, but 
        # the hull facets are triangles in 3D)
        # Wait, in 3D, the convex hull facets are triangles (3 points each)
        # The covering radius for a facet is the angular distance from the origin 
        # to the farthest point on the sphere that is closest to that facet's circumcenter.
        
        # Actually, the covering radius is the maximum over all facets of the 
        # angular distance from the origin to the facet plane.
        # For a facet with normal n and offset d (n·x = d), the angular distance 
        # from the origin to the plane is arccos(d) if d > 0, but we need the 
        # distance from any point on the sphere to the nearest point in the configuration.
        
        # Let me reconsider. The covering radius R satisfies:
        # For every point y on the sphere, min_i angle(y, p_i) <= R
        # This is equivalent to: the union of caps of angular radius R centered at p_i covers the sphere.
        # The critical points are the vertices of the Voronoi diagram on the sphere,
        # which are the circumcenters of the Delaunay triangles.
        
        # For a Delaunay triangle (p1, p2, p3), the circumcenter c is the point on the sphere
        # equidistant from p1, p2, p3. The angular distance from c to each of p1, p2, p3 
        # is the circumradius of that triangle.
        
        # So the covering radius is the maximum circumradius over all Delaunay triangles.
        
        # Using ConvexHull, we get the facets. In 3D, each facet is a triangle.
        # For each triangular facet, compute the circumradius.
        
        max_circumradius = 0.0
        for simplex in hull.simplices:
            p1, p2, p3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            # Compute circumradius of the spherical triangle
            # The circumradius is the angular distance from the circumcenter to each vertex.
            # We can compute this as follows:
            # Find the point c on the sphere such that c·p1 = c·p2 = c·p3 = cos(R)
            # This is equivalent to finding the intersection of the sphere with the plane
            # equidistant from p1, p2, p3.
            
            # The circumcenter of a spherical triangle can be found by:
            # c = normalize(p1 * sin(angle_opposite_1) + p2 * sin(angle_opposite_2) + p3 * sin(angle_opposite_3))
            # But there's a simpler formula using cross products.
            
            # Alternative: the circumradius R satisfies:
            # cos(R) = (p1·p2 * p2·p3 * p3·p1 + 2*(p1·p2)*(p2·p3)*(p3·p1) - (p1·p2)^2 - (p2·p3)^2 - (p3·p1)^2) / ...
            # This is getting complicated. Let's use a numerical approach.
            
            # Solve for c such that c·p1 = c·p2 = c·p3 and ||c|| = 1
            # This means c is orthogonal to (p1 - p2) and (p1 - p3)
            # So c is parallel to (p1 - p2) × (p1 - p3)
            
            v1 = p1 - p2
            v2 = p1 - p3
            n = np.cross(v1, v2)
            if np.linalg.norm(n) < 1e-10:
                continue  # Degenerate triangle
            c = n / np.linalg.norm(n)
            
            # The circumradius is the angular distance from c to p1 (or p2 or p3)
            cos_r = np.dot(c, p1)
            cos_r = np.clip(cos_r, -1.0, 1.0)
            r = np.arccos(cos_r)
            
            if r > max_circumradius:
                max_circumradius = r
        
        return np.degrees(max_circumradius)
    except Exception:
        # Fallback to grid-based computation
        # Create a grid on the sphere
        n_theta = 360
        n_phi = 180
        theta = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
        phi = np.linspace(0, np.pi, n_phi)
        
        # Create meshgrid
        Theta, Phi = np.meshgrid(theta, phi)
        
        # Convert to Cartesian
        x = np.sin(Phi) * np.cos(Theta)
        y = np.sin(Phi) * np.sin(Theta)
        z = np.cos(Phi)
        
        grid_points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
        
        # For each grid point, find the nearest configuration point
        # Compute angular distance
        max_dist = 0.0
        for i in range(0, len(grid_points), 10):  # Process in batches
            batch = grid_points[i:i+10]
            # Compute dot products
            dots = batch @ points.T  # shape (10, N)
            cos_dists = np.clip(dots, -1.0, 1.0)
            dists = np.arccos(cos_dists)
            min_dists = np.min(dists, axis=1)
            batch_max = np.max(min_dists)
            if batch_max > max_dist:
                max_dist = batch_max
        
        return np.degrees(max_dist)


def _optimize_covering(points, max_time=110.0):
    """
    Optimize the covering by pushing points away from the centers of the largest facets.
    Uses a gradient-free approach: repeatedly find the facet with the largest circumradius,
    and move the points of that facet outward (away from the origin) along their normal directions.
    """
    points = points.copy()
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    start_time = time.time()
    best_points = points.copy()
    best_radius = _covering_radius(points)
    
    # Parameters
    step_size = 0.05
    decay = 0.995
    min_step = 1e-6
    
    iteration = 0
    while time.time() - start_time < max_time:
        iteration += 1
        
        # Compute covering radius and identify the worst facet
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(points)
            
            max_circumradius = 0.0
            worst_simplex = None
            for simplex in hull.simplices:
                p1, p2, p3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
                v1 = p1 - p2
                v2 = p1 - p3
                n = np.cross(v1, v2)
                if np.linalg.norm(n) < 1e-10:
                    continue
                c = n / np.linalg.norm(n)
                cos_r = np.dot(c, p1)
                cos_r = np.clip(cos_r, -1.0, 1.0)
                r = np.arccos(cos_r)
                if r > max_circumradius:
                    max_circumradius = r
                    worst_simplex = simplex
            
            if worst_simplex is None:
                break
            
            # Move the points of the worst facet
            # The idea: push the points of the worst facet slightly outward
            # along the direction that increases the circumradius reduction
            
            # A simple heuristic: move each point in the worst simplex
            # in the direction away from the circumcenter
            p1, p2, p3 = points[worst_simplex[0]], points[worst_simplex[1]], points[worst_simplex[2]]
            v1 = p1 - p2
            v2 = p1 - p3
            n = np.cross(v1, v2)
            c = n / np.linalg.norm(n)
            
            # Move each point in the worst simplex away from c
            for idx in worst_simplex:
                # Direction away from c
                direction = points[idx] - np.dot(points[idx], c) * c
                direction = direction / (np.linalg.norm(direction) + 1e-10)
                points[idx] += step_size * direction
                points[idx] = points[idx] / np.linalg.norm(points[idx])
            
            # Also, to maintain balance, we could pull other points slightly,
            # but let's keep it simple for now.
            
            # Recompute radius
            current_radius = _covering_radius(points)
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = points.copy()
            
            # Decay step size
            step_size *= decay
            if step_size < min_step:
                step_size = min_step
                
        except Exception:
            # If convex hull fails, break
            break
    
    return best_points, best_radius


def construct():
    """
    Construct 130 points on the unit sphere that minimize the covering radius.
    """
    start_time = time.time()
    deadline = 115.0  # Leave 5s safety margin
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    # Optimize
    try:
        optimized_points, radius = _optimize_covering(points, max_time=110.0)
        return optimized_points
    except Exception:
        # Return the initial points if optimization fails
        return points


# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
