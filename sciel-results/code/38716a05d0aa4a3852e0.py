"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 130
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0  # seconds

def _covering_radius(points):
    """Compute the covering radius in degrees for points on unit sphere."""
    # points: (N, 3) array of unit vectors
    # The covering radius is the maximum angular distance from any point on the sphere
    # to the nearest configuration point.
    # This equals the largest circumradius of the facets of the convex hull.
    try:
        hull = ConvexHull(points)
        # For each facet, find the circumradius (angular distance from facet center to vertices)
        # Actually, for a sphere, the covering radius is determined by the largest empty cap.
        # The facets of the convex hull correspond to regions where the nearest point is the facet's vertices.
        # The center of the largest empty cap is the circumcenter of the facet (on the sphere).
        # For a facet with vertices v1, v2, ..., vk, the circumcenter on the sphere is the direction
        # that is equidistant from all vertices.
        
        # Alternative approach: for each facet, compute the normal direction (circumcenter direction)
        # and find the maximum angular distance to the facet's vertices.
        
        max_radius = 0.0
        for simplex in hull.simplices:
            # Get the vertices of this facet
            verts = points[simplex]
            # The facet is a planar polygon. Its circumcenter (on the sphere) is the point
            # equidistant from all vertices. For a spherical cap, this is the direction
            # perpendicular to the facet plane, pointing outward.
            
            # Compute the plane normal
            if len(verts) >= 3:
                # Use the first three vertices to define the plane
                v1, v2, v3 = verts[0], verts[1], verts[2]
                normal = np.cross(v2 - v1, v3 - v1)
                norm = np.linalg.norm(normal)
                if norm < 1e-10:
                    continue
                normal = normal / norm
                # Ensure normal points outward (away from origin)
                # The centroid of the facet should be in the direction of the normal
                centroid = np.mean(verts, axis=0)
                if np.dot(centroid, normal) < 0:
                    normal = -normal
                
                # The circumradius is the angle between the normal and any vertex
                # All vertices should be at the same angular distance from the normal
                cos_angles = np.dot(verts, normal)
                # Due to numerical issues, they might not be exactly equal
                # Use the minimum cos (maximum angle) to be safe
                min_cos = np.min(cos_angles)
                min_cos = np.clip(min_cos, -1.0, 1.0)
                angle = np.arccos(min_cos)
                radius_deg = np.degrees(angle)
                if radius_deg > max_radius:
                    max_radius = radius_deg
        return max_radius
    except Exception:
        return 180.0  # Invalid configuration


def _objective_and_gradient(x, points_flat):
    """Objective: maximize the minimum distance from any point on sphere to nearest point.
    We minimize the negative of this, or equivalently minimize the covering radius.
    
    Actually, we want to minimize the covering radius. The covering radius is the max over
    facets of the circumradius. This is a nonsmooth function.
    
    For optimization, we can use a different approach: minimize the maximum over a set of
    test points, or use the fact that the covering radius equals the max circumradius of hull facets.
    
    Let's use a simpler approach: for each pair of points, ensure they are sufficiently far apart.
    But that's not quite right either.
    
    Better approach: The covering radius R satisfies that for every direction y on the sphere,
    there exists a point p_i such that angle(y, p_i) <= R.
    
    Equivalently, the maximum over y of min_i angle(y, p_i) = R.
    
    This is the same as: the largest empty spherical cap has angular radius R.
    
    For optimization, we can parameterize the points and use gradient-based methods.
    The gradient of the covering radius with respect to point positions is complex.
    
    Alternative: Use a penalty method. Minimize sum of max(0, d_ij - target)^2 for all pairs,
    where d_ij is the angular distance, and target is some desired minimum separation.
    But this doesn't directly minimize the covering radius.
    
    Let's try a different formulation. The covering radius is determined by the facets.
    For each facet, the circumradius is a smooth function of the vertex positions (as long as
    the facet remains a facet). We can minimize the maximum of these circumradii.
    
    For now, let's use a simpler approach: start from a good initial configuration and
    use local optimization to minimize the covering radius.
    """
    points = x.reshape(-1, 3)
    # Normalize points to unit sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    points = points / norms
    
    # Compute covering radius
    cr = _covering_radius(points)
    return cr


def construct():
    start_time = time.time()
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    current_cr = _covering_radius(points)
    best_points = points.copy()
    best_cr = current_cr
    
    # Try to improve using local optimization
    # We'll use a simple iterative approach: for each point, try to move it to reduce
    # the covering radius.
    
    # First, let's try a gradient-free local search
    # For each point, compute the "pressure" from neighboring points and move it away.
    
    for iteration in range(50):
        if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        # For each point, compute the force from nearby points
        # Move each point in the direction that reduces the local density
        
        new_points = points.copy()
        for i in range(N):
            # Find the nearest neighbors
            dists = np.arccos(np.clip(points @ points[i], -1.0, 1.0))
            nearest_idx = np.argsort(dists)[1:6]  # 5 nearest neighbors (excluding self)
            
            # Compute the centroid of the nearest neighbors
            neighbor_centroid = np.mean(points[nearest_idx], axis=0)
            neighbor_centroid_norm = np.linalg.norm(neighbor_centroid)
            if neighbor_centroid_norm < 1e-10:
                continue
            neighbor_centroid = neighbor_centroid / neighbor_centroid_norm
            
            # Move point i away from the neighbor centroid
            # The direction to move is away from the centroid, projected onto the sphere
            move_dir = points[i] - np.dot(points[i], neighbor_centroid) * neighbor_centroid
            move_dir_norm = np.linalg.norm(move_dir)
            if move_dir_norm < 1e-10:
                continue
            move_dir = move_dir / move_dir_norm
            
            # Small step
            step_size = 0.01
            new_point = points[i] + step_size * move_dir
            new_point_norm = np.linalg.norm(new_point)
            new_point = new_point / new_point_norm
            new_points[i] = new_point
        
        new_cr = _covering_radius(new_points)
        if new_cr < current_cr:
            points = new_points
            current_cr = new_cr
            if current_cr < best_cr:
                best_cr = current_cr
                best_points = points.copy()
        else:
            # Reduce step size
            break
    
    # Final normalization
    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points
