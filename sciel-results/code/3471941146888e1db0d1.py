"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time


def _covering_radius(points):
    """Compute the covering radius in degrees for points on the unit sphere.
    
    The covering radius is the maximum angular distance from any point on the sphere
    to its nearest configuration point. This equals the largest circumradius of the
    facets of the convex hull (for facets not containing the origin in their interior).
    """
    if len(points) < 4:
        # Fallback: use grid-based approximation
        return _covering_radius_grid(points)
    
    try:
        hull = ConvexHull(points)
    except Exception:
        return _covering_radius_grid(points)
    
    # For each facet, compute the circumradius
    # A facet is defined by points hull.simplices
    # The circumradius of a spherical cap defined by a facet is related to the 
    # angular distance from the origin to the facet plane.
    
    # For a convex hull of points on the unit sphere, the covering radius for each facet
    # is the angle between the origin and the closest point on the facet plane.
    # For a triangular facet with vertices v1, v2, v3 (unit vectors), the normal is
    # n = (v2-v1) x (v3-v1), and the distance from origin to the plane is |n| / |n| 
    # ... actually we need the perpendicular distance from origin to the plane containing the facet.
    
    # The plane equation: n . x = d, where n is the normal and d is the offset.
    # For points on the unit sphere, the distance from origin to the plane is |d|/|n|.
    # The covering radius for that facet is arccos(|d|/|n|) if the origin is on the 
    # "inside" side (which it should be for a convex hull of points on the sphere).
    
    max_radius = 0.0
    for simplex in hull.simplices:
        v1 = points[simplex[0]]
        v2 = points[simplex[1]]
        v3 = points[simplex[2]]
        
        # Normal to the facet
        e1 = v2 - v1
        e2 = v3 - v1
        n = np.cross(e1, e2)
        n_norm = np.linalg.norm(n)
        
        if n_norm < 1e-15:
            continue
        
        # Distance from origin to the plane
        # Plane: n . (x - v1) = 0 => n . x = n . v1
        d = np.dot(n, v1)
        dist = abs(d) / n_norm
        
        # The angular distance from origin to the plane is arccos(dist)
        # but we need to be careful: if dist > 1, something is wrong
        if dist > 1.0:
            dist = 1.0
        elif dist < -1.0:
            dist = -1.0
        
        # The covering radius for this facet is the angle from the origin to the 
        # closest point on the facet. For a convex hull of unit vectors, the origin
        # is inside, so the closest point on the facet to the origin is the foot of
        # the perpendicular. The angular distance is arccos(dist).
        angle = np.arccos(min(max(dist, -1.0), 1.0))
        
        if angle > max_radius:
            max_radius = angle
    
    return np.degrees(max_radius)


def _covering_radius_grid(points, n_theta=720, n_phi=1440):
    """Fallback covering radius computation using a grid."""
    # Create a grid of points on the sphere
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    
    # For efficiency, use a smaller grid
    theta = np.linspace(0, np.pi, 360)
    phi = np.linspace(0, 2 * np.pi, 720)
    
    T, P = np.meshgrid(theta, phi, indexing='ij')
    
    # Convert to Cartesian
    x = np.sin(T) * np.cos(P)
    y = np.sin(T) * np.sin(P)
    z = np.cos(T)
    
    grid_points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    
    # For each grid point, find the nearest configuration point
    # This is O(n_grid * N), which for 360*720 = 259200 grid points and N=130 is ~34M operations
    # Use chunked computation to avoid memory issues
    
    max_dist = 0.0
    chunk_size = 10000
    for start in range(0, len(grid_points), chunk_size):
        end = min(start + chunk_size, len(grid_points))
        chunk = grid_points[start:end]
        
        # Compute distances (dot products)
        # dot = chunk @ points.T
        dots = chunk @ points.T
        # Clip to [-1, 1]
        dots = np.clip(dots, -1.0, 1.0)
        # Max dot product gives min angular distance
        max_dots = dots.max(axis=1)
        # Angular distance = arccos(max_dot)
        angles = np.arccos(max_dots)
        if angles.max() > max_dist:
            max_dist = angles.max()
    
    return np.degrees(max_dist)


def _refine_points(points, max_iters=50, time_limit=100.0):
    """Iteratively refine points to minimize covering radius using Lloyd-like iterations."""
    start_time = time.time()
    current = points.copy()
    
    for iteration in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # For each point, compute its Voronoi cell (approximated by nearest grid points)
        # and move it to the centroid of its cell (in angular space)
        
        # Use a moderate-resolution grid for efficiency
        n_theta = 360
        n_phi = 720
        theta = np.linspace(0, np.pi, n_theta)
        phi = np.linspace(0, 2 * np.pi, n_phi)
        T, P = np.meshgrid(theta, phi, indexing='ij')
        x = np.sin(T) * np.cos(P)
        y = np.sin(T) * np.sin(P)
        z = np.cos(T)
        grid_points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
        
        # Assign each grid point to its nearest configuration point
        # chunked for memory
        assignments = np.zeros(len(grid_points), dtype=int)
        chunk_size = 20000
        for start in range(0, len(grid_points), chunk_size):
            end = min(start + chunk_size, len(grid_points))
            chunk = grid_points[start:end]
            dots = chunk @ current.T
            assignments[start:end] = dots.argmax(axis=1)
        
        # For each point, compute the mean of its assigned grid points
        new_points = np.zeros_like(current)
        for i in range(len(current)):
            mask = assignments == i
            if np.sum(mask) > 0:
                centroid = grid_points[mask].mean(axis=0)
                norm = np.linalg.norm(centroid)
                if norm > 1e-10:
                    new_points[i] = centroid / norm
                else:
                    new_points[i] = current[i]
            else:
                new_points[i] = current[i]
        
        # Check if we made progress
        old_radius = _covering_radius_grid(current, 180, 360)
        new_radius = _covering_radius_grid(new_points, 180, 360)
        
        if new_radius < old_radius:
            current = new_points
        else:
            # Try smaller step
            alpha = 0.5
            blended = (1 - alpha) * current + alpha * new_points
            norms = np.linalg.norm(blended, axis=1, keepdims=True)
            blended = blended / norms
            blended_radius = _covering_radius_grid(blended, 180, 360)
            if blended_radius < old_radius:
                current = blended
            else:
                break
        
        # Check for coincident points
        for i in range(len(current)):
            for j in range(i + 1, len(current)):
                dot = np.dot(current[i], current[j])
                if dot > 0.999999:
                    # Nudge apart
                    diff = current[i] - current[j]
                    norm = np.linalg.norm(diff)
                    if norm < 1e-10:
                        diff = np.array([1.0, 0.0, 0.0])
                        norm = 1.0
                    diff /= norm
                    current[i] = current[i] + 0.01 * diff
                    current[j] = current[j] - 0.01 * diff
                    current[i] /= np.linalg.norm(current[i])
                    current[j] /= np.linalg.norm(current[j])
    
    return current


def construct():
    """Construct 130 points on the unit sphere minimizing the covering radius."""
    start_time = time.time()
    deadline = 110.0  # Leave safety margin before 120s limit
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Refine using Lloyd-like iterations
    if time.time() - start_time < deadline:
        points = _refine_points(points, max_iters=30, time_limit=deadline - (time.time() - start_time))
    
    # Final normalization
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    # Ensure no coincident points
    for i in range(N):
        for j in range(i + 1, N):
            dot = np.dot(points[i], points[j])
            if dot > 0.9999999:
                # Perturb
                points[i] += 0.001 * np.random.randn(3)
                points[i] /= np.linalg.norm(points[i])
    
    # Final check and normalization
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    return points
# EVOLVE-BLOCK-END
