"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time

def _compute_covering_radius(points):
    """Compute the covering radius (in degrees) for points on unit sphere."""
    try:
        hull = ConvexHull(points)
        # For each facet, compute the circumradius
        # The circumradius of a facet is the angular distance from the origin to the facet plane
        # Actually, for a point on the sphere, the covering radius is the max over all points on sphere
        # of the angular distance to the nearest point. This equals the max circumradius of the
        # Voronoi cells, which for the convex hull corresponds to the facets.
        
        # For a facet with vertices v1, v2, ..., vk, the farthest point on the sphere from all
        # points in the configuration that is covered by this facet's Voronoi cell is the
        # circumcenter of the facet (in angular terms).
        
        # The angular distance from the origin to the facet plane is the circumradius.
        # For a plane defined by points, the distance from origin to plane = |n · p| where n is normal
        # and p is a point on the plane. Since points are on unit sphere, the circumradius (angular)
        # is arccos(distance from origin to plane).
        
        max_cos = 1.0  # cos of min angular distance = max cos
        for simplex in hull.simplices:
            verts = points[simplex]
            # Compute normal to the facet
            if len(verts) == 3:
                # Triangle
                v1, v2, v3 = verts
                edge1 = v2 - v1
                edge2 = v3 - v1
                normal = np.cross(edge1, edge2)
                norm = np.linalg.norm(normal)
                if norm < 1e-12:
                    continue
                normal = normal / norm
                # Distance from origin to plane
                dist = np.abs(np.dot(normal, v1))
                # Clamp for numerical stability
                dist = np.clip(dist, -1.0, 1.0)
                cos_angle = dist
            elif len(verts) == 4:
                # Tetrahedron facet - use least squares to find plane
                # Actually, ConvexHull in 3D gives triangular facets only
                v1, v2, v3 = verts[0], verts[1], verts[2]
                edge1 = v2 - v1
                edge2 = v3 - v1
                normal = np.cross(edge1, edge2)
                norm = np.linalg.norm(normal)
                if norm < 1e-12:
                    continue
                normal = normal / norm
                dist = np.abs(np.dot(normal, v1))
                dist = np.clip(dist, -1.0, 1.0)
                cos_angle = dist
            else:
                continue
            
            max_cos = min(max_cos, cos_angle)
        
        if max_cos > 1.0:
            max_cos = 1.0
        if max_cos < -1.0:
            max_cos = -1.0
        
        radius_rad = np.arccos(max_cos)
        return np.degrees(radius_rad)
    except Exception:
        return 180.0  # Fallback: worst case


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_radius = _compute_covering_radius(best_points)
    
    # Local optimization using gradient-free approach
    # Use a simple iterative method: move each point to reduce local covering radius
    
    lr = 0.01
    for iteration in range(500):
        if time.time() > deadline:
            break
        
        # For each point, compute a gradient-like direction
        # Move points away from each other to maximize minimum distance
        # This is a simplified approach - actual gradient would be more complex
        
        new_points = points.copy()
        for idx in range(N):
            # Compute forces from other points
            forces = np.zeros(3)
            for jdx in range(N):
                if jdx == idx:
                    continue
                diff = points[idx] - points[jdx]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    # Random perturbation for coincident points
                    forces += np.random.randn(3) * 0.1
                else:
                    # Repulsive force, stronger when close
                    force_mag = 1.0 / (dist * dist + 0.01)
                    forces += force_mag * diff / dist
            
            # Normalize and apply
            force_norm = np.linalg.norm(forces)
            if force_norm > 1e-10:
                new_points[idx] = points[idx] + lr * forces / force_norm
                # Renormalize to unit sphere
                new_points[idx] /= np.linalg.norm(new_points[idx])
        
        points = new_points
        current_radius = _compute_covering_radius(points)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
    
    return best_points
# EVOLVE-BLOCK-END
