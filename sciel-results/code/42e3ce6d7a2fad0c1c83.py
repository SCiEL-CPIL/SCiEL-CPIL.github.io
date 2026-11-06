"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import math


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms


def _covering_radius_deg(pts):
    """Compute covering radius in degrees via convex hull facets."""
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 180.0

    # Get facet normals (outward) and distances
    # For a unit sphere, the circumradius of a facet is determined by the
    # distance from the origin to the facet plane and the radius of the
    # spherical cap.
    # For each facet, the covering radius contribution is the angular distance
    # from the sphere center to the farthest point in the facet's spherical polygon.
    # For a simplex facet (triangle in 3D), the circumcenter on the sphere is
    # along the outward normal.
    
    # Simpler approach: for each facet, compute the circumradius as the max
    # angular distance from the facet's vertices to the circumcenter.
    # The circumcenter of a facet on the unit sphere is in the direction of
    # the outward normal of the facet plane.
    
    # For each facet, the plane has normal n and equation n·x = d where d > 0.
    # The circumcenter on the unit sphere is at n/|n| (since the sphere is unit).
    # The angular distance from a vertex v to the circumcenter c is arccos(v·c).
    # The covering radius for this facet is the max over vertices of this distance.
    # But actually, the farthest point from all vertices in the spherical polygon
    # is the circumcenter only if it lies within the polygon. For convex hull
    # facets, the circumcenter (pole) should be inside the spherical facet.
    
    # The angular radius of the cap is arccos(d / |n|) where the plane is n·x = d.
    # Wait, let me think more carefully.
    
    # The facet is a planar polygon. The circumcircle of this polygon on the
    # sphere: the center is the intersection of the line through the origin
    # perpendicular to the facet plane with the unit sphere. That point is
    # c = n_hat where n_hat is the unit normal to the facet plane.
    # The angular radius is arccos(c · v) for any vertex v of the facet.
    # All vertices should give the same value (within numerical error) if
    # the circumcenter is correct.
    
    max_rad = 0.0
    for simplex in hull.simplices:
        # Get the vertices of this facet
        verts = pts[simplex]
        
        # Compute the normal to the facet plane
        if len(verts) == 3:
            v0, v1, v2 = verts
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
        elif len(verts) == 4:
            # For a quadrilateral, use the first 3 to get the normal
            v0, v1, v2 = verts[0], verts[1], verts[2]
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
        else:
            continue
        
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-15:
            continue
        normal = normal / norm_len
        
        # Ensure normal points outward (away from origin)
        centroid = np.mean(verts, axis=0)
        if np.dot(normal, centroid) < 0:
            normal = -normal
        
        # The circumcenter on the unit sphere is in the direction of the normal
        # The angular distance from a vertex to this point
        for v in verts:
            cos_ang = np.clip(np.dot(v, normal), -1.0, 1.0)
            ang = np.arccos(cos_ang)
            if ang > max_rad:
                max_rad = ang
    
    return math.degrees(max_rad)


def _repulsion_force(pts, k=5):
    """Compute repulsive forces between nearby points to spread them out."""
    n = len(pts)
    forces = np.zeros_like(pts)
    
    # Pairwise distances
    for i in range(n):
        for j in range(i + 1, n):
            d = pts[i] - pts[j]
            dist = np.linalg.norm(d)
            if dist < 1e-10:
                continue
            # Repulsion: force proportional to 1/dist^2 in the direction away
            # Only apply for nearby points to save time and focus on gaps
            if dist < 0.5:
                f = d / (dist ** 3)
                forces[i] += f
                forces[j] -= f
    
    return forces


def _project_to_sphere(pts):
    """Project points back to the unit sphere."""
    return _normalize(pts)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = _normalize(pts)
    
    best_pts = pts.copy()
    best_cr = _covering_radius_deg(pts)
    
    # Optimization loop
    lr = 0.01
    min_lr = 1e-6
    
    while time.time() < deadline:
        # Compute repulsion forces
        forces = _repulsion_force(pts)
        
        # Update positions
        pts = pts + lr * forces
        pts = _project_to_sphere(pts)
        
        # Evaluate
        cr = _covering_radius_deg(pts)
        
        if cr < best_cr:
            best_cr = cr
            best_pts = pts.copy()
            # Slightly increase learning rate if improving
            lr = min(lr * 1.1, 0.05)
        else:
            # Decrease learning rate
            lr *= 0.95
            if lr < min_lr:
                break
        
        # Check deadline
        if time.time() > deadline - 10.0:
            break
    
    return best_pts
# EVOLVE-BLOCK-END
