"""Spherical covering with 50 points via symplectic potential minimization."""
import numpy as np
from scipy.spatial import ConvexHull

N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms


def _compute_covering_radius(pts):
    """Compute the covering radius in degrees for points on the unit sphere."""
    try:
        hull = ConvexHull(pts)
        # Get facet normals and distances
        # For a convex hull of points on the unit sphere, each facet's circumradius
        # (angular) is determined by the facet's vertices.
        # The covering radius is the maximum over all facets of the angular distance
        # from the sphere to the facet plane (i.e., the cap height angle).
        
        # For each facet, compute the circumradius of the spherical polygon.
        # A simpler approach: for each facet, find the point on the sphere farthest
        # from the facet's vertices. This is equivalent to finding the maximum
        # angular distance from any point on the sphere to the nearest vertex.
        
        # Actually, the covering radius is the maximum over all points y on the sphere
        # of min_i angular_distance(y, p_i). This equals the maximum circumradius
        # of the Delaunay triangulation on the sphere, or equivalently, for each
        # facet of the convex hull, the angular distance from the sphere to the
        # facet's circumscribed circle.
        
        # For a facet with vertices v1, v2, ..., vk on the unit sphere, the covering
        # contribution is the angular radius of the smallest spherical cap centered
        # at some point that covers all vertices... no, that's not right.
        
        # The correct interpretation: the covering radius is the maximum over all
        # points on the sphere of the distance to the nearest point in the set.
        # This is equal to the maximum over all facets of the convex hull of the
        # angular distance from the center of the facet's circumcircle to the vertices.
        
        # For a planar facet of the convex hull, the point on the sphere farthest
        # from all vertices of that facet (and not closer to any other vertex) lies
        # along the outward normal of the facet. The angular distance is:
        #   arccos(h / d) where h is the distance from origin to facet plane,
        #   and we need the angular radius such that the cap covers the facet vertices.
        
        # More precisely: for a facet with plane n·x = h (where n is unit normal,
        # h > 0), the point on the sphere in direction n is at angular distance
        # arccos(h) from the origin... no.
        
        # Let me think again. The point on the unit sphere in the direction of the
        # facet normal n is at position n. The angular distance from n to a vertex v
        # is arccos(n·v). The covering radius contribution from this facet is the
        # maximum over vertices v of arccos(n·v), but we need the point on the sphere
        # that is farthest from ALL points, which for a facet is the point along n
        # only if n is the circumcenter direction.
        
        # Actually, for a convex hull facet, the relevant point is the one on the sphere
        # that is equidistant from all vertices of the facet (the circumcenter of the
        # spherical polygon). The angular distance from this point to each vertex is
        # the circumradius of that facet.
        
        # A practical approach: for each facet, compute the circumradius by finding
        # the point on the unit sphere that minimizes the maximum angular distance
        # to the facet's vertices. For a planar polygon on the sphere, this is the
        # intersection of the sphere with the line through the origin and the circumcenter
        # of the planar polygon (if the circumcenter is on the correct side).
        
        # Simpler: just do a dense grid check for the covering radius.
        # Generate many points on the sphere and compute min distance to each.
        
        # But that's slow. Let's use the facet approach properly.
        
        # For each facet, get its vertices, compute the circumcenter of the planar
        # polygon, project to the sphere, and compute the angular distance to vertices.
        
        max_radius = 0.0
        for simplex in hull.simplices:
            verts = pts[simplex]
            # Compute circumcenter of the triangle (all facets of a 3D convex hull are triangles)
            a, b, c = verts[0], verts[1], verts[2]
            
            # Circumcenter of triangle abc in 3D
            # Using the formula: circumcenter = a + ((|b-a|^2 * (c-a) - |c-a|^2 * (b-a)) cross (b-a) cross (c-a)) / (2 * |(b-a) cross (c-a)|^2)
            # Actually, let's use a simpler approach.
            
            # The circumcenter of a triangle on the unit sphere: we can find the point
            # equidistant from a, b, c on the sphere.
            
            # Method: solve for point p on sphere such that |p-a| = |p-b| = |p-c|.
            # This means p is in the plane perpendicular to the triangle's plane passing
            # through the circumcenter of the planar triangle.
            
            # Compute the normal to the triangle
            edge1 = b - a
            edge2 = c - a
            normal = np.cross(edge1, edge2)
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal = normal / norm
            
            # The circumcenter of the planar triangle lies along the normal direction
            # from the triangle's centroid... not exactly. Let's compute it properly.
            
            # For a triangle with vertices a, b, c, the circumcenter O satisfies:
            # |O-a|^2 = |O-b|^2 = |O-c|^2
            # This gives: 2*O·(b-a) = |b|^2 - |a|^2, etc.
            # Since all are on the unit sphere, |a|=|b|=|c|=1, so:
            # 2*O·(b-a) = 0, 2*O·(c-a) = 0
            # So O is perpendicular to both (b-a) and (c-a), meaning O is parallel to normal.
            
            # So O = t * normal for some t.
            # |O - a|^2 = |t*normal - a|^2 = t^2 - 2t*(normal·a) + |a|^2 = t^2 - 2t*(normal·a) + 1
            # We want this to be minimized... no, we want O on the unit sphere, so |O|=1,
            # which means |t*normal|=1, so t = ±1.
            
            # Wait, that's only if O is on the sphere. But the circumcenter of the spherical
            # triangle is the point on the sphere equidistant from all three vertices.
            # It lies along the direction of the normal from the origin.
            
            # So O = ±normal. We choose the sign such that O is on the same side as the
            # triangle (i.e., O·a > 0, etc., or more precisely, O is the outward normal).
            
            # The angular distance from O to a is arccos(O·a).
            # We want the maximum over the three vertices.
            
            # Choose the sign of normal such that it points outward (away from origin,
            # which is always the case for a convex hull facet).
            
            # Check: for a convex hull, the facet normal should point outward.
            # Let's verify by checking that the centroid of the triangle is on the
            # correct side.
            
            centroid = (a + b + c) / 3.0
            if np.dot(centroid, normal) < 0:
                normal = -normal
            
            # Now normal points outward. The circumcenter on the sphere is normal.
            # Angular distances to vertices:
            d_a = np.arccos(np.clip(np.dot(normal, a), -1, 1))
            d_b = np.arccos(np.clip(np.dot(normal, b), -1, 1))
            d_c = np.arccos(np.clip(np.dot(normal, c), -1, 1))
            
            facet_radius = max(d_a, d_b, d_c)
            if facet_radius > max_radius:
                max_radius = facet_radius
        
        return np.degrees(max_radius)
    
    except Exception:
        # Fallback: dense grid
        n_theta = 100
        n_phi = 200
        thetas = np.linspace(0, np.pi, n_theta)
        phis = np.linspace(0, 2 * np.pi, n_phi)
        TH, PH = np.meshgrid(thetas, phis, indexing='ij')
        Y = np.zeros((n_theta, n_phi, 3))
        Y[:, :, 0] = np.sin(TH) * np.cos(PH)
        Y[:, :, 1] = np.sin(TH) * np.sin(PH)
        Y[:, :, 2] = np.cos(TH)
        Y = Y.reshape(-1, 3)
        
        # Compute distances
        # Y @ pts.T gives cos of angular distances
        dots = Y @ pts.T
        min_dots = np.max(dots, axis=1)  # max dot = min angular distance
        max_angular = np.arccos(np.clip(min_dots, -1, 1))
        return np.degrees(np.max(max_angular))


def construct():
    """
    Use symplectic velocity Verlet integration to minimize a repulsive potential
    on the unit sphere, aiming to find a good covering configuration.
    """
    # Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Symplectic integrator parameters
    dt = 0.005
    n_steps = 5000
    gamma = 0.1  # damping (small, to preserve symplectic nature mostly)
    power = 4  # inverse power law
    
    # Velocity initialization
    vel = np.zeros((N, 3))
    
    for step in range(n_steps):
        # Compute forces: repulsive between all pairs + constraint to sphere
        forces = np.zeros((N, 3))
        
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                diff = pts[i] - pts[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    # Jitter
                    diff = np.random.randn(3) * 1e-6
                    dist = np.linalg.norm(diff)
                # Repulsive force: proportional to 1/dist^(power+1) in direction of diff
                force_mag = 1.0 / (dist ** (power + 1))
                forces[i] += force_mag * (diff / dist)
        
        # Add centripetal force to keep points on sphere (Lagrange multiplier approach)
        # For points on the unit sphere, the constraint force is along the radial direction
        # to maintain |p| = 1. We add a force that pushes points back to the sphere.
        for i in range(N):
            r_i = np.linalg.norm(pts[i])
            if r_i > 1e-10:
                # Force to maintain unit length: -2 * (r_i - 1) * p_i / r_i
                forces[i] -= 2.0 * (r_i - 1.0) * (pts[i] / r_i)
        
        # Velocity Verlet step
        # v_half = v + 0.5 * dt * F
        vel_half = vel + 0.5 * dt * forces
        
        # p_new = p + dt * v_half
        pts_new = pts + dt * vel_half
        
        # Normalize to unit sphere
        pts_new = _normalize(pts_new)
        
        # Compute forces at new position
        forces_new = np.zeros((N, 3))
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                diff = pts_new[i] - pts_new[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    diff = np.random.randn(3) * 1e-6
                    dist = np.linalg.norm(diff)
                force_mag = 1.0 / (dist ** (power + 1))
                forces_new[i] += force_mag * (diff / dist)
        
        for i in range(N):
            r_i = np.linalg.norm(pts_new[i])
            if r_i > 1e-10:
                forces_new[i] -= 2.0 * (r_i - 1.0) * (pts_new[i] / r_i)
        
        # v_new = v_half + 0.5 * dt * F_new
        vel_new = vel_half + 0.5 * dt * forces_new
        
        # Apply damping to prevent energy growth
        vel = vel_new * (1.0 - gamma * dt)
        pts = pts_new
    
    # Final normalization
    pts = _normalize(pts)
    
    return pts
# EVOLVE-BLOCK-END
