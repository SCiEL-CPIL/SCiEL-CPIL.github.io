"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time


def _compute_covering_radius(pts):
    """Compute the covering radius in radians from the convex hull facets."""
    try:
        hull = ConvexHull(pts)
        max_r = 0.0
        for simplex in hull.simplices:
            triangle = pts[simplex]
            # Compute circumradius of the spherical triangle
            # For three points on unit sphere, the circumradius of the facet
            # is the angular radius of the smallest circle on the sphere containing them
            a = triangle[0]
            b = triangle[1]
            c = triangle[2]
            
            # Side lengths (angular distances)
            ab = np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))
            bc = np.arccos(np.clip(np.dot(b, c), -1.0, 1.0))
            ca = np.arccos(np.clip(np.dot(c, a), -1.0, 1.0))
            
            # Spherical circumradius using formula:
            # R = arccos((cos(ab)cos(bc) + cos(ca)) / (sin(ab)sin(bc)))  -- not quite
            # Better: use the formula for circumradius of spherical triangle
            # Using the law of cosines approach:
            # The circumcenter is equidistant from all three vertices.
            # For a spherical triangle with sides a, b, c (angular),
            # the circumradius R satisfies:
            # sin(R) = sin(a)sin(b)sin(c) / (2 * area_factor) -- complex
            
            # Simpler approach: the circumradius of the planar triangle formed by
            # projecting onto the plane through the three points, then convert.
            # Actually, the covering radius contribution from a facet is the angular
            # distance from the facet's circumcenter (on the sphere) to the vertices.
            
            # Compute the circumcenter of the three points on the unit sphere.
            # The circumcenter is the point on the sphere equidistant from all three.
            # It lies in the direction of the normal to the plane through a,b,c,
            # but adjusted. Actually, for three points on a unit sphere, the circumcenter
            # is at the intersection of the sphere with the line through the origin 
            # perpendicular to the plane containing a, b, c... no, that's the normal.
            
            # Let's use: the circumcenter direction is proportional to:
            # n = (a × b) · c ... no.
            
            # The circumcenter of three points on a sphere is the point p on the sphere
            # such that |p-a| = |p-b| = |p-c|.
            # This means p · (a-b) = 0.5(|a|^2 - |b|^2) = 0 since all unit.
            # So p · (a-b) = 0, p · (b-c) = 0, p · a = p · b = p · c.
            
            # From p · (a-b) = 0 and p · (b-c) = 0:
            # p is orthogonal to (a-b) and (b-c), so p is parallel to (a-b) × (b-c).
            cross1 = np.cross(a - b, b - c)
            if np.linalg.norm(cross1) < 1e-15:
                # Degenerate, use centroid
                p = (a + b + c) / 3.0
                p /= np.linalg.norm(p)
            else:
                p = cross1 / np.linalg.norm(cross1)
            
            # p might point in either direction; we want the one on the same side
            # as the triangle interior. Check dot with centroid.
            centroid = (a + b + c) / 3.0
            if np.dot(p, centroid) < 0:
                p = -p
            
            # The angular distance from p to any vertex is the circumradius
            r_ang = np.arccos(np.clip(np.dot(p, a), -1.0, 1.0))
            max_r = max(max_r, r_ang)
        
        return max_r
    except Exception:
        return np.pi  # Fallback: worst case


def _compute_covering_radius_fast(pts):
    """Faster covering radius computation using vectorized operations."""
    try:
        hull = ConvexHull(pts)
        simplices = hull.simplices
        max_r = 0.0
        for simplex in simplices:
            a = pts[simplex[0]]
            b = pts[simplex[1]]
            c = pts[simplex[2]]
            
            cross1 = np.cross(a - b, b - c)
            norm_c = np.linalg.norm(cross1)
            if norm_c < 1e-15:
                p = (a + b + c) / 3.0
                pn = np.linalg.norm(p)
                if pn < 1e-15:
                    continue
                p /= pn
            else:
                p = cross1 / norm_c
            
            centroid = (a + b + c) / 3.0
            if np.dot(p, centroid) < 0:
                p = -p
            
            r_ang = np.arccos(np.clip(np.dot(p, a), -1.0, 1.0))
            if r_ang > max_r:
                max_r = r_ang
        
        return max_r
    except Exception:
        return np.pi


def _refine_repulsive(pts, start_time, deadline, max_iterations=5000):
    """Refine points using repulsive forces to minimize covering radius."""
    best_pts = pts.copy()
    best_cr = _compute_covering_radius_fast(pts)
    
    # Adaptive step size
    step_size = 0.01
    no_improve_count = 0
    max_no_improve = 200
    
    for it in range(max_iterations):
        if time.time() > deadline:
            break
        
        # Compute repulsive forces: push points away from each other
        # Force on point i is sum over j of (pi - pj) / |pi - pj|^2, but only for close pairs
        forces = np.zeros_like(pts)
        for i in range(N):
            for j in range(i + 1, N):
                d = pts[i] - pts[j]
                dist = np.linalg.norm(d)
                if dist < 0.5:  # Only interact if close
                    # Repulsive force
                    f = d / (dist * dist + 1e-10)
                    forces[i] += f
                    forces[j] -= f
        
        # Also add a small random perturbation to escape local minima
        noise = np.random.normal(0, 0.001, pts.shape)
        
        # Update positions
        new_pts = pts + step_size * (forces + noise)
        
        # Renormalize to unit sphere
        norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
        new_pts = new_pts / norms
        
        # Compute new covering radius
        new_cr = _compute_covering_radius_fast(new_pts)
        
        if new_cr < best_cr:
            best_cr = new_cr
            best_pts = new_pts.copy()
            pts = new_pts
            no_improve_count = 0
            # Slowly increase step size if improving
            step_size = min(step_size * 1.01, 0.05)
        else:
            no_improve_count += 1
            # Decrease step size when not improving
            step_size = max(step_size * 0.95, 0.001)
            
            if no_improve_count > max_no_improve:
                # Reset to best and try different step size
                pts = best_pts.copy()
                no_improve_count = 0
                step_size = 0.01
    
    return best_pts, best_cr


def _refine_gradient(pts, start_time, deadline, max_iterations=3000):
    """Refine using gradient descent on covering radius via facet circumradii."""
    best_pts = pts.copy()
    best_cr = _compute_covering_radius_fast(pts)
    
    step_size = 0.005
    no_improve_count = 0
    max_no_improve = 100
    
    for it in range(max_iterations):
        if time.time() > deadline:
            break
        
        # Compute gradient of covering radius w.r.t. each point
        # The covering radius is max over facets of circumradius.
        # For each facet, the circumcenter depends on the three vertices.
        # Gradient of circumradius w.r.t. each vertex points in a direction
        # that would decrease the circumradius.
        
        # Simplified: for each facet, compute the direction that moves
        # each vertex to reduce the circumradius, then aggregate.
        
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        grad = np.zeros_like(pts)
        for simplex in hull.simplices:
            a = pts[simplex[0]]
            b = pts[simplex[1]]
            c = pts[simplex[2]]
            
            cross1 = np.cross(a - b, b - c)
            norm_c = np.linalg.norm(cross1)
            if norm_c < 1e-15:
                continue
            p = cross1 / norm_c
            centroid = (a + b + c) / 3.0
            if np.dot(p, centroid) < 0:
                p = -p
            
            # p is the circumcenter. To reduce circumradius, move vertices
            # toward p (projected onto tangent plane).
            # Gradient direction for each vertex is (p - vertex) projected to tangent plane
            for idx, v in zip(simplex, [a, b, c]):
                # Tangent projection of (p - v)
                tang = p - v
                tang = tang - np.dot(tang, v) * v  # Project to tangent plane at v
                grad[idx] += tang
        
        # Normalize gradient
        grad_norms = np.linalg.norm(grad, axis=1, keepdims=True)
        grad_norms[grad_norms < 1e-15] = 1.0
        grad = grad / grad_norms
        
        # Update
        new_pts = pts - step_size * grad
        norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
        new_pts = new_pts / norms
        
        new_cr = _compute_covering_radius_fast(new_pts)
        
        if new_cr < best_cr:
            best_cr = new_cr
            best_pts = new_pts.copy()
            pts = new_pts
            no_improve_count = 0
            step_size = min(step_size * 1.02, 0.02)
        else:
            no_improve_count += 1
            step_size = max(step_size * 0.9, 0.0005)
            
            if no_improve_count > max_no_improve:
                pts = best_pts.copy()
                no_improve_count = 0
                step_size = 0.005
    
    return best_pts, best_cr


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Phase 1: Repulsive force refinement
    if time.time() < deadline - 30.0:
        pts, cr1 = _refine_repulsive(pts, start_time, deadline - 30.0)
    
    # Phase 2: Gradient refinement
    if time.time() < deadline - 10.0:
        pts, cr2 = _refine_gradient(pts, start_time, deadline - 10.0)
    
    # Final check
    final_cr = _compute_covering_radius_fast(pts)
    
    # Ensure valid: normalize again
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    pts = pts / norms
    
    return pts
# EVOLVE-BLOCK-END
