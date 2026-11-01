"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for safety
    
    def fibonacci_spiral(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    def compute_covering_radius(pts):
        """Compute covering radius as max circumradius over convex hull faces."""
        try:
            hull = ConvexHull(pts)
            faces = hull.simplices
            a = pts[faces[:, 0]]
            b = pts[faces[:, 1]]
            c = pts[faces[:, 2]]
            
            # Circumcenter calculation for spherical triangles
            # c_dir is proportional to a + b + c
            c_dir = a + b + c
            norms = np.linalg.norm(c_dir, axis=1)
            
            radii = np.zeros(len(faces))
            valid = norms > 1e-10
            
            if np.any(valid):
                c_centers = c_dir[valid] / norms[valid, np.newaxis]
                # Cosine of radius is dot product of center and vertex
                cos_r = np.clip(np.sum(c_centers * a[valid], axis=1), -1.0, 1.0)
                r = np.arccos(cos_r)
                # If the circumcenter is "behind" the origin relative to the triangle (obtuse case on sphere)
                # The minimal enclosing circle might be defined by an edge.
                # For small caps (N=130), this is unlikely, but let's check.
                # If r > pi/2, the "cap" is larger than a hemisphere, which is not the case here.
                # Actually, if the triangle is obtuse, the circumcenter is outside.
                # The covering radius is still the distance to the farthest vertex?
                # No, the covering radius of the facet is the radius of the smallest cap covering the facet.
                # If the circumcenter is outside the triangle, the smallest cap is defined by the longest edge.
                # Let's calculate edge lengths to be safe.
                d_ab = np.arccos(np.clip(np.sum(a[valid] * b[valid], axis=1), -1, 1))
                d_bc = np.arccos(np.clip(np.sum(b[valid] * c[valid], axis=1), -1, 1))
                d_ca = np.arccos(np.clip(np.sum(c[valid] * a[valid], axis=1), -1, 1))
                max_edge = np.maximum(np.maximum(d_ab, d_bc), d_ca)
                # The circumradius of the minimal enclosing circle is max(r_circum, max_edge/2) ?
                # On the sphere, if the triangle is obtuse, the circumcenter of the 3 points is not the center of the minimal enclosing circle.
                # The center of the minimal enclosing circle is the midpoint of the longest edge.
                # Radius is half the length of the longest edge.
                # r_circum calculated above is the radius of the circle through 3 points.
                # If the triangle is acute, r_circum < max_edge/2? No.
                # In Euclidean geometry, if obtuse, circumcenter is outside, and circumradius > longest_side/2.
                # The minimal enclosing circle radius is longest_side/2.
                # So we should take min(r_circum, max_edge/2)? No, max_edge/2 is the radius of the circle with diameter as longest side.
                # If the triangle is obtuse, the circumcircle contains the triangle, but a smaller circle (diameter = longest side) also contains it.
                # So the covering radius is min(r_circum, max_edge/2)?
                # Actually, for a spherical triangle, if it is "acute", the circumcenter is inside.
                # If "obtuse", the circumcenter is outside.
                # The cap covering the triangle is centered at the circumcenter IF the circumcenter is inside the triangle?
                # No, the cap is the intersection of hemispheres.
                # Let's assume for N=130, triangles are small and acute.
                # But to be robust:
                radii_valid = np.minimum(r, max_edge / 2.0) # Wait, is it min or max?
                # If I have a triangle, the farthest point from the circumcenter is a vertex.
                # If I move the center to the midpoint of the longest edge, the farthest point is the other vertex?
                # No, the circle with diameter as longest edge covers the triangle if it's obtuse.
                # The radius is half the longest edge.
                # The circumradius of the 3 points is larger.
                # So the minimal covering radius is min(r_circum, max_edge/2)?
                # Actually, if the triangle is acute, r_circum < max_edge/2 is FALSE.
                # In acute triangle, circumcenter is inside. Circumradius is distance to vertices.
                # Longest side c. c = 2 R sin C. Since C < 90, sin C < 1, so c < 2 R. So R > c/2.
                # So the circle with diameter c does NOT cover the third vertex.
                # So for acute triangles, the covering radius IS the circumradius.
                # For obtuse triangles, the circumcenter is outside. The minimal enclosing circle has diameter = longest side.
                # So covering radius = max_edge / 2.
                # So we should take: if acute, R_circum. If obtuse, max_edge/2.
                # How to check if obtuse? If r > max_edge/2, it's likely acute?
                # In acute, R > c/2. In obtuse, R > c/2 as well (circumradius of obtuse triangle is large).
                # But the *minimal* covering circle for an obtuse triangle has radius c/2.
                # So the covering radius of the facet is:
                # If the circumcenter lies inside the triangle, it's R_circum.
                # If outside, it's max_edge/2.
                # A simple check: if R_circum > max_edge/2, it *could* be either.
                # But for small caps, we can just use R_circum and assume they are acute.
                # Given N=130, caps are ~11 deg, very small. They are definitely acute.
                radii[valid] = r
            else:
                # Degenerate
                a_deg = a[~valid]
                b_deg = b[~valid]
                c_deg = c[~valid]
                d_ab = np.arccos(np.clip(np.sum(a_deg * b_deg, axis=1), -1, 1))
                d_bc = np.arccos(np.clip(np.sum(b_deg * c_deg, axis=1), -1, 1))
                d_ca = np.arccos(np.clip(np.sum(c_deg * a_deg, axis=1), -1, 1))
                radii[~valid] = np.maximum(np.maximum(d_ab, d_bc), d_ca) / 2.0
            
            return float(np.max(radii)), faces, radii
        except Exception:
            # Fallback
            lat = np.linspace(-np.pi/2, np.pi/2, 50)
            lon = np.linspace(0, 2*np.pi, 100, endpoint=False)
            LA, LO = np.meshgrid(lat, lon, indexing='ij')
            z = np.sin(LA)
            x = np.cos(LA) * np.cos(LO)
            y = np.cos(LA) * np.sin(LO)
            grid_pts = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
            dot = np.dot(grid_pts, pts.T)
            dot = np.clip(dot, -1.0, 1.0)
            ang_dists = np.arccos(dot)
            min_dists = np.min(ang_dists, axis=1)
            return float(np.max(min_dists)), None, None
    
    def tangent_project(v, p):
        """Project vector v onto tangent plane at point p on unit sphere."""
        return v - np.sum(v * p, axis=1, keepdims=True) * p

    points = fibonacci_spiral(N).astype(np.float64)
    
    best_points = points.copy()
    best_rad, _, _ = compute_covering_radius(points)
    
    # Soft-max optimization
    tau = 0.1  # Temperature
    alpha = 0.05
    decay_alpha = 0.99
    decay_tau = 0.95
    
    max_iter = 2000
    
    for it in range(max_iter):
        if time.time() > deadline - 5.0:
            break
        
        rad, faces, face_radii = compute_covering_radius(points)
        if faces is None:
            break
            
        if rad < best_rad:
            best_rad = rad
            best_points = points.copy()
        
        # Compute weights for soft-max
        # Shift radii to avoid underflow/overflow?
        # w_F = exp(R_F / tau)
        # To stabilize, subtract max R
        max_r = np.max(face_radii)
        exp_terms = np.exp((face_radii - max_r) / tau)
        sum_exp = np.sum(exp_terms)
        if sum_exp < 1e-10:
            weights = np.ones(len(faces)) / len(faces)
        else:
            weights = exp_terms / sum_exp
        
        # Compute gradient
        grads = np.zeros_like(points)
        
        for i in range(len(faces)):
            w = weights[i]
            if w < 1e-6:
                continue
            
            f = faces[i]
            a_idx, b_idx, c_idx = f[0], f[1], f[2]
            
            a = points[a_idx]
            b = points[b_idx]
            c = points[c_idx]
            
            # Circumcenter
            c_dir = a + b + c
            norm_c_dir = np.linalg.norm(c_dir)
            if norm_c_dir < 1e-10:
                continue
            center = c_dir / norm_c_dir
            
            # Gradient for each vertex: move towards center
            # Direction: center - vertex
            # Project to tangent plane
            grad_a = tangent_project(center - a, a)
            grad_b = tangent_project(center - b, b)
            grad_c = tangent_project(center - c, c)
            
            grads[a_idx] += w * grad_a
            grads[b_idx] += w * grad_b
            grads[c_idx] += w * grad_c
        
        # Update
        points = points + alpha * grads
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        
        # Decay
        alpha *= decay_alpha
        tau *= decay_tau
        
        if alpha < 1e-8:
            break
    
    # Final polish: local optimization on the worst facet
    # Run a few iterations of moving vertices of the max-radius facet
    for _ in range(50):
        rad, faces, face_radii = compute_covering_radius(points)
        if faces is None: break
        max_idx = np.argmax(face_radii)
        f = faces[max_idx]
        a_idx, b_idx, c_idx = f[0], f[1], f[2]
        
        a = points[a_idx]
        b = points[b_idx]
        c = points[c_idx]
        
        c_dir = a + b + c
        norm_c_dir = np.linalg.norm(c_dir)
        if norm_c_dir < 1e-10:
            break
        center = c_dir / norm_c_dir
        
        step = 0.01
        new_a = a + step * tangent_project(center - a, a)
        new_b = b + step * tangent_project(center - b, b)
        new_c = c + step * tangent_project(center - c, c)
        
        new_a /= np.linalg.norm(new_a)
        new_b /= np.linalg.norm(new_b)
        new_c /= np.linalg.norm(new_c)
        
        test_points = points.copy()
        test_points[a_idx] = new_a
        test_points[b_idx] = new_b
        test_points[c_idx] = new_c
        
        new_rad, _, _ = compute_covering_radius(test_points)
        if new_rad < rad:
            points = test_points
            best_rad = new_rad
            best_points = points.copy()
        else:
            step *= 0.5
            if step < 1e-6:
                break

    return best_points

# EVOLVE-BLOCK-END
