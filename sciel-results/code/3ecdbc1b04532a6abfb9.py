"""Optimized sphere covering for 130 points."""
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
import time
import sys

N = 130

def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    return pts / norms

def _covering_radius(pts):
    """Compute the covering radius in degrees using convex hull facet circumradii."""
    try:
        hull = ConvexHull(pts)
        max_circumradius = 0.0
        for simplex in hull.simplices:
            tri = pts[simplex]
            # Compute circumradius of the triangle on the sphere
            a = np.linalg.norm(tri[1] - tri[0])
            b = np.linalg.norm(tri[2] - tri[1])
            c = np.linalg.norm(tri[0] - tri[2])
            # Check if triangle is degenerate
            area = 0.5 * np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0]))
            if area < 1e-12:
                continue
            # Circumradius formula: R = abc / (4A) for Euclidean triangle
            # But we need the angular circumradius on the sphere
            # For a spherical triangle, the circumradius (angular) can be computed
            # from the side lengths (angular distances)
            # Side lengths are angular distances
            alpha = np.arccos(np.clip(np.dot(tri[0], tri[1]), -1, 1))
            beta = np.arccos(np.clip(np.dot(tri[1], tri[2]), -1, 1))
            gamma = np.arccos(np.clip(np.dot(tri[2], tri[0]), -1, 1))
            # Spherical circumradius: use formula R = (a*b*c) / (4*sqrt(s*(s-a)*(s-b)*(s-c)))
            # where s = (a+b+c)/2, but this is for the spherical excess...
            # Actually for spherical triangles, the circumradius R satisfies:
            # sin(R) = a*b*c / (4 * A_s) where A_s is the spherical area
            # Or we can use: the circumcenter is the point equidistant from all three vertices
            # Let's compute it directly: find point p such that |p-v1|=|p-v2|=|p-v3|
            # This is the intersection of the perpendicular bisectors on the sphere
            # Simpler: the circumradius is half the maximum angular distance between vertices
            # No, that's not right either.
            
            # Use the formula: for a spherical triangle with sides a,b,c (angular),
            # the circumradius R satisfies:
            # tan(R) = (a*b*c) / (4 * sin(a/2)*sin(b/2)*sin(c/2)) ... no
            # 
            # Let's use: the circumcenter of a spherical triangle can be found,
            # and R is the angular distance from circumcenter to any vertex.
            # 
            # Alternative: use the Euclidean circumradius of the triangle formed by
            # the three points in 3D, then convert to angular.
            # The Euclidean circumradius R_euclid = a_euclid * b_euclid * c_euclid / (4 * A_euclid)
            # where a_euclid etc are the chord lengths, and A_euclid is the Euclidean area.
            # Then the angular circumradius R_angular = arcsin(R_euclid) ... no
            # Actually, if the circumcenter in 3D is at distance d from origin,
            # and the vertices are at unit distance, then the angular radius R satisfies:
            # sin(R) = R_euclid / d ... this is getting complicated.
            
            # Simpler approach: the covering radius for a facet is the angular distance
            # from the facet center (circumcenter on sphere) to the farthest vertex.
            # For a triangle, this is the circumradius of the spherical triangle.
            # 
            # Let me use a different approach: the maximum angular distance from any
            # point on the sphere to the nearest configuration point, restricted to the
            # Voronoi cell of this facet. The worst case is at the circumcenter.
            # 
            # For a spherical triangle with angular sides a, b, c:
            # The circumradius R can be found from:
            # sin(R) = sin(a) * sin(b) * sin(c) / (4 * sin(A) * sin(B) * sin(C)) ... no
            # 
            # Let me just compute it numerically: find the point on the sphere equidistant
            # from all three vertices. This is the intersection of the sphere with the
            # line through the origin and the circumcenter of the Euclidean triangle.
            
            # Euclidean circumcenter of triangle in 3D
            v1, v2, v3 = tri
            # Solve for circumcenter: |c-v1|^2 = |c-v2|^2 = |c-v3|^2
            # This gives 2 linear equations in 3 unknowns, so the circumcenter lies on
            # a line. The circumcenter on the sphere is the point on this line closest
            # to the origin (normalized).
            
            # Actually, the circumcenter of the triangle (in the plane of the triangle)
            # projected onto the sphere gives the spherical circumcenter.
            
            # Let me use a simpler formula. For a spherical triangle, the circumradius R
            # satisfies:
            # R = (1/2) * arccos((cos(a) + cos(b) + cos(c) - 1) / (2 * cos(a/2) * cos(b/2) * cos(c/2)))
            # No, I'm not sure about this formula.
            
            # Let me use a robust numerical approach:
            # The circumcenter is the point p on the unit sphere that maximizes the
            # minimum distance to the three vertices... no, it's the point equidistant.
            # 
            # Solve: p·v1 = p·v2 = p·v3 = cos(R), |p|=1
            # This means p is orthogonal to (v1-v2) and (v2-v3), so p is parallel to
            # (v1-v2) × (v2-v3)
            
            n1 = np.cross(v1 - v2, v1 - v3)
            if np.linalg.norm(n1) < 1e-12:
                continue
            p = n1 / np.linalg.norm(n1)
            # Check if p is on the correct side (same hemisphere as the triangle)
            centroid = (v1 + v2 + v3) / 3.0
            if np.dot(p, centroid) < 0:
                p = -p
            R = np.arccos(np.clip(np.dot(p, v1), -1, 1))
            if R > max_circumradius:
                max_circumradius = R
        return np.degrees(max_circumradius)
    except Exception:
        return 180.0  # Invalid configuration

def _repulsion_energy(pts):
    """Compute a repulsion energy that encourages spreading."""
    n = len(pts)
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.dot(pts[i], pts[j])
            d = np.clip(d, -1, 1)
            # Use a repulsive potential
            energy += 1.0 / (1.0 - d) ** 2
    return energy

def _gradient_descent(pts, max_iter=2000, lr=0.01, deadline=None):
    """Perform gradient descent on the repulsion energy."""
    pts = pts.copy()
    n = len(pts)
    for it in range(max_iter):
        if deadline and time.time() > deadline:
            break
        # Compute gradient
        grad = np.zeros_like(pts)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                d = np.dot(pts[i], pts[j])
                d = np.clip(d, -0.9999, 0.9999)
                # Gradient of 1/(1-d)^2 with respect to pts[i]
                # d/d(pts[i]) [1/(1-d)^2] = 2/(1-d)^3 * d/d(pts[i]) d
                # d/d(pts[i]) d = pts[j]
                grad[i] += 2.0 * pts[j] / (1.0 - d) ** 3
        # Normalize gradient to keep points on sphere
        for i in range(n):
            # Project gradient to tangent space
            grad_tangent = grad[i] - np.dot(grad[i], pts[i]) * pts[i]
            # Update
            pts[i] -= lr * grad_tangent
            # Renormalize
            norm = np.linalg.norm(pts[i])
            if norm > 1e-10:
                pts[i] /= norm
        # Symmetrize to maintain symmetry
        pts = (pts + pts[::-1]) / 2.0
        for i in range(n):
            norm = np.linalg.norm(pts[i])
            if norm > 1e-10:
                pts[i] /= norm
    return pts

def construct():
    """Construct 130 points on the unit sphere minimizing covering radius."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Phase 1: Generate diverse initial configurations
    best_pts = None
    best_radius = 180.0
    
    # Try multiple random initializations
    np.random.seed(42)
    
    for trial in range(50):
        if time.time() > deadline - 60.0:
            break
        
        # Generate random points with some structure
        if trial % 3 == 0:
            # Fibonacci spiral with perturbation
            i = np.arange(N) + 0.5
            z = 1.0 - 2.0 * i / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i + np.random.randn(N) * 0.1
            r = np.sqrt(np.maximum(0, 1.0 - z * z))
            pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        elif trial % 3 == 1:
            # Random points
            pts = np.random.randn(N, 3)
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        else:
            # Symmetric random: generate N/2 random points, add their negatives
            half = N // 2
            pts_half = np.random.randn(half, 3)
            pts_half /= np.linalg.norm(pts_half, axis=1, keepdims=True)
            pts = np.vstack([pts_half, -pts_half])
            if len(pts) < N:
                extra = np.random.randn(N - len(pts), 3)
                extra /= np.linalg.norm(extra, axis=1, keepdims=True)
                pts = np.vstack([pts, extra])
            pts = pts[:N]
        
        pts = _normalize(pts)
        
        # Quick local optimization
        if time.time() > deadline - 30.0:
            break
        
        # Simple gradient descent
        for it in range(500):
            if time.time() > deadline - 30.0:
                break
            grad = np.zeros_like(pts)
            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue
                    d = np.dot(pts[i], pts[j])
                    d = np.clip(d, -0.9999, 0.9999)
                    grad[i] += 2.0 * pts[j] / (1.0 - d) ** 3
            for i in range(N):
                grad_tangent = grad[i] - np.dot(grad[i], pts[i]) * pts[i]
                pts[i] -= 0.005 * grad_tangent
                norm = np.linalg.norm(pts[i])
                if norm > 1e-10:
                    pts[i] /= norm
        
        # Evaluate
        radius = _covering_radius(pts)
        if radius < best_radius:
            best_radius = radius
            best_pts = pts.copy()
    
    # Phase 2: Refine the best configuration
    if best_pts is not None and time.time() < deadline - 10.0:
        # More careful refinement
        pts = best_pts.copy()
        for it in range(3000):
            if time.time() > deadline - 5.0:
                break
            grad = np.zeros_like(pts)
            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue
                    d = np.dot(pts[i], pts[j])
                    d = np.clip(d, -0.9999, 0.9999)
                    grad[i] += 2.0 * pts[j] / (1.0 - d) ** 3
            for i in range(N):
                grad_tangent = grad[i] - np.dot(grad[i], pts[i]) * pts[i]
                pts[i] -= 0.001 * grad_tangent
                norm = np.linalg.norm(pts[i])
                if norm > 1e-10:
                    pts[i] /= norm
        
        radius = _covering_radius(pts)
        if radius < best_radius:
            best_radius = radius
            best_pts = pts.copy()
    
    return best_pts if best_pts is not None else _normalize(np.random.randn(N, 3))
