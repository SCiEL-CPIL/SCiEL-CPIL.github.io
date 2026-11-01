"""Rewritten program for the sphere covering task; Lloyd + greedy polish."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull

_DEADLINE = 150.0
_START_TIME = time.time()


def _elapsed():
    return time.time() - _START_TIME


def _remaining():
    return _DEADLINE - _elapsed()


def _normalize(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return points / norms


def _covering_radius_deg(pts):
    """Compute the covering radius in degrees via convex hull facets."""
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 180.0
    max_r = 0.0
    for simplex in hull.simplices:
        face_pts = pts[simplex]
        # Circumcenter of the spherical facet (for a triangle on unit sphere,
        # the circumcenter in 3D is the normalized sum if the triangle is small,
        # but more accurately we compute the point equidistant from all face vertices
        # that lies on the sphere. For a spherical triangle, the "circumcenter" that
        # maximizes the minimum distance is the point on the sphere equidistant to
        # all three vertices. This is found by intersecting the plane through the
        # three points with the sphere... actually for covering radius we want the
        # point in the facet farthest from all vertices, which for a spherical cap
        # is the circumcenter of the spherical triangle (the point equidistant to
        # all three vertices on the sphere).
        #
        # For a spherical triangle, the circumcenter can be computed as follows:
        # Find the point c on the unit sphere such that |c - v1| = |c - v2| = |c - v3|.
        # This means c · v1 = c · v2 = c · v3.
        # So c is in the null space of (v1 - v2) and (v1 - v3) differences... 
        # Actually, c · v1 = c · v2 => c · (v1 - v2) = 0
        # c · v1 = c · v3 => c · (v1 - v3) = 0
        # So c is perpendicular to (v1-v2) and (v1-v3), meaning c is parallel to
        # (v1-v2) × (v1-v3) = v1 × v2 - v3 × v2 = v1×v2 + v2×v3 + v3×v1? 
        # Let's just compute the cross product of (v2-v1) and (v3-v1).
        v1, v2, v3 = face_pts
        d1 = v2 - v1
        d2 = v3 - v1
        c = np.cross(d1, d2)
        c_norm = np.linalg.norm(c)
        if c_norm < 1e-12:
            # Degenerate, use centroid
            c = (v1 + v2 + v3) / 3.0
            c_norm = np.linalg.norm(c)
        c = c / c_norm
        # Check that c is on the correct side (outside the convex hull origin side)
        # The circumcenter should be the one that is farthest from the origin in the
        # direction of the facet normal. The facet normal for the convex hull is
        # the cross product direction. We need c to be the point on the sphere
        # that is the "apex" of the cap, i.e., in the direction away from the
        # interior. For a convex hull of points on the unit sphere, the facet
        # normal points outward. c should be in the same hemisphere as the
        # outward normal.
        # The outward normal of the facet is c (if we oriented it correctly).
        # Let's check: the dot product of c with the centroid of the face should
        # be positive (c is on the same side as the face from the origin).
        centroid = (v1 + v2 + v3) / 3.0
        if np.dot(c, centroid) < 0:
            c = -c
        # Now compute the angular distance from c to the nearest vertex
        dists = np.arccos(np.clip(np.dot(face_pts, c), -1.0, 1.0))
        max_r = max(max_r, np.max(dists))
    return np.degrees(max_r)


def _lloyd_step(pts, n_iter=200):
    """Perform Lloyd's algorithm iterations for spherical Voronoi equalization."""
    pts = _normalize(pts.copy())
    for it in range(n_iter):
        if _remaining() < 5.0:
            break
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        # Compute Voronoi cell centroids (spherical)
        new_pts = np.zeros_like(pts)
        for i in range(len(pts)):
            # Find facets that include point i
            facet_indices = []
            for f in range(len(hull.simplices)):
                if i in hull.simplices[f]:
                    facet_indices.append(f)
            if len(facet_indices) == 0:
                new_pts[i] = pts[i]
                continue
            # The Voronoi cell of point i is bounded by the facets containing i.
            # The centroid of the spherical Voronoi cell can be approximated by
            # the normalized sum of the circumcenters of the facets containing i.
            # More accurately, we can compute the spherical centroid.
            # Simple approach: use the mean of the facet circumcenters.
            circumcenters = []
            for f in facet_indices:
                simplex = hull.simplices[f]
                face_pts = pts[simplex]
                v1, v2, v3 = face_pts
                d1 = v2 - v1
                d2 = v3 - v1
                c = np.cross(d1, d2)
                c_norm = np.linalg.norm(c)
                if c_norm < 1e-12:
                    c = (v1 + v2 + v3) / 3.0
                    c_norm = np.linalg.norm(c)
                c = c / c_norm
                centroid = (v1 + v2 + v3) / 3.0
                if np.dot(c, centroid) < 0:
                    c = -c
                circumcenters.append(c)
            # Weight by facet area (approximated by solid angle)
            # For simplicity, use unweighted mean
            mean_c = np.mean(circumcenters, axis=0)
            mean_norm = np.linalg.norm(mean_c)
            if mean_norm > 1e-12:
                new_pts[i] = mean_c / mean_norm
            else:
                new_pts[i] = pts[i]
        new_pts = _normalize(new_pts)
        # Check for convergence
        if np.max(np.linalg.norm(new_pts - pts, axis=1)) < 1e-8:
            pts = new_pts
            break
        pts = new_pts
    return pts


def _greedy_polish(pts, max_iters=5000):
    """Greedy polish: move worst facet's vertices toward circumcenter."""
    pts = _normalize(pts.copy())
    best_r = _covering_radius_deg(pts)
    for it in range(max_iters):
        if _remaining() < 3.0:
            break
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        # Find the worst facet
        worst_facet = -1
        worst_r = 0.0
        worst_c = None
        for f in range(len(hull.simplices)):
            simplex = hull.simplices[f]
            face_pts = pts[simplex]
            v1, v2, v3 = face_pts
            d1 = v2 - v1
            d2 = v3 - v1
            c = np.cross(d1, d2)
            c_norm = np.linalg.norm(c)
            if c_norm < 1e-12:
                continue
            c = c / c_norm
            centroid = (v1 + v2 + v3) / 3.0
            if np.dot(c, centroid) < 0:
                c = -c
            dists = np.arccos(np.clip(np.dot(face_pts, c), -1.0, 1.0))
            r = np.max(dists)
            if r > worst_r:
                worst_r = r
                worst_facet = f
                worst_c = c
        if worst_facet < 0:
            break
        # Move the vertices of the worst facet toward the circumcenter
        simplex = hull.simplices[worst_facet]
        step = 0.1
        for idx in simplex:
            direction = worst_c - pts[idx]
            dir_norm = np.linalg.norm(direction)
            if dir_norm > 1e-12:
                pts[idx] = pts[idx] + step * direction / dir_norm
        pts = _normalize(pts)
        new_r = _covering_radius_deg(pts)
        if new_r < best_r:
            best_r = new_r
        else:
            # Revert and try smaller step
            for idx in simplex:
                direction = worst_c - pts[idx]
                dir_norm = np.linalg.norm(direction)
                if dir_norm > 1e-12:
                    pts[idx] = pts[idx] - step * direction / dir_norm
            pts = _normalize(pts)
            break
    return pts, best_r


def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = _normalize(pts)
    
    best_pts = pts.copy()
    best_r = _covering_radius_deg(pts)
    
    # Phase 1: Lloyd's algorithm
    if _remaining() > 10.0:
        pts_lloyd = _lloyd_step(pts, n_iter=100)
        r_lloyd = _covering_radius_deg(pts_lloyd)
        if r_lloyd < best_r:
            best_r = r_lloyd
            best_pts = pts_lloyd.copy()
    
    # Phase 2: Greedy polish
    if _remaining() > 5.0:
        pts_polished, r_polished = _greedy_polish(best_pts, max_iters=3000)
        if r_polished < best_r:
            best_r = r_polished
            best_pts = pts_polished.copy()
    
    # Phase 3: More Lloyd if time remains
    if _remaining() > 10.0:
        pts_lloyd2 = _lloyd_step(best_pts, n_iter=100)
        r_lloyd2 = _covering_radius_deg(pts_lloyd2)
        if r_lloyd2 < best_r:
            best_r = r_lloyd2
            best_pts = pts_lloyd2.copy()
        # Polish again
        if _remaining() > 5.0:
            pts_polished2, r_polished2 = _greedy_polish(best_pts, max_iters=2000)
            if r_polished2 < best_r:
                best_r = r_polished2
                best_pts = pts_polished2.copy()
    
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
