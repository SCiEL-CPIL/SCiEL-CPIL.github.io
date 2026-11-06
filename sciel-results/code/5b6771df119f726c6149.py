"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
import time
import sys


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms


def _covering_radius(points):
    """Compute covering radius in degrees via convex hull facet circumradii."""
    try:
        hull = ConvexHull(points)
        max_rad = 0.0
        for simplex in hull.simplices:
            # Get the 3 (or 2) points of the facet
            facet_pts = points[simplex]
            if len(facet_pts) == 3:
                a, b, c = facet_pts
                # Circumcenter of triangle on sphere
                # Use spherical geometry: circumradius = angle from circumcenter to vertices
                # Compute circumcenter in 3D
                v1 = b - a
                v2 = c - a
                cross = np.cross(v1, v2)
                if np.linalg.norm(cross) < 1e-12:
                    continue
                # Circumcenter formula for 3 points on unit sphere
                # O = (a + b + c) / 2 projected... use standard formula
                # For 3 points on unit sphere, circumcenter is along normal
                # d = 2 * (a·(b×c))
                d = 2.0 * np.dot(a, np.cross(b, c))
                if abs(d) < 1e-12:
                    continue
                # Circumcenter = (a*(|b|^2|c|^2 - (a·b)^2) + ...) / d
                # Simplified for unit sphere:
                # O = (a*(b·c) + b*(c·a) + c*(a·b) - (a·b)(b·c)a - ...) -- too complex
                # Use: circumradius r satisfies sin(r) = |a×b| / (2*d_circum)
                # Actually, for points on unit sphere, the circumradius (angular) of facet
                # is the angle from the facet's circumcenter to a vertex.
                # The circumcenter of the triangle formed by 3 unit vectors:
                # C = (a + b + c) / |a + b + c| if not degenerate, but that's the centroid direction
                # Correct: circumcenter O satisfies O·a = O·b = O·c and |O|=1
                # O = (a*(b·c) + b*(c·a) + c*(a·b) - (a·b)(b·c)a - (b·c)(c·a)b - (c·a)(a·b)c) / det
                # Simpler: use the formula that the circumradius R satisfies
                # R = (|a-b||b-c||c-a|) / (4 * Area)
                # And angular distance = arcsin(R) for unit sphere? No.
                # For 3 points on unit sphere, the circumradius (angular) is:
                # sin(R) = |a×b| / (2 * |normal|) where normal = a×b×c related
                # Let's use: the circumcenter is O = (a + b + c) / ||a+b+c|| only if equilateral
                # General formula:
                # O = (a*(|b|^2|c|^2 - (a·b)^2) + b*(|c|^2|a|^2 - (b·c)^2) + c*(|a|^2|b|^2 - (c·a)^2)) / D
                # where D = 2*(a·(b×c))
                # Since |a|=|b|=|c|=1:
                # |b|^2|c|^2 - (a·b)^2 = 1 - (a·b)^2
                # So O = (a*(1-(a·b)^2) + b*(1-(b·c)^2) + c*(1-(c·a)^2)) / D
                # Wait, that's not right either. Let me use a robust numerical approach.
                
                # Robust: find circumcenter by solving linear system
                # O·(a-b) = (|a|^2 - |b|^2)/2 = 0
                # O·(b-c) = 0
                # |O| = 1
                # So O is perpendicular to (a-b) and (b-c), i.e., O is parallel to (a-b)×(b-c)
                n = np.cross(a - b, b - c)
                n_norm = np.linalg.norm(n)
                if n_norm < 1e-12:
                    continue
                O = n / n_norm
                # Check which side: ensure O·a >= O·b etc. (should be equal)
                # The circumradius is the angle between O and a
                cos_r = np.clip(np.dot(O, a), -1.0, 1.0)
                rad = np.arccos(cos_r)
                if rad > max_rad:
                    max_rad = rad
            elif len(facet_pts) == 2:
                # Degenerate: edge
                a, b = facet_pts
                # The "circumcenter" is the midpoint direction
                mid = (a + b) / np.linalg.norm(a + b)
                cos_r = np.clip(np.dot(mid, a), -1.0, 1.0)
                rad = np.arccos(cos_r)
                if rad > max_rad:
                    max_rad = rad
        return np.degrees(max_rad)
    except Exception:
        return 180.0


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin: 10s before 150s kill
    
    # Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    points = _normalize(points)
    
    best_points = points.copy()
    best_radius = _covering_radius(points)
    
    # Iterative improvement: find largest facet, move point to its circumcenter
    iteration = 0
    while time.time() < deadline:
        try:
            hull = ConvexHull(points)
            
            # Find the facet with the largest circumradius
            max_rad = 0.0
            best_facet = None
            for simplex in hull.simplices:
                facet_pts = points[simplex]
                if len(facet_pts) == 3:
                    a, b, c = facet_pts
                    n = np.cross(a - b, b - c)
                    n_norm = np.linalg.norm(n)
                    if n_norm < 1e-12:
                        continue
                    O = n / n_norm
                    cos_r = np.clip(np.dot(O, a), -1.0, 1.0)
                    rad = np.arccos(cos_r)
                    if rad > max_rad:
                        max_rad = rad
                        best_facet = simplex
                elif len(facet_pts) == 2:
                    a, b = facet_pts
                    mid = (a + b) / np.linalg.norm(a + b)
                    cos_r = np.clip(np.dot(mid, a), -1.0, 1.0)
                    rad = np.arccos(cos_r)
                    if rad > max_rad:
                        max_rad = rad
                        best_facet = simplex
            
            if best_facet is None:
                break
            
            # Get the circumcenter of the largest facet
            facet_pts = points[best_facet]
            if len(facet_pts) == 3:
                a, b, c = facet_pts
                n = np.cross(a - b, b - c)
                n_norm = np.linalg.norm(n)
                if n_norm < 1e-12:
                    break
                new_point = n / n_norm
            else:
                a, b = facet_pts
                new_point = (a + b) / np.linalg.norm(a + b)
            
            new_point = _normalize(new_point)
            
            # Find the point in the current configuration that is farthest from the new_point
            # and is "worst covered" - we'll remove the point that, when replaced, gives best improvement
            # Simple heuristic: replace the point that is closest to the facet's circumcenter direction
            # but is not part of the facet (or is part and we need to check)
            
            # Actually, a better strategy: replace the point that is farthest from the new_point
            # This ensures we're filling the largest gap
            dists = np.array([np.arccos(np.clip(np.dot(p, new_point), -1, 1)) for p in points])
            replace_idx = np.argmax(dists)
            
            # But we should avoid replacing a point that's part of the facet if possible
            # Let's try: replace the farthest point
            points[replace_idx] = new_point
            points = _normalize(points)
            
            # Evaluate
            current_radius = _covering_radius(points)
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = points.copy()
            else:
                # Revert if worse
                points[replace_idx] = best_points[replace_idx]
                # Try replacing a different point
                # Find second farthest
                dists2 = np.array([np.arccos(np.clip(np.dot(p, new_point), -1, 1)) for p in points])
                # Sort by distance descending
                order = np.argsort(-dists2)
                improved = False
                for idx in order[:5]:
                    if idx == replace_idx:
                        continue
                    orig = points[idx].copy()
                    points[idx] = new_point
                    points = _normalize(points)
                    test_radius = _covering_radius(points)
                    if test_radius < best_radius:
                        best_radius = test_radius
                        best_points = points.copy()
                        improved = True
                        break
                    else:
                        points[idx] = orig
                if not improved:
                    # Try a small perturbation approach
                    # Move the farthest point slightly toward the circumcenter
                    # Or just accept and continue
                    pass
            
            iteration += 1
            
        except Exception:
            break
    
    return best_points
# EVOLVE-BLOCK-END
