"""Heilbronn convex regions n=14: two nested regular heptagons."""
import time
import numpy as np
from scipy.spatial import ConvexHull

N = 14
DEADLINE = 145.0  # safety margin before 150s kill


def _triangle_area(p, q, r):
    """Area of triangle formed by points p, q, r."""
    return 0.5 * abs((q[0] - p[0]) * (r[1] - p[1]) - (r[0] - p[0]) * (q[1] - p[1]))


def _min_triangle_area(pts):
    """Minimum area over all triples of points."""
    n = len(pts)
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = _triangle_area(pts[i], pts[j], pts[k])
                if a < min_area:
                    min_area = a
    return min_area


def _convex_hull_area(pts):
    """Area of convex hull of points."""
    try:
        hull = ConvexHull(pts)
        return hull.volume  # In 2D, volume = area
    except Exception:
        # Fallback: use shoelace on sorted points (shouldn't happen for our symmetric configs)
        return 0.0


def _score(pts):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area <= 0:
        return 0.0
    min_tri = _min_triangle_area(pts)
    return min_tri / hull_area


def _make_heptagon_pair(phi, r_ratio):
    """
    Create 14 points as union of two regular heptagons.
    Outer heptagon: radius 1, angles starting at 0.
    Inner heptagon: radius r_ratio, angles starting at phi.
    """
    angles_outer = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    angles_inner = angles_outer + phi
    pts_outer = np.column_stack((np.cos(angles_outer), np.sin(angles_outer)))
    pts_inner = r_ratio * np.column_stack((np.cos(angles_inner), np.sin(angles_inner)))
    pts = np.vstack((pts_outer, pts_inner))
    return pts


def construct():
    t0 = time.time()
    
    # Phase 1: Dense grid search over phi and r_ratio
    best_score = -np.inf
    best_pts = None
    
    # Grid over r_ratio in (0.3, 1.0) and phi in [0, 2*pi/7)
    # Due to 7-fold symmetry, phi only needs to be in [0, 2*pi/7)
    phi_range = 2 * np.pi / 7
    n_phi = 200
    n_r = 50
    
    phi_vals = np.linspace(0, phi_range, n_phi, endpoint=False)
    r_vals = np.linspace(0.3, 1.0, n_r)
    
    # Coarse grid
    for r in r_vals:
        for phi in phi_vals:
            if time.time() > DEADLINE:
                break
            pts = _make_heptagon_pair(phi, r)
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts.copy()
    
    if time.time() > DEADLINE:
        return best_pts if best_pts is not None else _make_heptagon_pair(np.pi/7, 0.5)
    
    # Phase 2: Fine grid around the best
    best_phi = phi_vals[np.argmin(np.abs(phi_vals - (best_pts is None and 0 or 0)))]  # placeholder
    # Actually, let's track the best phi and r properly
    # Redo: track best params
    best_phi_param = None
    best_r_param = None
    best_score2 = -np.inf
    
    for r in r_vals:
        for phi in phi_vals:
            if time.time() > DEADLINE:
                break
            pts = _make_heptagon_pair(phi, r)
            s = _score(pts)
            if s > best_score2:
                best_score2 = s
                best_phi_param = phi
                best_r_param = r
                best_pts = pts.copy()
    
    if time.time() > DEADLINE:
        return best_pts
    
    # Phase 3: Golden-section refinement in 2D (small neighborhood)
    if best_phi_param is not None:
        # Local search around best
        phi_c = best_phi_param
        r_c = best_r_param
        # Ensure phi_c is in [0, phi_range)
        phi_c = phi_c % phi_range
        
        # Try a few local refinements
        for _iter in range(30):
            if time.time() > DEADLINE:
                break
            # Try small perturbations
            for dphi in [0.01, -0.01, 0.005, -0.005, 0.02, -0.02]:
                for dr in [0.01, -0.01, 0.005, -0.005]:
                    if time.time() > DEADLINE:
                        break
                    new_phi = (phi_c + dphi) % phi_range
                    new_r = r_c + dr
                    if new_r < 0.2 or new_r > 1.0:
                        continue
                    pts = _make_heptagon_pair(new_phi, new_r)
                    s = _score(pts)
                    if s > best_score2:
                        best_score2 = s
                        best_phi_param = new_phi
                        best_r_param = new_r
                        best_pts = pts.copy()
    
    return best_pts if best_pts is not None else _make_heptagon_pair(np.pi/7, 0.5)
