"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms

def _covering_radius(pts):
    """Compute the covering radius (in radians) using convex hull facets."""
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Fallback: compute max distance to nearest point on a fine grid
        # This should rarely happen
        theta = np.linspace(0, np.pi, 180)
        phi = np.linspace(0, 2 * np.pi, 360)
        TH, PH = np.meshgrid(theta, phi)
        Y = np.column_stack([
            np.sin(TH) * np.cos(PH).ravel(),
            np.sin(TH) * np.sin(PH).ravel(),
            np.cos(TH).ravel()
        ])
        # Compute distances
        dists = np.min(1.0 - Y @ pts.T, axis=1)
        return np.arccos(np.clip(dists.max(), -1.0, 1.0))
    
    max_r = 0.0
    for simplex in hull.simplices:
        face_pts = pts[simplex]
        # Compute circumcenter of the face
        # For a triangle, circumcenter is intersection of perpendicular bisectors
        a, b, c = face_pts[0], face_pts[1], face_pts[2]
        
        # Solve for circumcenter using linear algebra
        # |x|^2 = |x-a|^2 => 2 x·a = |a|^2 = 1 (since on unit sphere)
        # So 2 x·a = 1, 2 x·b = 1, 2 x·c = 1
        # x·(a-b) = 0, x·(a-c) = 0
        # And x·a = 0.5
        
        A = np.array([a, b, c])  # 3x3
        rhs = 0.5 * np.ones(3)
        try:
            center = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            continue
        
        # Check if circumcenter is inside the triangle (empty cap check)
        # The circumcenter should be on the same side of the face as the origin
        # For a valid facet, the circumcenter (scaled to unit sphere) should be
        # in the direction of the face normal
        
        # Face normal
        n = np.cross(b - a, c - a)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            continue
        n = n / n_norm
        
        # The circumcenter should be in the direction of n (outward)
        # Check: center · n > 0
        if np.dot(center, n) <= 0:
            # The circumcenter is on the wrong side; the cap is not empty
            # In this case, the worst point is the projection of the origin onto the face
            # Actually, for a non-acute triangle, the circumcenter is outside
            # The farthest point in the cap would be one of the vertices
            # But we need to check if the cap is empty
            # If circumcenter is outside the triangle, the max distance in the cap
            # is to one of the vertices, but those are already covered by other facets
            # Actually, we should check: is the circumcenter inside the triangle?
            # For the circumcenter to be a valid worst point, it must be inside the triangle
            # AND the cap must be empty (no other points inside the cap)
            
            # Check if circumcenter is inside the triangle
            # Using barycentric coordinates
            v0 = c - a
            v1 = b - a
            v2 = center - a
            
            d00 = np.dot(v0, v0)
            d01 = np.dot(v0, v1)
            d02 = np.dot(v0, v2)
            d11 = np.dot(v1, v1)
            d12 = np.dot(v1, v2)
            
            denom = d00 * d11 - d01 * d01
            if abs(denom) < 1e-12:
                continue
            
            v = (d11 * d02 - d01 * d12) / denom
            w = (d00 * d12 - d01 * d02) / denom
            
            if v < -1e-9 or w < -1e-9 or v + w > 1 + 1e-9:
                # Circumcenter is outside the triangle
                # The worst point in this cap is one of the vertices
                # But vertices are shared with other facets, so this is already accounted for
                # We can skip this facet
                continue
        
        # Check if the cap is empty: no other points should be inside the cap
        # The cap is centered at center/|center| with angular radius arccos(center·center/|center|^2)
        # Actually, the cap radius is arccos(1/|center|) since center·a = 0.5 and |a|=1
        # So cos(theta) = 0.5/|center|
        cos_r = 0.5 / np.linalg.norm(center)
        if cos_r > 1.0 or cos_r < -1.0:
            continue
        
        # Check all other points
        center_unit = center / np.linalg.norm(center)
        other_dists = pts @ center_unit
        # Points inside the cap have dist > cos_r
        inside = other_dists > cos_r + 1e-9
        if np.any(inside):
            # Cap is not empty, skip
            continue
        
        # This is a valid worst point
        r = np.arccos(np.clip(cos_r, -1.0, 1.0))
        max_r = max(max_r, r)
    
    return max_r

def _find_worst_point(pts):
    """Find the worst point on the sphere and the facet it belongs to."""
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return None, None, None
    
    best_r = 0.0
    best_point = None
    best_facet = None
    
    for simplex in hull.simplices:
        face_pts = pts[simplex]
        a, b, c = face_pts[0], face_pts[1], face_pts[2]
        
        A = np.array([a, b, c])
        rhs = 0.5 * np.ones(3)
        try:
            center = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            continue
        
        n = np.cross(b - a, c - a)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            continue
        n = n / n_norm
        
        if np.dot(center, n) <= 0:
            continue
        
        # Check if circumcenter is inside the triangle
        v0 = c - a
        v1 = b - a
        v2 = center - a
        
        d00 = np.dot(v0, v0)
        d01 = np.dot(v0, v1)
        d02 = np.dot(v0, v2)
        d11 = np.dot(v1, v1)
        d12 = np.dot(v1, v2)
        
        denom = d00 * d11 - d01 * d01
        if abs(denom) < 1e-12:
            continue
        
        v = (d11 * d02 - d01 * d12) / denom
        w = (d00 * d12 - d01 * d02) / denom
        
        if v < -1e-9 or w < -1e-9 or v + w > 1 + 1e-9:
            continue
        
        center_unit = center / np.linalg.norm(center)
        cos_r = 0.5 / np.linalg.norm(center)
        if cos_r > 1.0 or cos_r < -1.0:
            continue
        
        other_dists = pts @ center_unit
        inside = other_dists > cos_r + 1e-9
        if np.any(inside):
            continue
        
        r = np.arccos(np.clip(cos_r, -1.0, 1.0))
        if r > best_r:
            best_r = r
            best_point = center_unit
            best_facet = simplex
    
    return best_point, best_facet, best_r

def _gradient_step(pts, worst_point, worst_facet, step_size):
    """Move points in the worst facet toward the worst point."""
    pts = pts.copy()
    if worst_facet is None or worst_point is None:
        return pts
    
    # Move each point in the worst facet toward the worst point
    for idx in worst_facet:
        pts[idx] += step_size * worst_point
        pts[idx] = pts[idx] / np.linalg.norm(pts[idx])
    
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    best_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    best_r = _covering_radius(best_pts)
    
    # Multi-start with different random initializations
    np.random.seed(42)
    for restart in range(20):
        if time.time() > deadline:
            break
        
        if restart == 0:
            pts = best_pts.copy()
        else:
            # Random start
            pts = np.random.randn(N, 3)
            pts = _normalize(pts)
        
        # Local optimization
        step_size = 0.05
        for it in range(500):
            if time.time() > deadline:
                break
            
            worst_point, worst_facet, r = _find_worst_point(pts)
            if worst_point is None:
                break
            
            # Take a step
            new_pts = _gradient_step(pts, worst_point, worst_facet, step_size)
            new_r = _covering_radius(new_pts)
            
            if new_r < r:
                pts = new_pts
                r = new_r
                # Decrease step size slowly
                step_size *= 0.995
            else:
                # Halve step size
                step_size *= 0.5
                if step_size < 1e-6:
                    break
        
        if r < best_r:
            best_r = r
            best_pts = pts.copy()
    
    # Final polish with very small steps
    if time.time() < deadline - 5.0:
        pts = best_pts.copy()
        step_size = 0.01
        for it in range(200):
            if time.time() > deadline:
                break
            
            worst_point, worst_facet, r = _find_worst_point(pts)
            if worst_point is None:
                break
            
            new_pts = _gradient_step(pts, worst_point, worst_facet, step_size)
            new_r = _covering_radius(new_pts)
            
            if new_r < r:
                pts = new_pts
                r = new_r
                step_size *= 0.99
            else:
                step_size *= 0.5
                if step_size < 1e-8:
                    break
        
        if r < best_r:
            best_r = r
            best_pts = pts.copy()
    
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
