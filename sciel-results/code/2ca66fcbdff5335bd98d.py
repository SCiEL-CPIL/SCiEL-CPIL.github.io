import math
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 11
D = math.sqrt(3)
DEADLINE = 145.0

def _hex_vertices(cx, cy, angle_deg, s=1.0):
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + angle_deg)
    return np.column_stack([s * np.cos(angles), s * np.sin(angles)]) + np.array([cx, cy])

def _outer_hex_vertices(side, angle_deg):
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + angle_deg)
    return np.column_stack([side * np.cos(angles), side * np.sin(angles)])

def _point_in_convex(pt, poly):
    """Vectorized point-in-convex-polygon using cross products."""
    n = len(poly)
    signs = np.zeros(n)
    for i in range(n):
        a = poly[i]
        b = poly[(i+1) % n]
        signs[i] = (b[0]-a[0])*(pt[1]-a[1]) - (b[1]-a[1])*(pt[0]-a[0])
    if np.any(signs > 1e-9) and np.any(signs < -1e-9):
        return False
    return True

def _all_points_in_convex(pts, poly):
    """Check if all points are inside convex polygon. Vectorized."""
    n_poly = len(poly)
    ok = np.ones(len(pts), dtype=bool)
    for i in range(n_poly):
        a = poly[i]
        b = poly[(i+1) % n_poly]
        edge = b - a
        cross = edge[0]*(pts[:,1]-a[1]) - edge[1]*(pts[:,0]-a[0])
        if np.any(cross > 1e-9):
            # All must be on same side
            if np.any(cross < -1e-9):
                return False
        elif np.any(cross < -1e-9):
            if np.any(cross > 1e-9):
                return False
    return True

def _seg_seg_dist(p1, p2, p3, p4):
    d1 = p2 - p1
    d2 = p4 - p3
    r = p1 - p3
    a = np.dot(d1, d1)
    e = np.dot(d2, d2)
    f = np.dot(d2, r)
    if a < 1e-12 and e < 1e-12:
        return np.linalg.norm(p1 - p3)
    if a < 1e-12:
        t = np.clip(f/e, 0, 1)
        return np.linalg.norm(p1 - (p3 + t*d2))
    c = np.dot(d1, r)
    if e < 1e-12:
        s = np.clip(c/a, 0, 1)
        return np.linalg.norm((p1 + s*d1) - p3)
    b = np.dot(d1, d2)
    denom = a*e - b*b
    if denom > 1e-12:
        s = (b*f - c*e) / denom
        s = np.clip(s, 0, 1)
        t = (b*s + f) / e
        if t < 0:
            t = 0; s = np.clip(c/a, 0, 1)
        elif t > 1:
            t = 1; s = np.clip((b+f)/a, 0, 1)
        return np.linalg.norm((p1+s*d1) - (p3+t*d2))
    else:
        t = np.clip(f/e, 0, 1)
        return np.linalg.norm(p1 - (p3+t*d2))

def _poly_min_dist(p1, p2):
    """Minimum distance between two convex hexagons. 0 if overlapping."""
    for v in p1:
        if _point_in_convex(v, p2):
            return 0.0
    for v in p2:
        if _point_in_convex(v, p1):
            return 0.0
    md = float('inf')
    n1, n2 = len(p1), len(p2)
    for i in range(n1):
        a1, a2 = p1[i], p1[(i+1)%n1]
        for j in range(n2):
            b1, b2 = p2[j], p2[(j+1)%n2]
            d = _seg_seg_dist(a1, a2, b1, b2)
            if d < md:
                md = d
    return md

def _min_containment_side(inner_params, outer_angle_deg):
    """Binary search for minimum outer side containing all inner hexagons."""
    n = len(inner_params)
    # Collect all vertices of all inner hexagons
    all_verts = []
    for i in range(n):
        x, y, a = inner_params[i]
        all_verts.append(_hex_vertices(x, y, a))
    all_verts = np.vstack(all_verts)
    
    lo, hi = 3.0, 20.0
    for _ in range(40):
        mid = (lo + hi) / 2
        outer_v = _outer_hex_vertices(mid, outer_angle_deg)
        if _all_points_in_convex(all_verts, outer_v):
            hi = mid
        else:
            lo = mid
    return hi

def _overlap_penalty(inner_params, w=50.0):
    n = len(inner_params)
    pen = 0.0
    verts = [_hex_vertices(inner_params[i,0], inner_params[i,1], inner_params[i,2]) for i in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            sep = _poly_min_dist(verts[i], verts[j])
            if sep < 0.05:
                pen += w * (0.05 - sep)**2
    return pen

def _objective(params, outer_angle_deg, n):
    inner_params = params.reshape(n, 3)
    side = _min_containment_side(inner_params, outer_angle_deg)
    pen = _overlap_penalty(inner_params)
    return side + pen

def _make_seed(outer_angle_deg, compress=1.0):
    """Create seed configuration: honeycomb with optional compression."""
    eps = 1e-8
    centers = [(0.0, 0.0)]
    r1 = D * compress
    for k in range(6):
        ang = np.deg2rad(k*60)
        centers.append((r1*np.cos(ang), r1*np.sin(ang)))
    r2 = 3.0 * compress
    for k in range(4):
        ang = np.deg2rad(30 + k*60)
        centers.append((r2*np.cos(ang), r2*np.sin(ang)))
    inner = [[x, y, 30.0] for x, y in centers]
    side = _min_containment_side(np.array(inner), outer_angle_deg)
    return inner, side

def construct():
    start = time.time()
    n = 11
    
    # Try multiple outer angles and compressions to find good seeds
    best_inner = None
    best_side = float('inf')
    
    outer_angles = [30.0, 0.0, 15.0, 45.0, 60.0]
    compressions = [1.0, 0.95, 0.9, 0.85]
    
    for oa in outer_angles:
        for comp in compressions:
            if time.time() - start > DEADLINE - 60:
                break
            try:
                inner, side = _make_seed(oa, comp)
                if side < best_side:
                    best_side = side
                    best_inner = inner
                    best_oa = oa
            except Exception:
                pass
    
    if best_inner is None:
        best_inner, best_side = _make_seed(30.0, 1.0)
        best_oa = 30.0
    
    # Multi-start local optimization
    best_params = np.array(best_inner).flatten()
    
    for restart in range(8):
        if time.time() - start > DEADLINE - 15:
            break
        
        # Perturb initial guess
        if restart == 0:
            init = best_params.copy()
        else:
            init = best_params + np.random.normal(0, 0.3, size=3*n)
            # Keep angles reasonable
            for i in range(n):
                init[i*3+2] = np.mod(init[i*3+2], 60)
        
        try:
            res = minimize(_objective, init, args=(best_oa, n), method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
            if res.fun < best_side and time.time() - start < DEADLINE - 10:
                # Verify
                inner_test = res.x.reshape(n, 3)
                pen = _overlap_penalty(inner_test, w=1000)
                if pen < 1e-4:
                    side_test = _min_containment_side(inner_test, best_oa)
                    if side_test < best_side:
                        best_side = side_test
                        best_params = res.x.copy()
                        best_inner = res.x.reshape(n, 3).tolist()
        except Exception:
            pass
    
    # Final polish with tighter tolerances
    if time.time() - start < DEADLINE - 5:
        try:
            res = minimize(_objective, best_params, args=(best_oa, n), method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True})
            if res.fun < best_side:
                inner_test = res.x.reshape(n, 3)
                pen = _overlap_penalty(inner_test, w=1000)
                if pen < 1e-4:
                    side_test = _min_containment_side(inner_test, best_oa)
                    if side_test < best_side:
                        best_side = side_test
                        best_inner = res.x.reshape(n, 3).tolist()
        except Exception:
            pass
    
    # Final validation: ensure no overlaps and proper containment
    if best_inner:
        inner_arr = np.array(best_inner)
        # Check overlaps
        valid = True
        verts = [_hex_vertices(inner_arr[i,0], inner_arr[i,1], inner_arr[i,2]) for i in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if _poly_min_dist(verts[i], verts[j]) < -1e-6:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            side = _min_containment_side(inner_arr, best_oa)
            if side < best_side:
                best_side = side
        else:
            # Fallback to seed
            best_inner, best_side = _make_seed(best_oa, 1.0)
    
    return best_inner, (0.0, 0.0), best_side, best_oa
