import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

N = 12

def hex_vertices(cx, cy, angle_deg, s=1.0):
    angle = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    angles = np.radians(np.array([0, 60, 120, 180, 240, 300]))
    verts = s * np.column_stack([np.cos(angles), np.sin(angles)])
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    verts = verts @ R.T
    verts += np.array([cx, cy])
    return verts

def min_dist_between_hexes(v1, v2):
    """Approximate minimum distance between two hexagons using vertex distances."""
    # Quick check: if any vertex of one is inside the other, they overlap
    # Use SAT for accuracy
    n1, n2 = len(v1), len(v2)
    
    def project(poly, axis):
        dots = poly @ axis
        return dots.min(), dots.max()
    
    def get_axes(poly):
        axes = []
        for i in range(len(poly)):
            edge = poly[(i+1) % len(poly)] - poly[i]
            axis = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(axis)
            if norm > 1e-12:
                axes.append(axis / norm)
        return axes
    
    axes = get_axes(v1) + get_axes(v2)
    
    for axis in axes:
        min1, max1 = project(v1, axis)
        min2, max2 = project(v2, axis)
        if max1 < min2 or max2 < min1:
            return True  # Separated
    return False  # Overlapping

def are_disjoint(inner):
    """Check if all hexagons are pairwise disjoint."""
    verts = [hex_vertices(p[0], p[1], p[2]) for p in inner]
    for i in range(len(verts)):
        for j in range(i+1, len(verts)):
            if not min_dist_between_hexes(verts[i], verts[j]):
                return False
    return True

def point_in_hex(p, s, cx, cy, ang_deg):
    """Check if point p is inside hexagon with side s, center (cx,cy), angle ang_deg."""
    ang = np.radians(ang_deg)
    cos_a, sin_a = np.cos(ang), np.sin(ang)
    px, py = p[0] - cx, p[1] - cy
    lx = px * cos_a + py * sin_a
    ly = -px * sin_a + py * cos_a
    
    # Flat-top hex with "radius" s (center to vertex)
    # Inradius = s * sqrt(3)/2
    # For flat-top: |lx| <= s and |ly| <= s*sqrt(3)/2 and |lx| + |ly|/sqrt(3) <= s
    # Actually let me just use the vertex half-plane method
    
    angles = np.radians(np.array([0, 60, 120, 180, 240, 300]))
    verts = s * np.column_stack([np.cos(angles), np.sin(angles)])
    
    for i in range(6):
        v1 = verts[i]
        v2 = verts[(i+1) % 6]
        edge = v2 - v1
        normal = np.array([edge[1], -edge[0]])
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            continue
        normal /= norm
        val = (np.array([lx, ly]) - v1) @ normal
        if val > 1e-9:
            return False
    return True

def min_outer_side(inner, cx, cy, ang_deg):
    """Find minimum outer hex side length to contain all inner hexes."""
    all_verts = []
    for p in inner:
        all_verts.append(hex_vertices(p[0], p[1], p[2]))
    all_verts = np.vstack(all_verts)
    
    lo, hi = 0.0, 50.0
    for _ in range(50):
        mid = (lo + hi) / 2
        ok = True
        for v in all_verts:
            if not point_in_hex(v, mid, cx, cy, ang_deg):
                ok = False
                break
        if ok:
            hi = mid
        else:
            lo = mid
    return hi

def objective(params):
    deadline = time.time() + 145
    if time.time() > deadline:
        return 1e10
    
    outer_cx = params[-3]
    outer_cy = params[-2]
    outer_angle = params[-1]
    
    inner = [[params[3*i], params[3*i+1], params[3*i+2]] for i in range(N)]
    
    if not are_disjoint(inner):
        return 1e10
    
    if time.time() > deadline:
        return 1e10
    
    return min_outer_side(inner, outer_cx, outer_cy, outer_angle)

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    sqrt3 = np.sqrt(3)
    
    # Good initial config: honeycomb-like 4-4-4
    init_inner = []
    rows_y = [0, sqrt3, 2*sqrt3]
    for row_idx, y in enumerate(rows_y):
        x_start = -1.5 * sqrt3
        if row_idx % 2 == 1:
            x_start += 0.5 * sqrt3
        for k in range(4):
            x = x_start + k * sqrt3
            init_inner.append([x, y, 0.0])
    
    init_params = []
    for p in init_inner:
        init_params.extend([p[0], p[1], p[2]])
    init_params.extend([0.0, 0.0, 0.0])
    
    bounds = []
    for i in range(N):
        bounds.extend([(-15, 15), (-15, 15), (0, 180)])
    bounds.extend([(-15, 15), (-15, 15), (0, 180)])
    
    try:
        result = differential_evolution(
            objective, bounds, seed=42, maxiter=30, tol=1e-3,
            polish=False, init='sobol', workers=1
        )
        best_params = result.x
    except Exception:
        best_params = init_params
    
    if time.time() > deadline:
        best_params = init_params
    
    inner = [[best_params[3*i], best_params[3*i+1], best_params[3*i+2]] for i in range(N)]
    outer_cx, outer_cy, outer_angle = best_params[-3], best_params[-2], best_params[-1]
    
    if are_disjoint(inner):
        s = min_outer_side(inner, outer_cx, outer_cy, outer_angle)
    else:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        s, outer_cx, outer_cy, outer_angle = 40.0, 0.0, 0.0, 0.0
    
    return inner, (outer_cx, outer_cy), s, outer_angle

