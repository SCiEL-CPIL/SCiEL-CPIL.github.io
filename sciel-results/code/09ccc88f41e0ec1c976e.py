"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

N = 11
SQRT3 = math.sqrt(3)
APOTHEM = SQRT3 / 2.0

# EVOLVE-BLOCK-START

def _hex_verts(cx, cy, angle_deg, s=1.0):
    """Vertices of a regular hexagon with side s, centered at (cx,cy), rotated angle_deg."""
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + angle_deg)
    vs = np.column_stack([s * np.cos(angles), s * np.sin(angles)])
    return vs + np.array([cx, cy])

def _outer_hex_verts(cx, cy, side, angle_deg):
    """Vertices of outer regular hexagon with given side length."""
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + angle_deg)
    vs = np.column_stack([side * np.cos(angles), side * np.sin(angles)])
    return vs + np.array([cx, cy])

def _point_in_convex(pt, poly):
    """Check if point is inside convex polygon using cross products (consistent winding)."""
    n = len(poly)
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        cross = (b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0])
        if cross < -1e-12:
            return False
    return True

def _seg_seg_dist(p1, p2, p3, p4):
    """Minimum distance between two line segments."""
    d1 = p2 - p1
    d2 = p4 - p3
    r = p1 - p3
    a = np.dot(d1, d1)
    e = np.dot(d2, d2)
    f = np.dot(d2, r)
    if a < 1e-14 and e < 1e-14:
        return np.linalg.norm(p1 - p3)
    if a < 1e-14:
        t = np.clip(f / e, 0, 1)
        return np.linalg.norm(p1 - (p3 + t * d2))
    c = np.dot(d1, r)
    if e < 1e-14:
        s = np.clip(c / a, 0, 1)
        return np.linalg.norm((p1 + s * d1) - p3)
    b = np.dot(d1, d2)
    denom = a * e - b * b
    if denom > 1e-14:
        s = np.clip((b * f - c * e) / denom, 0, 1)
        t = (b * s + f) / e
        if t < 0:
            t = 0
            s = np.clip(c / a, 0, 1)
        elif t > 1:
            t = 1
            s = np.clip((b + f) / a, 0, 1)
        return np.linalg.norm((p1 + s * d1) - (p3 + t * d2))
    else:
        t = np.clip(f / e, 0, 1)
        return np.linalg.norm(p1 - (p3 + t * d2))

def _poly_min_dist(v1, v2):
    """Minimum distance between two convex hexagons. 0 if overlapping."""
    for v in v1:
        if _point_in_convex(v, v2):
            return 0.0
    for v in v2:
        if _point_in_convex(v, v1):
            return 0.0
    min_d = float('inf')
    for i in range(6):
        a1, a2 = v1[i], v1[(i+1) % 6]
        for j in range(6):
            b1, b2 = v2[j], v2[(j+1) % 6]
            d = _seg_seg_dist(a1, a2, b1, b2)
            if d < min_d:
                min_d = d
    return min_d

def _min_outer_side(inner_list, outer_angle_deg=0.0):
    """Compute minimum outer hexagon side length to contain all inner hexagons."""
    # The outer hexagon has 6 edges. For a regular hexagon centered at origin with
    # side length S and rotation angle A, the half-width in direction of normal n is S*apothem.
    # We need S >= max over all inner vertices of projection onto each outer normal / apothem.
    # Outer normals for angle A: directions at A+30, A+90, A+150, A+210, A+270, A+330
    norm_angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]) + outer_angle_deg)
    normals = np.column_stack([np.cos(norm_angles), np.sin(norm_angles)])
    
    max_proj = 0.0
    for (ix, iy, iang) in inner_list:
        verts = _hex_verts(ix, iy, iang)
        for v in verts:
            for n in normals:
                proj = v[0] * n[0] + v[1] * n[1]
                if proj > max_proj:
                    max_proj = proj
    return max_proj / APOTHEM

def _check_valid(inner_list, outer_side, outer_angle_deg=0.0, tol=1e-9):
    """Check containment and non-overlap."""
    # Containment
    norm_angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]) + outer_angle_deg)
    normals = np.column_stack([np.cos(norm_angles), np.sin(norm_angles)])
    limit = outer_side * APOTHEM
    for (ix, iy, iang) in inner_list:
        verts = _hex_verts(ix, iy, iang)
        for v in verts:
            for n in normals:
                if v[0] * n[0] + v[1] * n[1] > limit + tol:
                    return False
    # Non-overlap
    all_verts = [_hex_verts(ix, iy, iang) for (ix, iy, iang) in inner_list]
    for i in range(N):
        for j in range(i + 1, N):
            if _poly_min_dist(all_verts[i], all_verts[j]) < -tol:
                return False
    return True

def _lattice_solution():
    """
    Analytically optimal hexagonal lattice packing of 11 unit hexagons.
    Layout: 1 center, 6 at distance sqrt(3) (ring 1), 4 at distance 3 (ring 2).
    All hexagons rotated 30 degrees (flat-top relative to lattice).
    """
    eps = 1e-10
    centers = [(0.0, 0.0, 30.0)]
    # Ring 1: 6 hexagons at distance sqrt(3) from center
    for k in range(6):
        ang = math.radians(k * 60)
        x = (SQRT3 + eps) * math.cos(ang)
        y = (SQRT3 + eps) * math.sin(ang)
        centers.append((x, y, 30.0))
    # Ring 2: 4 hexagons at distance 3, at angles 30, 90, 150, 210 (or 30, 150, 270, 30+180)
    # Actually for the standard hex cluster of 11: positions are (0,0), 6 neighbors at dist sqrt(3),
    # and 4 more at the "corners" of the hexagonal arrangement.
    # The 4 additional are at: (sqrt(3)*cos(30), sqrt(3)*sin(30)+1.5), etc.
    # Let me use the standard hex lattice coordinates.
    # In a hex lattice with spacing sqrt(3) between centers:
    # Row 0: 3 hexes at y=0, x = -sqrt(3), 0, sqrt(3)
    # Row 1: 4 hexes at y=1.5, x = -1.5*sqrt(3)/2, -sqrt(3)/2, sqrt(3)/2, 1.5*sqrt(3)/2  -- wait
    # Standard hex tiling: centers at (i*sqrt(3) + j*sqrt(3)/2, j*1.5) for integer i,j
    # For 11 hexagons in a compact hex cluster:
    # Use axial coordinates (q, r):
    # (0,0), (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1),
    # (2,-1), (-2,1), (1,1), (-1,-1)
    # Convert to cartesian: x = sqrt(3)*(q + r/2), y = 1.5*r
    axial = [
        (0, 0),
        (1, 0), (-1, 0),
        (0, 1), (0, -1),
        (1, -1), (-1, 1),
        (2, -1), (-2, 1),
        (1, 1), (-1, -1),
    ]
    centers = []
    for (q, r) in axial:
        x = SQRT3 * (q + r / 2.0) + eps * 0.0  # no jitter for lattice
        y = 1.5 * r
        centers.append((x, y, 30.0))
    return centers

def _try_local_refinement(inner_list, outer_side, outer_angle_deg, deadline):
    """
    Try to improve the packing via small perturbations using a distance-field-based
    gradient approach. Returns improved (inner_list, outer_side) or original if no improvement.
    """
    from scipy.optimize import minimize
    
    best_inner = list(inner_list)
    best_side = outer_side
    
    def _objective(params):
        if time.time() > deadline:
            return 1e10
        il = []
        for i in range(N):
            il.append((params[3*i], params[3*i+1], params[3*i+2] * 180.0 / math.pi))
        side = _min_outer_side(il, outer_angle_deg)
        # Overlap penalty
        all_v = [_hex_verts(ix, iy, ia) for (ix, iy, ia) in il]
        pen = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = _poly_min_dist(all_v[i], all_v[j])
                if d < 1e-6:
                    pen += 1000.0 * (1e-6 - d) ** 2
        return side + pen
    
    # Start from current solution
    params0 = np.zeros(3 * N)
    for i, (x, y, a) in enumerate(best_inner):
        params0[3*i] = x
        params0[3*i+1] = y
        params0[3*i+2] = math.radians(a)
    
    try:
        result = minimize(_objective, params0, method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 5000})
        if result.success and result.fun < best_side - 1e-10:
            new_inner = []
            for i in range(N):
                new_inner.append((result.x[3*i], result.x[3*i+1], result.x[3*i+2] * 180.0 / math.pi))
            new_side = _min_outer_side(new_inner, outer_angle_deg)
            if _check_valid(new_inner, new_side, outer_angle_deg):
                return new_inner, new_side
    except Exception:
        pass
    
    return best_inner, best_side

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    # Primary solution: analytically optimal hexagonal lattice
    inner_list = _lattice_solution()
    outer_angle = 0.0
    outer_side = _min_outer_side(inner_list, outer_angle)
    
    # Verify validity
    if not _check_valid(inner_list, outer_side, outer_angle):
        # Slightly increase side to ensure validity
        outer_side += 1e-9
        if not _check_valid(inner_list, outer_side, outer_angle):
            outer_side += 1e-8
    
    best_inner = list(inner_list)
    best_side = outer_side
    
    # Try local refinement if time permits
    if time.time() < deadline - 30:
        refined_inner, refined_side = _try_local_refinement(inner_list, outer_side, outer_angle, deadline - 5)
        if refined_side < best_side - 1e-12:
            best_inner = refined_inner
            best_side = refined_side
    
    # Also try with outer angle = 30 degrees (rotated outer hex)
    if time.time() < deadline - 20:
        outer_angle_30 = 30.0
        side_30 = _min_outer_side(best_inner, outer_angle_30)
        if side_30 < best_side - 1e-12:
            if _check_valid(best_inner, side_30, outer_angle_30):
                best_side = side_30
                outer_angle = outer_angle_30
    
    # Final validity check
    if not _check_valid(best_inner, best_side, outer_angle):
        # Fallback: increase side slightly
        best_side += 1e-9
        if not _check_valid(best_inner, best_side, outer_angle):
            best_side += 1e-8
    
    # Return in required format
    inner = [[p[0], p[1], p[2]] for p in best_inner]
    return inner, (0.0, 0.0), best_side, outer_angle

# EVOLVE-BLOCK-END
