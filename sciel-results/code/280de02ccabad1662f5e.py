"""Packing 11 unit hexagons in a regular hexagon; minimize outer side length."""
import math
import time
import numpy as np
from scipy.optimize import linprog, minimize_scalar

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 115.0  # safety margin within 150s limit

def _compute_all_vertices(inner_hexes):
    """Precompute all vertices of inner hexagons as a numpy array (N*6, 2)."""
    n_inner = len(inner_hexes)
    verts = np.zeros((n_inner * 6, 2))
    for idx, (ix, iy, iangle) in enumerate(inner_hexes):
        angle_rad = math.radians(iangle)
        for k in range(6):
            a = angle_rad + math.pi * k / 3
            verts[idx * 6 + k, 0] = ix + math.cos(a)
            verts[idx * 6 + k, 1] = iy + math.sin(a)
    return verts

def _min_enclosing_hex_side_lp(inner_hexes, angle_deg, all_vertices=None):
    """For a fixed outer hexagon angle, find optimal center and side length using LP."""
    if all_vertices is None:
        all_vertices = _compute_all_vertices(inner_hexes)
    
    n_pts = all_vertices.shape[0]
    if n_pts == 0:
        return 0.0, 0.0, 0.0, angle_deg
    
    # For each of 6 directions, compute normals
    angles = np.array([math.radians(angle_deg + 30 + 60 * k) for k in range(6)])
    normals = np.column_stack([np.cos(angles), np.sin(angles)])  # (6, 2)
    
    # Vectorized constraint construction
    rhs = -(all_vertices @ normals.T)  # (n_pts, 6)
    b_ub = rhs.T.flatten()  # (6 * n_pts,)
    
    A_ub_parts = np.column_stack([
        np.repeat(-normals[:, 0], n_pts),
        np.repeat(-normals[:, 1], n_pts),
        np.full(6 * n_pts, -1.0)
    ])
    
    # Add r >= 0 constraint
    A_ub = np.vstack([A_ub_parts, [[0.0, 0.0, -1.0]]])
    b_ub = np.append(b_ub, 0.0)
    
    c = np.array([0.0, 0.0, 1.0])  # minimize r
    
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, method='highs')
    
    if result.success:
        cx_opt = result.x[0]
        cy_opt = result.x[1]
        r_opt = result.x[2]
    else:
        cx_mean = all_vertices[:, 0].mean()
        cy_mean = all_vertices[:, 1].mean()
        centered = all_vertices - np.array([cx_mean, cy_mean])
        projs = centered @ normals.T
        r_opt = projs.max()
        if r_opt < 0:
            r_opt = 0.0
        cx_opt = cx_mean
        cy_opt = cy_mean
    
    side = r_opt / (math.sqrt(3) / 2)
    return cx_opt, cy_opt, side, angle_deg

def _min_enclosing_hex_side(inner_hexes):
    """Find minimum side length of a regular hexagon that contains all inner hexagons."""
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    all_vertices = _compute_all_vertices(inner_hexes)
    
    # Coarse sweep
    angles_coarse = np.arange(0, 60, 2.0)
    for angle_deg in angles_coarse:
        cx, cy, side, ang = _min_enclosing_hex_side_lp(inner_hexes, float(angle_deg), all_vertices)
        if side < best_side:
            best_side = side
            best_center = (cx, cy)
            best_angle = float(angle_deg)
    
    # Fine sweep
    if best_side < float('inf'):
        angles_fine = np.arange(max(0.0, best_angle - 5), min(60.0, best_angle + 5), 0.25)
        for angle_deg in angles_fine:
            cx, cy, side, ang = _min_enclosing_hex_side_lp(inner_hexes, float(angle_deg), all_vertices)
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_angle = float(angle_deg)
    
    # Even finer sweep
    if best_side < float('inf') and time.time() < _DEADLINE - 20.0:
        angles_finer = np.arange(max(0.0, best_angle - 1.0), min(60.0, best_angle + 1.0), 0.05)
        for angle_deg in angles_finer:
            cx, cy, side, ang = _min_enclosing_hex_side_lp(inner_hexes, float(angle_deg), all_vertices)
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_angle = float(angle_deg)
    
    # Very fine sweep
    if best_side < float('inf') and time.time() < _DEADLINE - 15.0:
        angles_very_fine = np.arange(max(0.0, best_angle - 0.5), min(60.0, best_angle + 0.5), 0.01)
        for angle_deg in angles_very_fine:
            cx, cy, side, ang = _min_enclosing_hex_side_lp(inner_hexes, float(angle_deg), all_vertices)
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_angle = float(angle_deg)
    
    # Bounded scalar minimization polish
    if best_side < float('inf') and time.time() < _DEADLINE - 10.0:
        def objective_angle(ang_p):
            ang_p = ang_p % 60.0
            _, _, side_p, _ = _min_enclosing_hex_side_lp(inner_hexes, float(ang_p), all_vertices)
            return side_p
        
        try:
            bounds = (max(0.0, best_angle - 1.0), min(60.0, best_angle + 1.0))
            res = minimize_scalar(objective_angle, bounds=bounds, method='bounded',
                                  options={'xatol': 1e-10, 'maxiter': 200})
            if res.success:
                ang_opt = float(res.x % 60.0)
                cx_v, cy_v, side_v, _ = _min_enclosing_hex_side_lp(inner_hexes, ang_opt, all_vertices)
                if side_v < best_side:
                    best_side = side_v
                    best_center = (cx_v, cy_v)
                    best_angle = ang_opt
        except Exception:
            pass
    
    # Final micro-adjustments
    if best_side < float('inf') and time.time() < _DEADLINE - 5.0:
        for delta in [0.001, -0.001, 0.002, -0.002, 0.005, -0.005, 0.01, -0.01]:
            ang_test = (best_angle + delta) % 60.0
            cx, cy, side, ang = _min_enclosing_hex_side_lp(inner_hexes, float(ang_test), all_vertices)
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_angle = float(ang_test)
    
    return best_center[0], best_center[1], best_side, best_angle

def _hex_vertices(cx, cy, angle_deg):
    """Return the 6 vertices of a unit hexagon centered at (cx, cy) with given rotation."""
    angle_rad = math.radians(angle_deg)
    verts = []
    for k in range(6):
        a = angle_rad + math.pi * k / 3
        verts.append((cx + math.cos(a), cy + math.sin(a)))
    return verts

def _segments_intersect(p1, p2, p3, p4):
    """Check if line segment p1-p2 intersects line segment p3-p4 (excluding endpoints)."""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False

def _hexagons_disjoint(h1, h2, tol=1e-9):
    """Check if two unit hexagons are disjoint (non-overlapping)."""
    v1 = _hex_vertices(h1[0], h1[1], h1[2])
    v2 = _hex_vertices(h2[0], h2[1], h2[2])
    
    # Check if any vertex of h1 is inside h2
    for vx, vy in v1:
        if _point_in_hex(vx, vy, h2[0], h2[1], h2[2], tol):
            return False
    
    # Check if any vertex of h2 is inside h1
    for vx, vy in v2:
        if _point_in_hex(vx, vy, h1[0], h1[1], h1[2], tol):
            return False
    
    # Check edge intersections
    for i in range(6):
        for j in range(6):
            if _segments_intersect(v1[i], v1[(i+1)%6], v2[j], v2[(j+1)%6]):
                return False
    
    return True

def _point_in_hex(px, py, cx, cy, angle_deg, tol=1e-9):
    """Check if point (px, py) is inside the unit hexagon centered at (cx, cy)."""
    dx = px - cx
    dy = py - cy
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    # Rotate point into hexagon's local frame
    lx = dx * cos_a + dy * sin_a
    ly = -dx * sin_a + dy * cos_a
    
    # Unit hexagon with flat top: |lx| <= 1, |ly| <= sqrt(3)/2, |lx| + |ly|/sqrt(3) <= 1
    # Actually for pointy-top (rotated 30 deg), the conditions are different
    # Let's use the standard: for a hexagon with vertices at angles 0, 60, 120, ...
    # The hexagon is defined by 6 half-planes.
    
    # For a unit hexagon with vertices at (cos(theta), sin(theta)) for theta = 0, 60, 120, 180, 240, 300
    # The supporting lines are at distance cos(30) = sqrt(3)/2 from center in 6 directions
    
    # In local frame (before rotation), the hexagon has vertices at:
    # (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
    
    # The 6 half-plane constraints in local frame:
    # 1. lx <= 1
    # 2. 0.5*lx + sqrt(3)/2*ly <= sqrt(3)/2  => lx + sqrt(3)*ly <= sqrt(3)
    # 3. -0.5*lx + sqrt(3)/2*ly <= sqrt(3)/2  => -lx + sqrt(3)*ly <= sqrt(3)
    # 4. -lx <= 1  => lx >= -1
    # 5. -0.5*lx - sqrt(3)/2*ly <= sqrt(3)/2  => -lx - sqrt(3)*ly <= sqrt(3)
    # 6. 0.5*lx - sqrt(3)/2*ly <= sqrt(3)/2  => lx - sqrt(3)*ly <= sqrt(3)
    
    sqrt3 = math.sqrt(3)
    if lx > 1 + tol: return False
    if lx < -1 - tol: return False
    if lx + sqrt3 * ly > sqrt3 + tol: return False
    if -lx + sqrt3 * ly > sqrt3 + tol: return False
    if -lx - sqrt3 * ly > sqrt3 + tol: return False
    if lx - sqrt3 * ly > sqrt3 + tol: return False
    return True

def _all_disjoint(inner_hexes, tol=1e-9):
    """Check if all inner hexagons are pairwise disjoint."""
    n = len(inner_hexes)
    for i in range(n):
        for j in range(i+1, n):
            if not _hexagons_disjoint(inner_hexes[i], inner_hexes[j], tol):
                return False
    return True

def _inner_in_outer(inner_hexes, outer_center, outer_side, outer_angle, tol=1e-9):
    """Check if all inner hexagons are inside the outer hexagon."""
    outer_verts = _hex_vertices(outer_center[0], outer_center[1], outer_angle)
    # For each inner hexagon, check if all its vertices are inside the outer hexagon
    for ix, iy, iangle in inner_hexes:
        inner_verts = _hex_vertices(ix, iy, iangle)
        for vx, vy in inner_verts:
            if not _point_in_hex(vx, vy, outer_center[0], outer_center[1], outer_angle, tol):
                return False
    return True

def _compute_outer_for_positions(positions, angles, scale=1.0):
    """Given positions and angles of inner hexagons, compute the minimum enclosing outer hexagon."""
    inner_hexes = [[positions[i][0], positions[i][1], angles[i]] for i in range(len(positions))]
    cx, cy, side, angle = _min_enclosing_hex_side(inner_hexes)
    return cx, cy, side, angle

def construct():
    global _START_TIME, _DEADLINE
    
    if time.time() > _DEADLINE - 5.0:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    sqrt3 = math.sqrt(3)
    
    best_result = None
    best_side = float('inf')
    
    # Configuration 1: Standard triangular lattice, pointy-top (30 deg), tight packing
    for scale in [1.001, 1.005, 1.01]:
        u = (sqrt3 * scale, 0.0)
        v = (sqrt3 / 2 * scale, 1.5 * scale)
        
        candidates = []
        for i in range(-3, 4):
            for j in range(-3, 4):
                x = i * u[0] + j * v[0]
                y = i * u[1] + j * v[1]
                dist_sq = x * x + y * y
                candidates.append((dist_sq, x, y))
        
        candidates.sort(key=lambda t: t[0])
        
        positions = []
        for _, x, y in candidates:
            positions.append((x, y))
            if len(positions) == N:
                break
        
        inner = [[pos[0], pos[1], 30.0] for pos in positions]
        
        # Verify disjointness
        if not _all_disjoint(inner):
            continue
        
        if time.time() > _DEADLINE - 15.0:
            xs = [h[0] for h in inner]
            ys = [h[1] for h in inner]
            cx = (max(xs) + min(xs)) / 2
            cy = (max(ys) + min(ys)) / 2
            max_dist = 0
            for h in inner:
                d = math.sqrt((h[0] - cx)**2 + (h[1] - cy)**2)
                if d > max_dist:
                    max_dist = d
            side = (max_dist + 1.0) / (math.sqrt(3) / 2) + 0.1
            if side < best_side:
                best_side = side
                best_result = (inner, (cx, cy), side, 0.0)
            continue
        
        cx, cy, side, angle = _min_enclosing_hex_side(inner)
        
        # Verify inner hexagons are inside outer
        if _inner_in_outer(inner, (cx, cy), side, angle):
            if side < best_side:
                best_side = side
                best_result = (inner, (cx, cy), side, angle)
    
    # Configuration 2: Try flat-top (0 deg) inner hexagons
    for scale in [1.001, 1.005]:
        u = (sqrt3 * scale, 0.0)
        v = (sqrt3 / 2 * scale, 1.5 * scale)
        
        candidates = []
        for i in range(-3, 4):
            for j in range(-3, 4):
                x = i * u[0] + j * v[0]
                y = i * u[1] + j * v[1]
                dist_sq = x * x + y * y
                candidates.append((dist_sq, x, y))
        
        candidates.sort(key=lambda t: t[0])
        
        positions = []
        for _, x, y in candidates:
            positions.append((x, y))
            if len(positions) == N:
                break
        
        inner = [[pos[0], pos[1], 0.0] for pos in positions]
        
        if not _all_disjoint(inner):
            continue
        
        if time.time() > _DEADLINE - 15.0:
            xs = [h[0] for h in inner]
            ys = [h[1] for h in inner]
            cx = (max(xs) + min(xs)) / 2
            cy = (max(ys) + min(ys)) / 2
            max_dist = 0
            for h in inner:
                d = math.sqrt((h[0] - cx)**2 + (h[1] - cy)**2)
                if d > max_dist:
                    max_dist = d
            side = (max_dist + 1.0) / (math.sqrt(3) / 2) + 0.1
            if side < best_side:
                best_side = side
                best_result = (inner, (cx, cy), side, 0.0)
            continue
        
        cx, cy, side, angle = _min_enclosing_hex_side(inner)
        
        if _inner_in_outer(inner, (cx, cy), side, angle):
            if side < best_side:
                best_side = side
                best_result = (inner, (cx, cy), side, angle)
    
    # Configuration 3: Try a slightly rotated arrangement
    for rot_deg in [0.0, 5.0, 10.0, 15.0, 30.0]:
        if time.time() > _DEADLINE - 20.0:
            break
        
        rot_rad = math.radians(rot_deg)
        scale = 1.005
        u = (sqrt3 * scale, 0.0)
        v = (sqrt3 / 2 * scale, 1.5 * scale)
        
        candidates = []
        for i in range(-3, 4):
            for j in range(-3, 4):
                x = i * u[0] + j * v[0]
                y = i * u[1] + j * v[1]
                dist_sq = x * x + y * y
                candidates.append((dist_sq, x, y))
        
        candidates.sort(key=lambda t: t[0])
        
        positions = []
        for _, x, y in candidates:
            positions.append((x, y))
            if len(positions) == N:
                break
        
        inner = []
        for (px, py) in positions:
            rx = px * math.cos(rot_rad) - py * math.sin(rot_rad)
            ry = px * math.sin(rot_rad) + py * math.cos(rot_rad)
            inner.append([rx, ry, 30.0 + rot_deg])
        
        if not _all_disjoint(inner):
            continue
        
        if time.time() > _DEADLINE - 15.0:
            xs = [h[0] for h in inner]
            ys = [h[1] for h in inner]
            cx = (max(xs) + min(xs)) / 2
            cy = (max(ys) + min(ys)) / 2
            max_dist = 0
            for h in inner:
                d = math.sqrt((h[0] - cx)**2 + (h[1] - cy)**2)
                if d > max_dist:
                    max_dist = d
            side = (max_dist + 1.0) / (math.sqrt(3) / 2) + 0.1
            if side < best_side:
                best_side = side
                best_result = (inner, (cx, cy), side, 0.0)
            continue
        
        cx, cy, side, angle = _min_enclosing_hex_side(inner)
        
        if _inner_in_outer(inner, (cx, cy), side, angle):
            if side < best_side:
                best_side = side
                best_result = (inner, (cx, cy), side, angle)
    
    # Configuration 4: Try different lattice spacings to find tighter packings
    for scale in [0.99, 0.995, 1.0, 1.002, 1.004, 1.006, 1.008]:
        if time.time() > _DEADLINE - 25.0:
            break
        
        u = (sqrt3 * scale, 0.0)
        v = (sqrt3 / 2 * scale, 1.5 * scale)
        
        candidates = []
        for i in range(-3, 4):
            for j in range(-3, 4):
                x = i * u[0] + j * v[0]
                y = i * u[1] + j * v[1]
                dist_sq = x * x + y * y
                candidates.append((dist_sq, x, y))
        
        candidates.sort(key=lambda t: t[0])
        
        positions = []
        for _, x, y in candidates:
            positions.append((x, y))
            if len(positions) == N:
                break
        
        inner = [[pos[0], pos[1], 30.0] for pos in positions]
        
        if not _all_disjoint(inner):
            continue
        
        cx, cy, side, angle = _min_enclosing_hex_side(inner)
        
        if _inner_in_outer(inner, (cx, cy), side, angle):
            if side < best_side:
                best_side = side
                best_result = (inner, (cx, cy), side, angle)
    
    # Configuration 5: Try non-uniform scaling in x and y
    for sx in [0.99, 1.0, 1.005, 1.01]:
        for sy in [0.99, 1.0, 1.005, 1.01]:
            if time.time() > _DEADLINE - 25.0:
                break
            
            u = (sqrt3 * sx, 0.0)
            v = (sqrt3 / 2 * sx, 1.5 * sy)
            
            candidates = []
            for i in range(-3, 4):
                for j in range(-3, 4):
                    x = i * u[0] + j * v[0]
                    y = i * u[1] + j * v[1]
                    dist_sq = x * x + y * y
                    candidates.append((dist_sq, x, y))
            
            candidates.sort(key=lambda t: t[0])
            
            positions = []
            for _, x, y in candidates:
                positions.append((x, y))
                if len(positions) == N:
                    break
            
            inner = [[pos[0], pos[1], 30.0] for pos in positions]
            
            if not _all_disjoint(inner):
                continue
            
            cx, cy, side, angle = _min_enclosing_hex_side(inner)
            
            if _inner_in_outer(inner, (cx, cy), side, angle):
                if side < best_side:
                    best_side = side
                    best_result = (inner, (cx, cy), side, angle)
    
    if best_result is None:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_result
