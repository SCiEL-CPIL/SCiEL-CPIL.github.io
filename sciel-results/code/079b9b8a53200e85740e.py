"""Packing 11 unit hexagons in a regular hexagon; minimize outer side length."""
import math
import time
import numpy as np
from scipy.optimize import linprog, minimize, minimize_scalar

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0  # safety margin within 150s limit

def _remaining_time():
    return max(0.0, _DEADLINE - time.time())

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

def _min_enclosing_hex_side_lp(all_vertices, angle_deg):
    """For a fixed outer hexagon angle, find optimal center and side length using LP."""
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

def _min_enclosing_hex_side(inner_hexes, all_vertices=None):
    """Find minimum side length of a regular hexagon that contains all inner hexagons."""
    if all_vertices is None:
        all_vertices = _compute_all_vertices(inner_hexes)
    
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Coarse sweep
    angles_coarse = np.arange(0, 60, 3.0)
    for angle_deg in angles_coarse:
        cx, cy, side, ang = _min_enclosing_hex_side_lp(all_vertices, float(angle_deg))
        if side < best_side:
            best_side = side
            best_center = (cx, cy)
            best_angle = float(angle_deg)
    
    # Fine sweep
    if best_side < float('inf') and _remaining_time() > 5.0:
        angles_fine = np.arange(max(0.0, best_angle - 6), min(60.0, best_angle + 6), 0.5)
        for angle_deg in angles_fine:
            cx, cy, side, ang = _min_enclosing_hex_side_lp(all_vertices, float(angle_deg))
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_angle = float(angle_deg)
    
    # Even finer sweep
    if best_side < float('inf') and _remaining_time() > 3.0:
        angles_finer = np.arange(max(0.0, best_angle - 1.5), min(60.0, best_angle + 1.5), 0.05)
        for angle_deg in angles_finer:
            cx, cy, side, ang = _min_enclosing_hex_side_lp(all_vertices, float(angle_deg))
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_angle = float(angle_deg)
    
    # Bounded scalar minimization polish
    if best_side < float('inf') and _remaining_time() > 2.0:
        def objective_angle(ang_p):
            ang_p = ang_p % 60.0
            _, _, side_p, _ = _min_enclosing_hex_side_lp(all_vertices, float(ang_p))
            return side_p
        
        try:
            bounds = (max(0.0, best_angle - 1.5), min(60.0, best_angle + 1.5))
            res = minimize_scalar(objective_angle, bounds=bounds, method='bounded',
                                  options={'xatol': 1e-10, 'maxiter': 100})
            if res.success:
                ang_opt = float(res.x % 60.0)
                cx_v, cy_v, side_v, _ = _min_enclosing_hex_side_lp(all_vertices, ang_opt)
                if side_v < best_side:
                    best_side = side_v
                    best_center = (cx_v, cy_v)
                    best_angle = ang_opt
        except Exception:
            pass
    
    # Final micro-adjustments on angle
    if best_side < float('inf') and _remaining_time() > 1.0:
        for delta in [0.001, -0.001, 0.002, -0.002, 0.005, -0.005]:
            ang_test = (best_angle + delta) % 60.0
            cx, cy, side, ang = _min_enclosing_hex_side_lp(all_vertices, float(ang_test))
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_angle = float(ang_test)
    
    return best_center[0], best_center[1], best_side, best_angle

def _hex_vertices(cx, cy, side, angle_deg):
    """Get 6 vertices of a regular hexagon."""
    angle_rad = math.radians(angle_deg)
    verts = []
    for k in range(6):
        a = angle_rad + math.pi * k / 3
        verts.append((cx + side * math.cos(a), cy + side * math.sin(a)))
    return np.array(verts)

def _segment_intersection(p1, p2, p3, p4):
    """Check if segments p1-p2 and p3-p4 intersect (proper intersection only)."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-12:
        return False
    
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
    
    eps = 1e-9
    return eps < t < 1 - eps and eps < u < 1 - eps

def _hexagons_disjoint(h1, h2, eps=1e-7):
    """Check if two hexagons are disjoint (no interior overlap)."""
    v1 = _hex_vertices(h1[0], h1[1], 1.0, h1[2])
    v2 = _hex_vertices(h2[0], h2[1], 1.0, h2[2])
    
    # Check if any edge of h1 intersects any edge of h2
    for i in range(6):
        p1 = v1[i]
        p2 = v1[(i + 1) % 6]
        for j in range(6):
            p3 = v2[j]
            p4 = v2[(j + 1) % 6]
            if _segment_intersection(p1, p2, p3, p4):
                return False
    
    # Check if center of h1 is inside h2 or vice versa
    # Point in hexagon: check all 6 edge half-planes
    def point_in_hex(pt, hex_verts):
        for i in range(6):
            a = hex_verts[i]
            b = hex_verts[(i + 1) % 6]
            # Cross product to check which side
            cross = (b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0])
            if cross < -eps:
                return False
        return True
    
    c1 = np.array([h1[0], h1[1]])
    c2 = np.array([h2[0], h2[1]])
    
    if point_in_hex(c1, v2) or point_in_hex(c2, v1):
        return False
    
    # Check if any vertex of one is inside the other
    for v in v1:
        if point_in_hex(v, v2):
            return False
    for v in v2:
        if point_in_hex(v, v1):
            return False
    
    return True

def _all_disjoint(inner_hexes, eps=1e-7):
    """Check if all inner hexagons are pairwise disjoint."""
    n = len(inner_hexes)
    for i in range(n):
        for j in range(i + 1, n):
            if not _hexagons_disjoint(inner_hexes[i], inner_hexes[j], eps):
                return False
    return True

def _separation_penalty(inner_hexes, min_gap=1e-4):
    """Compute a penalty based on minimum separation between hexagons."""
    n = len(inner_hexes)
    max_overlap = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            h1 = inner_hexes[i]
            h2 = inner_hexes[j]
            v1 = _hex_vertices(h1[0], h1[1], 1.0, h1[2])
            v2 = _hex_vertices(h2[0], h2[1], 1.0, h2[2])
            
            # Compute minimum distance between any edge pair
            min_dist = float('inf')
            for i_e in range(6):
                p1 = v1[i_e]
                p2 = v1[(i_e + 1) % 6]
                for j_e in range(6):
                    p3 = v2[j_e]
                    p4 = v2[(j_e + 1) % 6]
                    
                    # Check segment-segment distance
                    # Parametric: find closest points
                    d1 = p2 - p1
                    d2 = p4 - p3
                    r = p1 - p3
                    a = np.dot(d1, d1)
                    e = np.dot(d2, d2)
                    f = np.dot(d2, r)
                    
                    if a < 1e-12 and e < 1e-12:
                        dist = np.linalg.norm(r)
                    elif a < 1e-12:
                        t = 0.0
                        s = np.clip(f / e, 0, 1)
                        dist = np.linalg.norm(r + s * d2)
                    elif e < 1e-12:
                        s = 0.0
                        t = np.clip(-np.dot(d1, r) / a, 0, 1)
                        dist = np.linalg.norm(d1 * t - r)
                    else:
                        c = np.dot(d1, r)
                        b = np.dot(d1, d2)
                        denom = a * e - b * b
                        if denom < 1e-12:
                            t = 0.0
                        else:
                            t = np.clip((b * f - c * e) / denom, 0, 1)
                        s = (b * t + f) / e
                        s = np.clip(s, 0, 1)
                        t = np.clip(t, 0, 1)
                        dist = np.linalg.norm(d1 * t - r + d2 * s)
                    
                    if dist < min_dist:
                        min_dist = dist
            
            # Also check vertex-to-edge distances
            for v in v1:
                for j_e in range(6):
                    p3 = v2[j_e]
                    p4 = v2[(j_e + 1) % 6]
                    # Distance from point v to segment p3-p4
                    d = p4 - p3
                    if np.dot(d, d) < 1e-12:
                        dist = np.linalg.norm(v - p3)
                    else:
                        t = np.clip(np.dot(v - p3, d) / np.dot(d, d), 0, 1)
                        dist = np.linalg.norm(v - (p3 + t * d))
                    if dist < min_dist:
                        min_dist = dist
            
            for v in v2:
                for i_e in range(6):
                    p1 = v1[i_e]
                    p2 = v1[(i_e + 1) % 6]
                    d = p2 - p1
                    if np.dot(d, d) < 1e-12:
                        dist = np.linalg.norm(v - p1)
                    else:
                        t = np.clip(np.dot(v - p1, d) / np.dot(d, d), 0, 1)
                        dist = np.linalg.norm(v - (p1 + t * d))
                    if dist < min_dist:
                        min_dist = dist
            
            if min_dist < min_gap:
                overlap = min_gap - min_dist
                if overlap > max_overlap:
                    max_overlap = overlap
    
    return max_overlap

def _build_initial_positions(scale=1.001, rot_deg=0.0):
    """Build initial positions on a hexagonal lattice."""
    sqrt3 = math.sqrt(3)
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
    
    rot_rad = math.radians(rot_deg)
    inner = []
    for (px, py) in positions:
        rx = px * math.cos(rot_rad) - py * math.sin(rot_rad)
        ry = px * math.sin(rot_rad) + py * math.cos(rot_rad)
        inner.append([rx, ry, 30.0 + rot_deg])
    
    return inner

def _optimize_positions(inner, all_vertices=None):
    """Optimize inner hexagon positions using scipy minimize."""
    if all_vertices is None:
        all_vertices = _compute_all_vertices(inner)
    
    # Flatten inner hexes into a parameter vector
    x0 = np.array([v for h in inner for v in h], dtype=float)
    
    def objective(params):
        # Reshape params back to hexagons
        inner_temp = []
        for k in range(N):
            ix = params[k * 3]
            iy = params[k * 3 + 1]
            iangle = params[k * 3 + 2]
            inner_temp.append([ix, iy, iangle])
        
        # Check disjointness
        max_overlap = _separation_penalty(inner_temp, min_gap=1e-4)
        
        # Compute outer side length
        all_verts = _compute_all_vertices(inner_temp)
        # Quick estimate: use current best angle
        cx = all_verts[:, 0].mean()
        cy = all_verts[:, 1].mean()
        
        # Try a few angles for quick estimate
        best_side_est = float('inf')
        for ang in [0.0, 15.0, 30.0, 45.0]:
            angles = np.array([math.radians(ang + 30 + 60 * k) for k in range(6)])
            normals = np.column_stack([np.cos(angles), np.sin(angles)])
            projs = (all_verts - np.array([cx, cy])) @ normals.T
            r = projs.max()
            side = r / (math.sqrt(3) / 2)
            if side < best_side_est:
                best_side_est = side
        
        # Objective: outer side + large penalty for overlap
        return best_side_est + 1000.0 * max_overlap ** 2
    
    try:
        res = minimize(objective, x0, method='L-BFGS-B',
                       options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12})
        if res.fun < objective(x0):
            inner_opt = []
            for k in range(N):
                ix = res.x[k * 3]
                iy = res.x[k * 3 + 1]
                iangle = res.x[k * 3 + 2]
                inner_opt.append([ix, iy, iangle])
            return inner_opt
    except Exception:
        pass
    
    return inner

def construct():
    global _START_TIME, _DEADLINE
    
    if time.time() > _DEADLINE - 5.0:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    best_result = None
    best_side = float('inf')
    
    # Try multiple configurations
    configs = []
    
    # Standard hexagonal lattice with different scales and rotations
    for scale in [1.001, 1.003, 1.005, 1.01]:
        for rot_deg in [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0]:
            if time.time() > _DEADLINE - 10.0:
                break
            inner = _build_initial_positions(scale, rot_deg)
            configs.append(inner)
    
    # Try different lattice orientations
    sqrt3 = math.sqrt(3)
    for scale in [1.001, 1.005]:
        for rot_deg in [0.0, 15.0, 30.0, 45.0]:
            if time.time() > _DEADLINE - 10.0:
                break
            # Rotated lattice
            u_angle = math.radians(rot_deg)
            v_angle = math.radians(rot_deg + 60)
            u = (sqrt3 * scale * math.cos(u_angle), sqrt3 * scale * math.sin(u_angle))
            v = (sqrt3 * scale * math.cos(v_angle), sqrt3 * scale * math.sin(v_angle))
            
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
            
            inner = [[pos[0], pos[1], 30.0 + rot_deg] for pos in positions]
            configs.append(inner)
    
    # Evaluate each configuration
    for inner in configs:
        if time.time() > _DEADLINE - 15.0:
            break
        
        # Quick check: are hexagons disjoint?
        if not _all_disjoint(inner, eps=1e-6):
            continue
        
        # Compute outer hexagon
        cx, cy, side, angle = _min_enclosing_hex_side(inner)
        
        if side < best_side:
            best_side = side
            best_result = (inner, (cx, cy), side, angle)
    
    # Optimize the best configuration further
    if best_result is not None and time.time() < _DEADLINE - 30.0:
        inner_best = best_result[0]
        inner_opt = _optimize_positions(inner_best)
        
        if _all_disjoint(inner_opt, eps=1e-6):
            cx, cy, side, angle = _min_enclosing_hex_side(inner_opt)
            if side < best_side:
                best_side = side
                best_result = (inner_opt, (cx, cy), side, angle)
    
    # Try a few more optimized configurations
    if time.time() < _DEADLINE - 25.0:
        for inner in configs[:5]:
            if time.time() > _DEADLINE - 20.0:
                break
            if not _all_disjoint(inner, eps=1e-6):
                continue
            inner_opt = _optimize_positions(inner)
            if _all_disjoint(inner_opt, eps=1e-6):
                cx, cy, side, angle = _min_enclosing_hex_side(inner_opt)
                if side < best_side:
                    best_side = side
                    best_result = (inner_opt, (cx, cy), side, angle)
    
    if best_result is None:
        # Fallback: linear arrangement
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_result
