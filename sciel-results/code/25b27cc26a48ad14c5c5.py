"""Hexagon packing: 12 unit hexagons inside a regular hexagon, minimize outer side."""
import math
import time
import random
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 12
TIME_LIMIT = 145.0  # safety margin below 150s

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    """Vertices of a regular hexagon centered at (cx,cy) with rotation angle_deg."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    verts = []
    for k in range(6):
        theta = math.radians(60 * k)
        x = side * math.cos(theta)
        y = side * math.sin(theta)
        verts.append((cx + ca * x - sa * y, cy + sa * x + ca * y))
    return verts

def _project_polygon(verts, axis):
    """Project polygon vertices onto axis, return (min, max)."""
    ax, ay = axis
    vals = [vx * ax + vy * ay for vx, vy in verts]
    return min(vals), max(vals)

def _axes_from_polygon(verts):
    """Get edge normals (perpendicular to edges) for SAT."""
    axes = []
    n = len(verts)
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < 1e-12:
            continue
        nx, ny = -dy / length, dx / length
        axes.append((nx, ny))
    return axes

def _hex_normals(angle_deg):
    """Edge normals for a unit hexagon with given rotation angle."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    normals = []
    for k in range(6):
        theta = math.radians(30 + 60 * k)
        nx = math.cos(theta)
        ny = math.sin(theta)
        rnx = ca * nx - sa * ny
        rny = sa * nx + ca * ny
        normals.append((rnx, rny))
    return normals

def _support_function(cx, cy, angle_deg, nx, ny):
    """Support function of unit hexagon at (cx,cy) with rotation angle in direction (nx,ny)."""
    # Precompute vertex offsets for a unit hexagon at angle_deg
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    max_val = 0.0
    for k in range(6):
        theta = math.radians(60 * k)
        x = math.cos(theta)
        y = math.sin(theta)
        vx = cx + ca * x - sa * y
        vy = cy + sa * x + ca * y
        dot = vx * nx + vy * ny
        if dot > max_val:
            max_val = dot
    return max_val

def _separating_axis_check(vertsA, vertsB, normalsA, normalsB, tol=1e-9):
    """Check if two convex polygons overlap using SAT with precomputed normals."""
    axes = normalsA + normalsB
    for axis in axes:
        minA, maxA = _project_polygon(vertsA, axis)
        minB, maxB = _project_polygon(vertsB, axis)
        if maxA <= minB + tol or maxB <= minA + tol:
            return False
    return True

def _min_overlap_depth(vertsA, vertsB, normalsA, normalsB):
    """Compute minimum overlap depth between two convex polygons (0 if disjoint)."""
    axes = normalsA + normalsB
    max_overlap = 0.0
    for axis in axes:
        minA, maxA = _project_polygon(vertsA, axis)
        minB, maxB = _project_polygon(vertsB, axis)
        overlap = min(maxA, maxB) - max(minA, minB)
        if overlap > max_overlap:
            max_overlap = overlap
        if overlap < -1e-9:
            return 0.0
    return max_overlap

def _width_based_distance(p1, p2):
    """
    Width-based distance between two hexagons using support functions.
    Positive = separated, negative = overlapping.
    Uses the support function in the direction of the center-to-center vector.
    """
    x1, y1, a1 = p1[0], p1[1], p1[2]
    x2, y2, a2 = p2[0], p2[1], p2[2]
    dx = x2 - x1
    dy = y2 - y1
    dist = math.hypot(dx, dy)
    if dist < 1e-10:
        return 0.0
    nx, ny = dx / dist, dy / dist
    # Support of hex1 in direction toward hex2
    s1 = _support_function(0, 0, a1, nx, ny)
    # Support of hex2 in direction toward hex1 (opposite direction)
    s2 = _support_function(0, 0, a2, -nx, -ny)
    return dist - s1 - s2

def _outer_hex_side(centers_angles, center, angle_deg):
    """Compute minimum side length S of outer hexagon containing all inner hexagons."""
    a = math.radians(angle_deg)
    apothem_unit = math.sqrt(3) / 2.0
    normals = []
    for k in range(6):
        na = a + math.radians(90 + 60 * k)
        normals.append((math.cos(na), math.sin(na)))
    cx, cy = center
    max_need = 0.0
    for (px, py, pangle) in centers_angles:
        verts = _hex_vertices(px, py, pangle)
        for (vx, vy) in verts:
            for (nx, ny) in normals:
                val = (nx * (vx - cx) + ny * (vy - cy)) / apothem_unit
                if val > max_need:
                    max_need = val
    return max_need

def _is_valid(centers_angles, tol=1e-9):
    """Check all pairs of hexagons are disjoint."""
    m = len(centers_angles)
    polys = [_hex_vertices(p[0], p[1], p[2]) for p in centers_angles]
    normals = [_hex_normals(p[2]) for p in centers_angles]
    for i in range(m):
        for j in range(i + 1, m):
            # Quick center-distance rejection: if centers are far enough apart, skip
            dx = centers_angles[i][0] - centers_angles[j][0]
            dy = centers_angles[i][1] - centers_angles[j][1]
            center_dist = math.hypot(dx, dy)
            if center_dist > 4.0:  # Max possible overlap distance for unit hexagons
                continue
            if _separating_axis_check(polys[i], polys[j], normals[i], normals[j], tol):
                return False
    return True

def _smooth_collision_penalty(cfg, beta=20.0):
    """
    Smooth collision penalty using width-based distances (support functions).
    This allows hexagons to pack closer when their orientations permit.
    """
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            d = _width_based_distance(cfg[i], cfg[j])
            if d < 0:
                # Use a smoother penalty that decays faster for deep overlaps
                penalty += d * d * (1.0 + 5.0 * abs(d))
    return penalty

def _smooth_outer_side(cfg, center, angle_deg, beta=15.0):
    """Smooth outer side length using log-sum-exp over all vertex-projection pairs."""
    a = math.radians(angle_deg)
    apothem_unit = math.sqrt(3) / 2.0
    normals = []
    for k in range(6):
        na = a + math.radians(90 + 60 * k)
        normals.append((math.cos(na), math.sin(na)))
    cx, cy = center
    vals = []
    for (px, py, pangle) in cfg:
        verts = _hex_vertices(px, py, pangle)
        for (vx, vy) in verts:
            for (nx, ny) in normals:
                val = (nx * (vx - cx) + ny * (vy - cy)) / apothem_unit
                vals.append(val)
    vals = np.array(vals)
    max_val = np.max(vals)
    shifted = vals - max_val
    lse = max_val + (1.0 / beta) * np.log(np.sum(np.exp(beta * shifted)))
    return lse

def construct():
    t0 = time.time()
    random.seed(42)
    np.random.seed(42)

    def full_score(cfg, center, angle):
        """Return outer side if valid, else None."""
        if not _is_valid(cfg):
            return None
        return _outer_hex_side(cfg, tuple(center), angle)

    def make_hex_lattice():
        """3x4 hexagonal lattice with offset."""
        pts = []
        row_height = math.sqrt(3) * 1.0
        col_width = 2.0 * 1.0
        for r in range(3):
            for c in range(4):
                x = c * col_width + (col_width / 2 if r % 2 else 0.0)
                y = r * row_height
                pts.append([x - 3.0, y - row_height, 0.0])
        return pts

    def make_ring():
        """12 hexagons in a ring, oriented outward."""
        pts = []
        R = 2.5
        for i in range(N):
            theta = 2 * math.pi * i / N
            x = R * math.cos(theta)
            y = R * math.sin(theta)
            angle_deg = math.degrees(theta)
            pts.append([x, y, angle_deg])
        return pts

    def make_flower():
        """Flower pattern: 1 center + 6 inner ring + 5 outer."""
        pts = []
        pts.append([0.0, 0.0, 0.0])
        R1 = math.sqrt(3)
        for i in range(6):
            theta = 2 * math.pi * i / 6
            x = R1 * math.cos(theta)
            y = R1 * math.sin(theta)
            pts.append([x, y, 0.0])
        R2 = 2 * math.sqrt(3)
        for i in range(5):
            theta = 2 * math.pi * (i + 0.5) / 5
            x = R2 * math.cos(theta)
            y = R2 * math.sin(theta)
            pts.append([x, y, 0.0])
        return pts

    def make_dense_cluster():
        """Tight cluster: center + 6 around + 5 more."""
        pts = []
        pts.append([0.0, 0.0, 0.0])
        for i in range(6):
            theta = 2 * math.pi * i / 6
            x = math.sqrt(3) * math.cos(theta)
            y = math.sqrt(3) * math.sin(theta)
            pts.append([x, y, 0.0])
        for i in range(5):
            theta = 2 * math.pi * (i + 0.5) / 5
            x = 2 * math.sqrt(3) * math.cos(theta)
            y = 2 * math.sqrt(3) * math.sin(theta)
            pts.append([x, y, 0.0])
        return pts

    def make_hexagonal_12():
        """12 hexagons: center + 5 inner + 6 outer."""
        pts = []
        pts.append([0.0, 0.0, 0.0])
        R1 = math.sqrt(3)
        for i in range(5):
            theta = 2 * math.pi * i / 5 + math.pi / 5
            x = R1 * math.cos(theta)
            y = R1 * math.sin(theta)
            pts.append([x, y, 0.0])
        R2 = 2 * math.sqrt(3)
        for i in range(6):
            theta = 2 * math.pi * i / 6
            x = R2 * math.cos(theta)
            y = R2 * math.sin(theta)
            pts.append([x, y, 0.0])
        return pts

    def make_honeycomb_cluster(scale=1.0):
        """High-density honeycomb cluster of 12 hexagons."""
        pts = []
        # Center
        pts.append([0.0, 0.0, 0.0])
        # First ring: 6 hexagons
        R1 = math.sqrt(3) * scale
        for i in range(6):
            theta = 2 * math.pi * i / 6
            x = R1 * math.cos(theta)
            y = R1 * math.sin(theta)
            pts.append([x, y, 0.0])
        # Second ring: 5 more hexagons (partial ring)
        R2 = 2 * math.sqrt(3) * scale
        for i in range(5):
            theta = 2 * math.pi * (i + 0.5) / 5
            x = R2 * math.cos(theta)
            y = R2 * math.sin(theta)
            pts.append([x, y, 0.0])
        return pts

    def smooth_objective(v, beta=15.0):
        """
        Smooth objective: smooth outer side + width-based collision penalty.
        v = [x1,y1,a1, ..., x12,y12,a12, cx,cy, outer_angle]
        """
        cfg = []
        for i in range(N):
            cfg.append([v[3*i], v[3*i+1], v[3*i+2]])
        center = (v[3*N], v[3*N+1])
        outer_angle = v[3*N+2]

        smooth_side = _smooth_outer_side(cfg, center, outer_angle, beta=beta)
        collision_penalty = _smooth_collision_penalty(cfg, beta=20.0)

        return smooth_side + 100.0 * collision_penalty

    def rigid_cluster_rotation_search(cfg, center, angle, time_budget):
        """
        1D golden-section search over global rotation of a rigid cluster.
        Treats the 12 hexagons as a rigid body and optimizes orientation.
        """
        def rotate_cfg(cfg, center, global_rot_deg):
            """Rotate all hexagons around center by global_rot_deg."""
            a = math.radians(global_rot_deg)
            ca, sa = math.cos(a), math.sin(a)
            new_cfg = []
            for (px, py, pangle) in cfg:
                # Translate to origin, rotate, translate back
                dx, dy = px - center[0], py - center[1]
                rx = ca * dx - sa * dy
                ry = sa * dx + ca * dy
                new_cfg.append([rx + center[0], ry + center[1], pangle + global_rot_deg])
            return new_cfg

        def objective(rot_deg):
            rotated_cfg = rotate_cfg(cfg, center, rot_deg)
            if not _is_valid(rotated_cfg):
                return 100.0
            return _outer_hex_side(rotated_cfg, tuple(center), angle)

        # Golden-section search in [-30, 30] degrees
        res = minimize_scalar(objective, bounds=(-30.0, 30.0),
                             method='bounded',
                             options={'xatol': 0.01, 'maxiter': 200})
        if res.success:
            best_rot = res.x
            best_cfg = rotate_cfg(cfg, center, best_rot)
            best_score = res.fun
            return best_cfg, best_score
        return cfg, None

    def gradient_descent_polish(cfg_init, center_init, angle_init, time_budget, beta=15.0):
        """L-BFGS-B polish with smooth objective."""
        def flatten(cfg, center, angle):
            v = []
            for p in cfg:
                v.extend([p[0], p[1], p[2]])
            v.extend([center[0], center[1], angle])
            return np.array(v, dtype=float)

        def obj(v):
            if time.time() - t0 > TIME_LIMIT - 2.0:
                return 1e6
            return smooth_objective(v, beta=beta)

        v = flatten(cfg_init, center_init, angle_init)
        
        # Use multiple optimization methods and take the best result
        best_v = v.copy()
        best_f = smooth_objective(v, beta=beta)
        
        # Method 1: L-BFGS-B
        try:
            result1 = minimize(
                obj, v, method='L-BFGS-B',
                options={'maxiter': 1500, 'maxfun': 8000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if result1.fun < best_f:
                best_v = result1.x.copy()
                best_f = result1.fun
        except Exception:
            pass
        
        # Method 2: Nelder-Mead for local refinement
        remaining = time_budget - (time.time() - t0)
        if remaining > 5.0:
            try:
                result2 = minimize(
                    obj, best_v, method='Nelder-Mead',
                    options={'maxiter': 500, 'maxfev': 3000, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                if result2.fun < best_f:
                    best_v = result2.x.copy()
                    best_f = result2.fun
            except Exception:
                pass
        
        return best_v, best_f

    # Try multiple initial configurations
    best_cfg = None
    best_score = float('inf')
    best_center = [0.0, 0.0]
    best_angle = 0.0

    inits = [make_hex_lattice(), make_ring(), make_flower(),
             make_dense_cluster(), make_hexagonal_12(), make_honeycomb_cluster()]
    for init in inits:
        s = full_score(init, (0.0, 0.0), 0.0)
        if s is not None and s < best_score:
            best_score = s
            best_cfg = [row[:] for row in init]
            best_center = [0.0, 0.0]
            best_angle = 0.0

    if best_cfg is None:
        cfg = []
        for i in range(N):
            cfg.append([random.uniform(-3, 3), random.uniform(-3, 3), 0.0])
        best_cfg = cfg
        best_score = 50.0
        best_center = [0.0, 0.0]
        best_angle = 0.0

    cfg = [row[:] for row in best_cfg]
    center = best_center[:]
    angle = best_angle
    cur_score = best_score

    # Rigid cluster rotation search as initial refinement
    remaining_time = TIME_LIMIT - (time.time() - t0)
    if remaining_time > 60.0:
        try:
            rotated_cfg, rot_score = rigid_cluster_rotation_search(
                best_cfg, best_center, best_angle, min(remaining_time * 0.2, 15.0)
            )
            if rot_score is not None and rot_score < best_score:
                best_score = rot_score
                best_cfg = rotated_cfg
        except Exception:
            pass

    cfg = [row[:] for row in best_cfg]
    center = best_center[:]
    angle = best_angle
    cur_score = best_score

    # Simulated annealing phase
    T = 0.8
    T_end = 0.003
    alpha = 0.9997

    sa_time_limit = TIME_LIMIT - 50.0  # Reserve 50s for polishing

    while time.time() - t0 < sa_time_limit:
        r = random.random()

        if r < 0.45:
            idx = random.randrange(N)
            old_x, old_y = cfg[idx][0], cfg[idx][1]
            step = 0.15 * (1.0 + 0.3 * T)
            cfg[idx][0] += random.gauss(0, step)
            cfg[idx][1] += random.gauss(0, step)
        elif r < 0.65:
            idx = random.randrange(N)
            old_ang = cfg[idx][2]
            cfg[idx][2] += random.gauss(0, 4.0 + 2.0 * T)
        elif r < 0.75:
            old_cx, old_cy = center[0], center[1]
            step = 0.08 * (1.0 + 0.3 * T)
            center[0] += random.gauss(0, step)
            center[1] += random.gauss(0, step)
        elif r < 0.85:
            old_angle = angle
            angle += random.gauss(0, 1.5 + 1.5 * T)
        elif r < 0.92:
            # Swap two hexagons' positions (preserves validity)
            idx1, idx2 = random.sample(range(N), 2)
            old_x1, old_y1, old_a1 = cfg[idx1][0], cfg[idx1][1], cfg[idx1][2]
            old_x2, old_y2, old_a2 = cfg[idx2][0], cfg[idx2][1], cfg[idx2][2]
            cfg[idx1][0], cfg[idx1][1], cfg[idx1][2] = old_x2, old_y2, old_a2
            cfg[idx2][0], cfg[idx2][1], cfg[idx2][2] = old_x1, old_y1, old_a1
        else:
            idx1, idx2 = random.sample(range(N), 2)
            old_x1, old_y1 = cfg[idx1][0], cfg[idx1][1]
            old_x2, old_y2 = cfg[idx2][0], cfg[idx2][1]
            step = 0.12 * (1.0 + 0.3 * T)
            cfg[idx1][0] += random.gauss(0, step)
            cfg[idx1][1] += random.gauss(0, step)
            cfg[idx2][0] += random.gauss(0, step)
            cfg[idx2][1] += random.gauss(0, step)

        new_score = full_score(cfg, tuple(center), angle)

        if new_score is not None:
            if new_score < cur_score or (T > T_end and random.random() < math.exp(-(new_score - cur_score) / T)):
                cur_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_cfg = [row[:] for row in cfg]
                    best_center = center[:]
                    best_angle = angle
            else:
                if r < 0.45:
                    cfg[idx][0], cfg[idx][1] = old_x, old_y
                elif r < 0.65:
                    cfg[idx][2] = old_ang
                elif r < 0.75:
                    center[0], center[1] = old_cx, old_cy
                elif r < 0.85:
                    angle = old_angle
                elif r < 0.92:
                    cfg[idx1][0], cfg[idx1][1], cfg[idx1][2] = old_x1, old_y1, old_a1
                    cfg[idx2][0], cfg[idx2][1], cfg[idx2][2] = old_x2, old_y2, old_a2
                else:
                    cfg[idx1][0], cfg[idx1][1] = old_x1, old_y1
                    cfg[idx2][0], cfg[idx2][1] = old_x2, old_y2
        else:
            if r < 0.45:
                cfg[idx][0], cfg[idx][1] = old_x, old_y
            elif r < 0.65:
                cfg[idx][2] = old_ang
            elif r < 0.75:
                center[0], center[1] = old_cx, old_cy
            elif r < 0.85:
                angle = old_angle
            elif r < 0.92:
                cfg[idx1][0], cfg[idx1][1], cfg[idx1][2] = old_x1, old_y1, old_a1
                cfg[idx2][0], cfg[idx2][1], cfg[idx2][2] = old_x2, old_y2, old_a2
            else:
                cfg[idx1][0], cfg[idx1][1] = old_x1, old_y1
                cfg[idx2][0], cfg[idx2][1] = old_x2, old_y2

        T = max(T_end, T * alpha)

    # Multi-stage gradient descent polish
    remaining_time = TIME_LIMIT - (time.time() - t0)
    if remaining_time > 20.0:
        try:
            # Stage 1: Coarse polish with lower beta
            polished_v, polished_f = gradient_descent_polish(
                best_cfg, best_center, best_angle, min(remaining_time * 0.3, 15.0), beta=10.0
            )
            cfg_polished = []
            for i in range(N):
                cfg_polished.append([polished_v[3*i], polished_v[3*i+1], polished_v[3*i+2]])
            center_polished = [polished_v[3*N], polished_v[3*N+1]]
            angle_polished = polished_v[3*N+2]

            polished_score = full_score(cfg_polished, tuple(center_polished), angle_polished)
            if polished_score is not None and polished_score < best_score:
                best_score = polished_score
                best_cfg = cfg_polished
                best_center = center_polished
                best_angle = angle_polished
        except Exception:
            pass

    # Stage 2: Medium polish
    remaining_time = TIME_LIMIT - (time.time() - t0)
    if remaining_time > 15.0:
        try:
            polished_v2, polished_f2 = gradient_descent_polish(
                best_cfg, best_center, best_angle, min(remaining_time * 0.3, 12.0), beta=20.0
            )
            cfg_polished2 = []
            for i in range(N):
                cfg_polished2.append([polished_v2[3*i], polished_v2[3*i+1], polished_v2[3*i+2]])
            center_polished2 = [polished_v2[3*N], polished_v2[3*N+1]]
            angle_polished2 = polished_v2[3*N+2]

            polished_score2 = full_score(cfg_polished2, tuple(center_polished2), angle_polished2)
            if polished_score2 is not None and polished_score2 < best_score:
                best_score = polished_score2
                best_cfg = cfg_polished2
                best_center = center_polished2
                best_angle = angle_polished2
        except Exception:
            pass

    # Stage 3: Fine polish with high beta for tighter convergence
    remaining_time = TIME_LIMIT - (time.time() - t0)
    if remaining_time > 10.0:
        try:
            polished_v3, polished_f3 = gradient_descent_polish(
                best_cfg, best_center, best_angle, min(remaining_time * 0.4, 10.0), beta=40.0
            )
            cfg_polished3 = []
            for i in range(N):
                cfg_polished3.append([polished_v3[3*i], polished_v3[3*i+1], polished_v3[3*i+2]])
            center_polished3 = [polished_v3[3*N], polished_v3[3*N+1]]
            angle_polished3 = polished_v3[3*N+2]

            polished_score3 = full_score(cfg_polished3, tuple(center_polished3), angle_polished3)
            if polished_score3 is not None and polished_score3 < best_score:
                best_score = polished_score3
                best_cfg = cfg_polished3
                best_center = center_polished3
                best_angle = angle_polished3
        except Exception:
            pass

    # Rigid cluster rotation search as final refinement
    remaining_time = TIME_LIMIT - (time.time() - t0)
    if remaining_time > 8.0:
        try:
            rotated_cfg, rot_score = rigid_cluster_rotation_search(
                best_cfg, best_center, best_angle, min(remaining_time * 0.5, 8.0)
            )
            if rot_score is not None and rot_score < best_score:
                best_score = rot_score
                best_cfg = rotated_cfg
        except Exception:
            pass

    # Final quick refinement with multiple small perturbations
    remaining_time = TIME_LIMIT - (time.time() - t0)
    if remaining_time > 3.0:
        for _ in range(500):
            if time.time() - t0 > TIME_LIMIT - 1.0:
                break
            cfg_try = [row[:] for row in best_cfg]
            center_try = best_center[:]
            angle_try = best_angle
            # Try different perturbation strategies
            strategy = random.random()
            if strategy < 0.4:
                idx = random.randrange(N)
                cfg_try[idx][0] += random.gauss(0, 0.015)
                cfg_try[idx][1] += random.gauss(0, 0.015)
            elif strategy < 0.6:
                idx = random.randrange(N)
                cfg_try[idx][2] += random.gauss(0, 1.0)
            elif strategy < 0.8:
                center_try[0] += random.gauss(0, 0.01)
                center_try[1] += random.gauss(0, 0.01)
            else:
                angle_try += random.gauss(0, 0.5)
            score_try = full_score(cfg_try, tuple(center_try), angle_try)
            if score_try is not None and score_try < best_score:
                best_score = score_try
                best_cfg = cfg_try
                best_center = center_try
                best_angle = angle_try

    inner = [[float(p[0]), float(p[1]), float(p[2])] for p in best_cfg]
    return inner, (float(best_center[0]), float(best_center[1])), float(best_score), float(best_angle)

if __name__ == "__main__":
    inner, c, s, a = construct()
    print("side:", s)
