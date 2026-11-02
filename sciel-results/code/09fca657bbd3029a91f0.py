import math
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]] * N, [0.0, 0.0], 1.0, 0.0

    def hex_vertices(cx, cy, side, angle_rad):
        """Get vertices of a regular hexagon (pointy-top at angle=0)."""
        verts = np.zeros((6, 2))
        for i in range(6):
            a = angle_rad + i * math.pi / 3
            verts[i, 0] = cx + side * math.cos(a)
            verts[i, 1] = cy + side * math.sin(a)
        return verts

    def point_in_hexagon(px, py, cx, cy, side, angle_rad):
        """Check if point is inside hexagon using half-plane tests."""
        dx = px - cx
        dy = py - cy
        for k in range(6):
            a = angle_rad + k * math.pi / 3 + math.pi / 6
            nx = math.cos(a)
            ny = math.sin(a)
            if dx * nx + dy * ny > APOTHEM * side:
                return False
        return True

    def seg_seg_dist(p1, p2, p3, p4):
        """Distance between two line segments (numpy arrays of shape (2,))."""
        d = min(
            point_seg_dist(p1, p3, p4),
            point_seg_dist(p2, p3, p4),
            point_seg_dist(p3, p1, p2),
            point_seg_dist(p4, p1, p2)
        )
        return d

    def point_seg_dist(p, a, b):
        """Distance from point to line segment."""
        ab = b - a
        ap = p - a
        denom = np.dot(ab, ab)
        if denom < 1e-12:
            return np.linalg.norm(p - a)
        t = np.dot(ap, ab) / denom
        t = max(0.0, min(1.0, t))
        proj = a + t * ab
        return np.linalg.norm(p - proj)

    def min_dist_between_hexes(c1, a1, c2, a2):
        """
        Compute minimum distance between two unit hexagons.
        Returns negative if they overlap.
        """
        dc = math.hypot(c1[0] - c2[0], c1[1] - c2[1])
        if dc >= 2.0:
            return dc - 2.0

        v1 = hex_vertices(c1[0], c1[1], 1.0, a1)
        v2 = hex_vertices(c2[0], c2[1], 1.0, a2)

        # Check vertex containment
        for vx, vy in v1:
            if point_in_hexagon(vx, vy, c2[0], c2[1], 1.0, a2):
                return -1.0
        for vx, vy in v2:
            if point_in_hexagon(vx, vy, c1[0], c1[1], 1.0, a1):
                return -1.0

        # Compute edge-edge distances
        edges1 = [(v1[i], v1[(i + 1) % 6]) for i in range(6)]
        edges2 = [(v2[i], v2[(i + 1) % 6]) for i in range(6)]

        min_dist = dc
        for e1 in edges1:
            for e2 in edges2:
                d = seg_seg_dist(e1[0], e1[1], e2[0], e2[1])
                if d < min_dist:
                    min_dist = d

        return min_dist - 2.0

    def outer_hexagon_size(positions, angles_rad, center, rotation_rad):
        """
        Compute the minimum side length of a regular hexagon (with given rotation)
        that contains all unit hexagons.
        """
        max_proj = 0.0
        for k in range(6):
            a = rotation_rad + k * math.pi / 3 + math.pi / 6
            nx = math.cos(a)
            ny = math.sin(a)
            for i in range(N):
                cx, cy = positions[i]
                v = hex_vertices(cx, cy, 1.0, angles_rad[i])
                for vx, vy in v:
                    proj = (vx - center[0]) * nx + (vy - center[1]) * ny
                    if abs(proj) > max_proj:
                        max_proj = abs(proj)
        S = max_proj / APOTHEM
        return S

    def objective(params):
        """Objective: outer hexagon side length, with penalty for overlaps."""
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return 1e10

        pos = np.array(params[:2 * N]).reshape(N, 2)
        angs = params[2 * N:3 * N]
        cx, cy = params[3 * N:3 * N + 2]
        outer_rot = params[3 * N + 2]

        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                d = min_dist_between_hexes(pos[i], angs[i], pos[j], angs[j])
                if d < 0:
                    penalty += 10.0 * (-d) ** 2

        S = outer_hexagon_size(pos, angs, (cx, cy), outer_rot)
        return S + penalty

    def make_axial_config():
        """Standard 11-hexagon hexagonal cluster in axial coordinates."""
        axial = [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
                 (2, 0), (-2, 0), (0, 2), (0, -2)]
        pos = []
        angs = []
        for q, r in axial:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pos.append([x, y])
            angs.append(0.0)
        return np.array(pos), np.array(angs)

    def make_ring_config():
        """Center + 6 ring + 4 outer."""
        pos = []
        angs = []
        pos.append([0.0, 0.0])
        angs.append(0.0)
        for i in range(6):
            a = i * math.pi / 3
            pos.append([2.0 * math.cos(a), 2.0 * math.sin(a)])
            angs.append(0.0)
        for i in range(4):
            a = math.pi / 6 + i * math.pi / 2
            pos.append([4.0 * math.cos(a), 4.0 * math.sin(a)])
            angs.append(0.0)
        return np.array(pos), np.array(angs)

    def make_cross_config():
        """Symmetric cross pattern."""
        pos = []
        angs = []
        pos.append([0.0, 0.0])
        angs.append(0.0)
        for a in [0, math.pi / 2, math.pi, 3 * math.pi / 2]:
            pos.append([2.0 * math.cos(a), 2.0 * math.sin(a)])
            angs.append(0.0)
        for a in [math.pi / 4, 3 * math.pi / 4, 5 * math.pi / 4, 7 * math.pi / 4]:
            pos.append([2.5 * math.cos(a), 2.5 * math.sin(a)])
            angs.append(0.0)
        return np.array(pos), np.array(angs)

    # Generate initial configurations
    configs = [make_axial_config(), make_ring_config(), make_cross_config()]

    best_S = float('inf')
    best_params = None

    for pos, angs in configs:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break

        cx = np.mean(pos[:, 0])
        cy = np.mean(pos[:, 1])

        # Try multiple outer rotations
        for outer_rot_deg in [0, 30, 15, 45, 60]:
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break

            outer_rot = math.radians(outer_rot_deg)
            params = np.concatenate([
                pos.ravel(),
                angs,
                [cx, cy],
                [outer_rot]
            ])

            # Nelder-Mead optimization
            try:
                res = minimize(objective, params, method='Nelder-Mead',
                               options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10})
                if res.fun < best_S:
                    best_S = res.fun
                    best_params = res.x.copy()
            except Exception:
                pass

            # Powell optimization
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                try:
                    res = minimize(objective, params, method='Powell',
                                   options={'maxiter': 800, 'xtol': 1e-10, 'ftol': 1e-10})
                    if res.fun < best_S:
                        best_S = res.fun
                        best_params = res.x.copy()
                except Exception:
                    pass

    if best_params is None:
        pos, angs = make_axial_config()
        cx = np.mean(pos[:, 0])
        cy = np.mean(pos[:, 1])
        best_params = np.concatenate([pos.ravel(), angs, [cx, cy], [0.0]])
        best_S = objective(best_params)

    # Extract final result
    pos = best_params[:2 * N].reshape(N, 2)
    angs = best_params[2 * N:3 * N]
    cx, cy = best_params[3 * N:3 * N + 2]
    outer_rot = best_params[3 * N + 2]

    # Compute actual outer hexagon size
    S = outer_hexagon_size(pos, angs, (cx, cy), outer_rot)

    inner = []
    for i in range(N):
        inner.append([float(pos[i][0]), float(pos[i][1]), float(math.degrees(angs[i]))])

    center = [float(cx), float(cy)]
    side = float(S)
    rotation = float(math.degrees(outer_rot))

    return inner, center, side, rotation
