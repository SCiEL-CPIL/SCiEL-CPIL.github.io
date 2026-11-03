import math
import time
import numpy as np
from scipy.optimize import linprog

START_TIME = time.time()
TIME_LIMIT = 148.0
SAFETY_MARGIN = 6.0
APOTHEM = math.sqrt(3) / 2.0
N = 11
EPS = 1e-10


def _hexagon_vertices(cx, cy, side, angle_rad):
    """Vertices of a regular hexagon (flat-top convention, angle=0 -> flat top)."""
    # For a flat-top hexagon with side s centered at (cx,cy):
    # vertices at angles 0,60,...,300 deg from +x, scaled by s.
    # Actually flat-top: vertices at 0,60,...; pointy-top: 30,90,...
    # We use the convention: outward normals at angle = angle_rad + pi/6 + k*pi/3
    # which means the hexagon has a vertex in direction angle_rad (pointy-top style)
    # Let's define: vertex k at angle = angle_rad + k*pi/3, radius = side
    verts = []
    for k in range(6):
        a = angle_rad + k * math.pi / 3.0
        verts.append((cx + side * math.cos(a), cy + side * math.sin(a)))
    return np.array(verts)


def _separating_axis_overlap(v1, v2):
    """True if two convex hexagons (each 6 verts) overlap using SAT."""
    axes = []
    for v in (v1, v2):
        for i in range(6):
            p1 = v[i]
            p2 = v[(i + 1) % 6]
            edge = p2 - p1
            n = np.array([-edge[1], edge[0]])
            nl = np.linalg.norm(n)
            if nl > 1e-12:
                axes.append(n / nl)
    for n in axes:
        d1 = v1 @ n
        d2 = v2 @ n
        min1, max1 = d1.min(), d1.max()
        min2, max2 = d2.min(), d2.max()
        if max1 < min2 - EPS or max2 < min1 - EPS:
            return False
    return True


def _check_disjoint(centers, angle_rad):
    """Check all pairs of unit hexagons are disjoint."""
    verts = [_hexagon_vertices(centers[i, 0], centers[i, 1], 1.0, angle_rad) for i in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            if _separating_axis_overlap(verts[i], verts[j]):
                return False
    return True


def _smallest_enclosing_hex(centers, angle_rad):
    """Return (side, center) of smallest regular hexagon (rotated by angle_rad) enclosing centers."""
    # Outward normals of the enclosing hexagon
    normals = []
    for k in range(6):
        a = angle_rad + math.pi / 6.0 + k * math.pi / 3.0
        normals.append((math.cos(a), math.sin(a)))
    normals = np.array(normals)  # (6,2)

    # LP: minimize D s.t. (c - p)·n_k <= D and (p - c)·n_k <= D  for all k,p
    # vars: [cx, cy, D]
    A_ub = []
    b_ub = []
    for nx, ny in normals:
        for px, py in centers:
            # (cx - px)*nx + (cy - py)*ny <= D  =>  nx*cx + ny*cy - D <= px*nx + py*ny
            A_ub.append([nx, ny, -1.0])
            b_ub.append(px * nx + py * ny)
            # (px - cx)*nx + (py - cy)*ny <= D  =>  -nx*cx - ny*cy - D <= -px*nx - py*ny
            A_ub.append([-nx, -ny, -1.0])
            b_ub.append(-px * nx - py * ny)
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    c_obj = np.array([0.0, 0.0, 1.0])
    bounds = [(None, None), (None, None), (0.0, None)]
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.status == 0:
            cx, cy, D = res.x
            side = (D + APOTHEM) / APOTHEM
            return float(side), (float(cx), float(cy))
    except Exception:
        pass
    # Fallback: centroid
    cx = float(centers[:, 0].mean())
    cy = float(centers[:, 1].mean())
    D = 0.0
    for nx, ny in normals:
        proj = (centers[:, 0] - cx) * nx + (centers[:, 1] - cy) * ny
        D = max(D, proj.max(), -proj.min())
    side = (D + APOTHEM) / APOTHEM
    return float(side), (float(cx), float(cy))


def _honeycomb_init():
    """Standard honeycomb arrangement of 11 hexagons (1 center + 6 ring + 4 extra)."""
    pts = [(0.0, 0.0)]
    for k in range(6):
        a = k * math.pi / 3.0
        pts.append((math.sqrt(3) * math.cos(a), math.sqrt(3) * math.sin(a)))
    # 4 extra placed in gaps to keep compact
    pts.append((2 * math.sqrt(3) * math.cos(0.0), 2 * math.sqrt(3) * math.sin(0.0)))
    pts.append((2 * math.sqrt(3) * math.cos(math.pi / 3.0), 2 * math.sqrt(3) * math.sin(math.pi / 3.0)))
    pts.append((2 * math.sqrt(3) * math.cos(2 * math.pi / 3.0), 2 * math.sqrt(3) * math.sin(2 * math.pi / 3.0)))
    pts.append((2 * math.sqrt(3) * math.cos(math.pi), 2 * math.sqrt(3) * math.sin(math.pi)))
    return np.array(pts)


def _cluster_init():
    """A more compact 4-3-4 style cluster."""
    pts = []
    # row 0: 4
    for i in range(4):
        pts.append((i * math.sqrt(3), 0.0))
    # row 1: 3
    for i in range(3):
        pts.append((math.sqrt(3) / 2 + i * math.sqrt(3), 1.5))
    # row 2: 4
    for i in range(4):
        pts.append((i * math.sqrt(3), 3.0))
    return np.array(pts)


def _relax(centers0, angle_rad, max_iter=400):
    """Iteratively relax: repel overlaps, then project onto enclosing hexagon."""
    centers = centers0.copy()
    step = 0.05
    for it in range(max_iter):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        # Repulsion: for each overlapping pair, push apart along center-to-center
        moved = 0.0
        verts = [_hexagon_vertices(centers[i, 0], centers[i, 1], 1.0, angle_rad) for i in range(N)]
        for i in range(N):
            for j in range(i + 1, N):
                if _separating_axis_overlap(verts[i], verts[j]):
                    d = centers[j] - centers[i]
                    dl = np.linalg.norm(d)
                    if dl < 1e-9:
                        d = np.array([1.0, 0.0]); dl = 1.0
                    d = d / dl
                    push = step * 0.5
                    centers[i] -= d * push
                    centers[j] += d * push
                    moved += push
                    verts = [_hexagon_vertices(centers[k, 0], centers[k, 1], 1.0, angle_rad) for k in range(N)]
        # Project onto enclosing hexagon (this only shrinks if we then scale centers toward center)
        # We do NOT force centers inside; instead we record the enclosing hexagon.
        if moved < 1e-9:
            break
        # Decay step
        step *= 0.98
    # Final enclosing hex
    side, (cx, cy) = _smallest_enclosing_hex(centers, angle_rad)
    return side, centers, (cx, cy)


def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    inits = [_honeycomb_init(), _cluster_init()]

    best_side = float('inf')
    best_centers = None
    best_center = (0.0, 0.0)
    best_angle = 0.0

    for init in inits:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        # Coarse angle sweep
        for ai in range(61):
            angle_rad = ai * (math.pi / 6.0) / 60.0
            side, centers, center = _relax(init, angle_rad, max_iter=200)
            if side < best_side:
                best_side = side
                best_centers = centers.copy()
                best_center = center
                best_angle = angle_rad

        # Local angle refinement around the best found so far for this init
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        # find best angle in sweep quickly
        local_best_angle = best_angle
        for _ in range(3):
            improved = False
            for da in [0.005, -0.005, 0.01, -0.01, 0.002, -0.002]:
                if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                    break
                ang = local_best_angle + da
                side, centers, center = _relax(init, ang, max_iter=300)
                if side < best_side - 1e-12:
                    best_side = side
                    best_centers = centers.copy()
                    best_center = center
                    best_angle = ang
                    local_best_angle = ang
                    improved = True
            if not improved:
                break

    if best_centers is None:
        best_centers = _honeycomb_init()
        best_center = (0.0, 0.0)
        best_angle = 0.0
        best_side, best_centers, best_center = _relax(best_centers, 0.0, max_iter=100)

    # Safety: verify disjoint; if not, scale down slightly by reducing side (we can't shrink hexagons,
    # so if overlap remains, we accept it only if it's sub-pixel; otherwise nudge)
    if not _check_disjoint(best_centers, best_angle):
        # Tiny random nudge to break degeneracy
        rng = np.random.default_rng(42)
        for _ in range(50):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            jitter = rng.normal(0, 1e-4, size=(N, 2))
            cand = best_centers + jitter
            if _check_disjoint(cand, best_angle):
                side, _, center = _smallest_enclosing_hex(cand, best_angle)
                if side < best_side + 1e-6:
                    best_centers = cand
                    best_side = side
                    best_center = center
                break

    cx, cy = best_center
    angle_deg = math.degrees(best_angle)
    inner = [[float(best_centers[i, 0]), float(best_centers[i, 1]), float(angle_deg)] for i in range(N)]
    center = [float(cx), float(cy)]
    side = float(best_side)
    rotation = float(angle_deg)
    return inner, center, side, rotation
