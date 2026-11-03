"""
Pack 12 unit hexagons inside a regular hexagon, minimizing outer side length.
Combines topology-constrained optimization (parameterized rings/lattices) with
free-placement refinement via smooth SAT-gap penalties and Nelder-Mead.
"""
import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = time.time() + 145.0


def hex_vertices(cx, cy, s, angle_deg):
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts.append((cx + s * math.cos(theta), cy + s * math.sin(theta)))
    return verts


def point_in_convex_polygon(px, py, poly):
    n = len(poly)
    sign = 0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if abs(cross) < 1e-10:
            continue
        s = 1 if cross > 0 else -1
        if sign == 0:
            sign = s
        elif s != sign:
            return False
    return True


def sat_gap(A, B):
    """Maximum SAT gap: positive = separated, negative = overlap depth."""
    max_gap = -1e10
    for poly in (A, B):
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            dx, dy = x2 - x1, y2 - y1
            nx, ny = -dy, dx
            norm = math.sqrt(nx * nx + ny * ny)
            if norm < 1e-12:
                continue
            nx, ny = nx / norm, ny / norm
            minA = min(nx * px + ny * py for px, py in A)
            maxA = max(nx * px + ny * py for px, py in A)
            minB = min(nx * px + ny * py for px, py in B)
            maxB = max(nx * px + ny * py for px, py in B)
            if maxA < minB:
                gap = minB - maxA
            elif maxB < minA:
                gap = minA - maxB
            else:
                gap = -(min(maxA, maxB) - max(minA, minB))
            if gap > max_gap:
                max_gap = gap
    return max_gap


def polygons_overlap_sat(A, B):
    return sat_gap(A, B) < -1e-9


def compute_min_outer_side(centers, angles, ocx, ocy, otheta_deg):
    otheta = math.radians(otheta_deg)
    max_proj = 0.0
    for k in range(6):
        normal_angle = otheta + math.pi / 6 + k * math.pi / 3
        nx = math.cos(normal_angle)
        ny = math.sin(normal_angle)
        for i in range(len(centers)):
            cx, cy = centers[i]
            angle = math.radians(angles[i])
            for vi in range(6):
                vert_angle = angle + vi * math.pi / 3
                vx = cx + math.cos(vert_angle)
                vy = cy + math.sin(vert_angle)
                proj = (vx - ocx) * nx + (vy - ocy) * ny
                if proj > max_proj:
                    max_proj = proj
    return 2.0 * max_proj / math.sqrt(3)


def check_validity(centers, angles, ocx, ocy, outer_side, outer_angle):
    outer = hex_vertices(ocx, ocy, outer_side, outer_angle)
    polys = []
    for i in range(len(centers)):
        cx, cy = centers[i]
        ang = angles[i]
        p = hex_vertices(cx, cy, 1.0, ang)
        polys.append(p)
    for p in polys:
        for (px, py) in p:
            if not point_in_convex_polygon(px, py, outer):
                return False
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if polygons_overlap_sat(polys[i], polys[j]):
                return False
    return True


def smooth_max(vals, beta=50.0):
    v = np.array(vals, dtype=float)
    v_max = np.max(v)
    if v_max > 1e10:
        return 1e10
    shifted = v - v_max
    return v_max + math.log(np.sum(np.exp(beta * shifted))) / beta


def topology_objective(params, topo_type):
    """Objective for topology-constrained parameters."""
    if time.time() > DEADLINE - 25:
        return 1e10
    if topo_type == 'ring12':
        scale, rot_in, rot_out = params
        if scale < 1.5 or scale > 4.0:
            return 1e10
        centers = []
        angles = []
        for k in range(12):
            theta = math.radians(rot_in + k * 30.0)
            centers.append([scale * math.cos(theta), scale * math.sin(theta)])
            angles.append(rot_in)
    elif topo_type == 'center_ring':
        r1, r2, rot_out = params
        if r1 < 1.8 or r2 < r1 + 0.5 or r2 > 5.0:
            return 1e10
        centers = [[0.0, 0.0]]
        angles = [0.0]
        for k in range(6):
            theta = math.radians(k * 60.0)
            centers.append([r1 * math.cos(theta), r1 * math.sin(theta)])
            angles.append(0.0)
        for k in range(5):
            theta = math.radians(k * 72.0 + 18.0)
            centers.append([r2 * math.cos(theta), r2 * math.sin(theta)])
            angles.append(0.0)
    elif topo_type == 'hex6_plus6':
        s1, s2, r1, r2, rot_out = params
        if s1 < 1.8 or s2 < s1 + 0.3 or s2 > 5.0:
            return 1e10
        centers = []
        angles = []
        for k in range(6):
            theta = math.radians(r1 + k * 60.0)
            centers.append([s1 * math.cos(theta), s1 * math.sin(theta)])
            angles.append(r1)
        for k in range(6):
            theta = math.radians(r2 + k * 60.0)
            centers.append([s2 * math.cos(theta), s2 * math.sin(theta)])
            angles.append(r2)
    elif topo_type == 'rect3x4':
        sx, sy, rot, rot_out = params
        if sx < 2.0 or sy < 1.8 or sx > 5.0 or sy > 5.0:
            return 1e10
        centers = []
        angles = []
        for i in range(3):
            for j in range(4):
                x = (i - 1) * sx
                y = (j - 1.5) * sy
                if i % 2 == 1:
                    x += sx / 2.0
                centers.append([x, y])
                angles.append(rot)
    else:
        return 1e10

    side = compute_min_outer_side(centers, angles, 0.0, 0.0, rot_out)

    # Overlap penalty using smooth SAT gap
    penalty = 0.0
    polys = [hex_vertices(c[0], c[1], 1.0, a) for c, a in zip(centers, angles)]
    for i in range(N):
        for j in range(i + 1, N):
            gap = sat_gap(polys[i], polys[j])
            if gap < 0.001:
                penalty += (0.001 - gap) ** 2
    return side + 500.0 * penalty


def free_placement_objective(x, beta=40.0):
    """Full free-placement objective with smooth overlap penalty."""
    if time.time() > DEADLINE - 25:
        return 1e10
    centers = [[x[3 * i], x[3 * i + 1]] for i in range(N)]
    angles = [x[3 * i + 2] for i in range(N)]
    ocx, ocy, otheta = x[3 * N], x[3 * N + 1], x[3 * N + 2]

    # Smooth outer side
    otheta_r = math.radians(otheta)
    projections = []
    for k in range(6):
        normal_angle = otheta_r + math.pi / 6 + k * math.pi / 3
        nx = math.cos(normal_angle)
        ny = math.sin(normal_angle)
        for i in range(N):
            cx, cy = centers[i]
            angle = math.radians(angles[i])
            for vi in range(6):
                vert_angle = angle + vi * math.pi / 3
                vx = cx + math.cos(vert_angle)
                vy = cy + math.sin(vert_angle)
                proj = (vx - ocx) * nx + (vy - ocy) * ny
                projections.append(proj)
    side = 2.0 * smooth_max(projections, beta) / math.sqrt(3)

    # Smooth overlap penalty
    penalty = 0.0
    polys = [hex_vertices(c[0], c[1], 1.0, a) for c, a in zip(centers, angles)]
    for i in range(N):
        for j in range(i + 1, N):
            gap = sat_gap(polys[i], polys[j])
            if gap < 0.002:
                penalty += (0.002 - gap) ** 2
    return side + 800.0 * penalty


def polish_outer(centers, angles, ocx, ocy, otheta, max_rounds=4):
    best_side = compute_min_outer_side(centers, angles, ocx, ocy, otheta)
    best_ocx, best_ocy, best_otheta = ocx, ocy, otheta
    for _ in range(max_rounds):
        if time.time() > DEADLINE - 10:
            break
        improved = False
        for scale in [0.05, 0.01, 0.002, 0.0005]:
            for dx in [-scale, 0, scale]:
                for dy in [-scale, 0, scale]:
                    for da in [-scale * 5, 0, scale * 5]:
                        side = compute_min_outer_side(centers, angles,
                                                      best_ocx + dx, best_ocy + dy,
                                                      best_otheta + da)
                        if side < best_side - 1e-12:
                            best_side = side
                            best_ocx += dx
                            best_ocy += dy
                            best_otheta += da
                            improved = True
        if not improved:
            break
    return best_ocx, best_ocy, best_otheta, best_side


def construct():
    t0 = time.time()
    best_side = float('inf')
    best_result = None

    def try_result(centers, angles, ocx, ocy, otheta):
        nonlocal best_side, best_result
        if time.time() > DEADLINE - 8:
            return
        side = compute_min_outer_side(centers, angles, ocx, ocy, otheta)
        if check_validity(centers, angles, ocx, ocy, side, otheta):
            if side < best_side:
                best_side = side
                best_result = (centers, angles, ocx, ocy, side, otheta)

    # Phase 1: Topology-constrained optimization
    topologies = [
        ('ring12', [2.5, 0.0, 0.0]),
        ('ring12', [2.5, 30.0, 30.0]),
        ('center_ring', [2.0, 3.8, 0.0]),
        ('center_ring', [2.1, 4.0, 30.0]),
        ('hex6_plus6', [2.0, 3.8, 0.0, 30.0, 0.0]),
        ('hex6_plus6', [2.1, 4.0, 30.0, 0.0, 30.0]),
        ('rect3x4', [2.2, 2.0, 0.0, 0.0]),
        ('rect3x4', [2.3, 2.1, 30.0, 30.0]),
    ]

    for topo_type, x0 in topologies:
        if time.time() > DEADLINE - 40:
            break
        try:
            res = minimize(topology_objective, x0, args=(topo_type,),
                           method='Nelder-Mead',
                           options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < 1e9:
                params = res.x
                if topo_type == 'ring12':
                    scale, rot_in, rot_out = params
                    centers = []
                    angles = []
                    for k in range(12):
                        theta = math.radians(rot_in + k * 30.0)
                        centers.append([scale * math.cos(theta), scale * math.sin(theta)])
                        angles.append(rot_in)
                elif topo_type == 'center_ring':
                    r1, r2, rot_out = params
                    centers = [[0.0, 0.0]]
                    angles = [0.0]
                    for k in range(6):
                        theta = math.radians(k * 60.0)
                        centers.append([r1 * math.cos(theta), r1 * math.sin(theta)])
                        angles.append(0.0)
                    for k in range(5):
                        theta = math.radians(k * 72.0 + 18.0)
                        centers.append([r2 * math.cos(theta), r2 * math.sin(theta)])
                        angles.append(0.0)
                elif topo_type == 'hex6_plus6':
                    s1, s2, r1, r2, rot_out = params
                    centers = []
                    angles = []
                    for k in range(6):
                        theta = math.radians(r1 + k * 60.0)
                        centers.append([s1 * math.cos(theta), s1 * math.sin(theta)])
                        angles.append(r1)
                    for k in range(6):
                        theta = math.radians(r2 + k * 60.0)
                        centers.append([s2 * math.cos(theta), s2 * math.sin(theta)])
                        angles.append(r2)
                elif topo_type == 'rect3x4':
                    sx, sy, rot, rot_out = params
                    centers = []
                    angles = []
                    for i in range(3):
                        for j in range(4):
                            x = (i - 1) * sx
                            y = (j - 1.5) * sy
                            if i % 2 == 1:
                                x += sx / 2.0
                            centers.append([x, y])
                            angles.append(rot)
                else:
                    continue

                ocx, ocy, otheta, side = polish_outer(centers, angles, 0.0, 0.0, rot_out)
                try_result(centers, angles, ocx, ocy, otheta)
        except Exception:
            pass

    # Phase 2: Free-placement refinement from best topology result
    if best_result is not None and time.time() < DEADLINE - 30:
        c0, a0, ocx0, ocy0, _, otheta0 = best_result
        x0 = []
        for i in range(N):
            x0.extend([c0[i][0], c0[i][1], a0[i]])
        x0.extend([ocx0, ocy0, otheta0])

        bounds = []
        for i in range(N):
            bounds.extend([(-6, 6), (-6, 6), (-30, 30)])
        bounds.extend([(-3, 3), (-3, 3), (-30, 30)])

        try:
            res = minimize(free_placement_objective, x0, method='SLSQP',
                           bounds=bounds, options={'maxiter': 150, 'ftol': 1e-12})
            if res.fun < 1e9:
                x = res.x
                centers = [[x[3 * i], x[3 * i + 1]] for i in range(N)]
                angles = [x[3 * i + 2] for i in range(N)]
                ocx, ocy, otheta = x[3 * N], x[3 * N + 1], x[3 * N + 2]
                ocx, ocy, otheta, side = polish_outer(centers, angles, ocx, ocy, otheta)
                try_result(centers, angles, ocx, ocy, otheta)
        except Exception:
            pass

    # Phase 3: Nelder-Mead free refinement (fewer params, more robust)
    if best_result is not None and time.time() < DEADLINE - 20:
        c0, a0, ocx0, ocy0, _, otheta0 = best_result
        # Use only positions and outer params (fix angles to 0 for NM)
        x0 = []
        for i in range(N):
            x0.extend([c0[i][0], c0[i][1]])
        x0.extend([ocx0, ocy0, otheta0])

        def nm_obj(x):
            if time.time() > DEADLINE - 10:
                return 1e10
            centers = [[x[2 * i], x[2 * i + 1]] for i in range(N)]
            angles = [a0[i] for i in range(N)]
            ocx, ocy, otheta = x[2 * N], x[2 * N + 1], x[2 * N + 2]
            side = compute_min_outer_side(centers, angles, ocx, ocy, otheta)
            penalty = 0.0
            polys = [hex_vertices(c[0], c[1], 1.0, a) for c, a in zip(centers, angles)]
            for i in range(N):
                for j in range(i + 1, N):
                    gap = sat_gap(polys[i], polys[j])
                    if gap < 0.001:
                        penalty += (0.001 - gap) ** 2
            return side + 500.0 * penalty

        try:
            res = minimize(nm_obj, x0, method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-7, 'fatol': 1e-9})
            if res.fun < 1e9:
                x = res.x
                centers = [[x[2 * i], x[2 * i + 1]] for i in range(N)]
                angles = [a0[i] for i in range(N)]
                ocx, ocy, otheta = x[2 * N], x[2 * N + 1], x[2 * N + 2]
                ocx, ocy, otheta, side = polish_outer(centers, angles, ocx, ocy, otheta)
                try_result(centers, angles, ocx, ocy, otheta)
        except Exception:
            pass

    # Phase 4: Random restarts with quick SLSQP
    if time.time() < DEADLINE - 15:
        rng = random.Random(123)
        for attempt in range(3):
            if time.time() > DEADLINE - 10:
                break
            centers = []
            angles = []
            for i in range(N):
                r = rng.uniform(0.5, 3.5)
                theta = rng.uniform(0, 2 * math.pi)
                centers.append([r * math.cos(theta), r * math.sin(theta)])
                angles.append(rng.choice([0.0, 30.0]))
            x0 = []
            for i in range(N):
                x0.extend([centers[i][0], centers[i][1], angles[i]])
            x0.extend([0.0, 0.0, 0.0])
            bounds = []
            for i in range(N):
                bounds.extend([(-5, 5), (-5, 5), (-30, 30)])
            bounds.extend([(-2, 2), (-2, 2), (-30, 30)])
            try:
                res = minimize(free_placement_objective, x0, method='SLSQP',
                               bounds=bounds, options={'maxiter': 80, 'ftol': 1e-10})
                if res.fun < 1e9:
                    x = res.x
                    c = [[x[3 * i], x[3 * i + 1]] for i in range(N)]
                    a = [x[3 * i + 2] for i in range(N)]
                    ocx, ocy, otheta = x[3 * N], x[3 * N + 1], x[3 * N + 2]
                    ocx, ocy, otheta, side = polish_outer(c, a, ocx, ocy, otheta)
                    try_result(c, a, ocx, ocy, otheta)
            except Exception:
                pass

    # Fallback
    if best_result is None:
        a = 2.002
        positions = [(0.0, 0.0)]
        for k in range(6):
            theta = math.radians(k * 60.0)
            positions.append((a * math.cos(theta), a * math.sin(theta)))
        for k in range(5):
            theta = math.radians(k * 72.0 + 18.0)
            positions.append((2.0 * a * math.cos(theta), 2.0 * a * math.sin(theta)))
        centers = [list(p) for p in positions]
        angles = [0.0] * 12
        outer_side = compute_min_outer_side(centers, angles, 0.0, 0.0, 0.0)
        inner = [(centers[i][0], centers[i][1], angles[i]) for i in range(N)]
        return inner, (0.0, 0.0), outer_side, 0.0

    c, a, ocx, ocy, side, otheta = best_result
    inner = [(c[i][0], c[i][1], a[i]) for i in range(N)]
    return inner, (ocx, ocy), side, otheta
