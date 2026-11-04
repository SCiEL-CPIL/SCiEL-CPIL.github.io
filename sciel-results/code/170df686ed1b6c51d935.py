"""
Pack 12 unit hexagons inside a regular hexagon, minimizing outer side length.
Topology-constrained parameterization + SAT-gap smooth separation + direct
outer-side minimization via Nelder-Mead and SLSQP.
"""
import math
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 12
DEADLINE = time.time() + 140.0  # safety margin before 150s kill


def hex_vertices(cx, cy, s, angle_deg):
    """Vertices of a regular hexagon centered at (cx,cy), side s, rotation angle_deg."""
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts.append((cx + s * math.cos(theta), cy + s * math.sin(theta)))
    return verts


def point_in_convex_polygon(px, py, poly):
    """Check if point (px,py) is inside convex polygon (list of (x,y) vertices)."""
    n = len(poly)
    sign = 0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if abs(cross) < 1e-12:
            continue
        s = 1 if cross > 0 else -1
        if sign == 0:
            sign = s
        elif s != sign:
            return False
    return True


def polygons_overlap_sat(A, B):
    """SAT: True if two convex polygons intersect (including touching)."""
    for poly in (A, B):
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            dx, dy = x2 - x1, y2 - y1
            nx, ny = -dy, dx
            if nx == 0 and ny == 0:
                continue
            minA = min(nx * px + ny * py for px, py in A)
            maxA = max(nx * px + ny * py for px, py in A)
            minB = min(nx * px + ny * py for px, py in B)
            maxB = max(nx * px + ny * py for px, py in B)
            if maxA < minB - 1e-9 or maxB < minA - 1e-9:
                return False
    return True


def sat_gap(A, B):
    """Compute maximum SAT gap between two polygons. Positive = separated, negative = overlap."""
    max_gap = -float('inf')
    for poly in (A, B):
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            dx, dy = x2 - x1, y2 - y1
            nx, ny = -dy, dx
            norm = math.sqrt(nx * nx + ny * ny)
            if norm < 1e-9:
                continue
            nx, ny = nx / norm, ny / norm
            projA = [nx * px + ny * py for px, py in A]
            projB = [nx * px + ny * py for px, py in B]
            minA, maxA = min(projA), max(projA)
            minB, maxB = min(projB), max(projB)
            if maxA < minB:
                gap = minB - maxA
            elif maxB < minA:
                gap = minA - maxB
            else:
                gap = -(min(maxA, maxB) - max(minA, minB))
            if gap > max_gap:
                max_gap = gap
    return max_gap


def compute_min_outer_side(centers, angles, ocx, ocy, otheta_deg):
    """Compute minimum outer hexagon side for given configuration."""
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


def smooth_max(values, beta=40.0):
    """Log-sum-exp based smooth max."""
    v = np.array(values, dtype=float)
    v_max = float(np.max(v))
    if v_max > 1e10:
        return 1e10
    shifted = v - v_max
    result = v_max + math.log(np.sum(np.exp(beta * shifted)) / beta) / beta
    return result


def compute_min_outer_side_smooth(centers, angles, ocx, ocy, otheta_deg, beta=40.0):
    """Smooth approximation of min outer side using log-sum-exp."""
    otheta = math.radians(otheta_deg)
    projections = []
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
                projections.append(proj)
    smooth_max_val = smooth_max(projections, beta)
    return 2.0 * smooth_max_val / math.sqrt(3)


def check_validity(centers, angles, ocx, ocy, outer_side, outer_angle):
    """Check all inner hexagons are inside outer and pairwise non-overlapping."""
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


def sat_gap_penalty(centers, angles, margin=0.001, beta=30.0):
    """Smooth penalty based on SAT gap: penalize when gap < margin."""
    n = len(centers)
    penalty = 0.0
    for i in range(n):
        pi = hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
        for j in range(i + 1, n):
            pj = hex_vertices(centers[j][0], centers[j][1], 1.0, angles[j])
            gap = sat_gap(pi, pj)
            if gap < margin:
                # Smooth penalty: use softplus-like function
                deficit = margin - gap
                penalty += math.exp(beta * deficit) / beta
    return penalty


# --- Topology builders ---

def build_ring12(scale, inner_rot, outer_rot):
    """12 hexagons in a ring of radius `scale`."""
    centers = []
    angles = []
    for k in range(12):
        theta = math.radians(inner_rot + k * 30.0)
        x = scale * math.cos(theta)
        y = scale * math.sin(theta)
        centers.append([x, y])
        angles.append(inner_rot)
    return centers, angles, (0.0, 0.0), outer_rot


def build_hex6_plus6(scale1, scale2, rot1, rot2, outer_rot):
    """6 hexagons at radius scale1 + 6 at radius scale2."""
    centers = []
    angles = []
    for k in range(6):
        theta = math.radians(rot1 + k * 60.0)
        x = scale1 * math.cos(theta)
        y = scale1 * math.sin(theta)
        centers.append([x, y])
        angles.append(rot1)
    for k in range(6):
        theta = math.radians(rot2 + k * 60.0)
        x = scale2 * math.cos(theta)
        y = scale2 * math.sin(theta)
        centers.append([x, y])
        angles.append(rot2)
    return centers, angles, (0.0, 0.0), outer_rot


def build_rect3x4(scale_x, scale_y, rot, outer_rot):
    """3x4 rectangular grid."""
    centers = []
    angles = []
    for i in range(3):
        for j in range(4):
            x = (i - 1) * scale_x
            y = (j - 1.5) * scale_y
            centers.append([x, y])
            angles.append(rot)
    return centers, angles, (0.0, 0.0), outer_rot


def build_hex_lattice(r1, r2, outer_rot):
    """1 center + 6 at r1 + 5 at r2."""
    centers = [[0.0, 0.0]]
    angles = [0.0]
    for k in range(6):
        theta = math.radians(k * 60.0)
        x = r1 * math.cos(theta)
        y = r1 * math.sin(theta)
        centers.append([x, y])
        angles.append(0.0)
    for k in range(5):
        theta = math.radians(k * 72.0 + 18.0)
        x = r2 * math.cos(theta)
        y = r2 * math.sin(theta)
        centers.append([x, y])
        angles.append(0.0)
    return centers, angles, (0.0, 0.0), outer_rot


def build_linear_chain(spacing, rot, outer_rot):
    """12 hexagons in a line."""
    centers = []
    angles = []
    for k in range(12):
        x = (k - 5.5) * spacing
        centers.append([x, 0.0])
        angles.append(rot)
    return centers, angles, (0.0, 0.0), outer_rot


def build_hexagonal_cluster(r1, r2, outer_rot):
    """1 center + 6 at r1 + 5 at r2 (not full ring)."""
    centers = [[0.0, 0.0]]
    angles = [0.0]
    for k in range(6):
        theta = math.radians(k * 60.0)
        x = r1 * math.cos(theta)
        y = r1 * math.sin(theta)
        centers.append([x, y])
        angles.append(0.0)
    for k in range(5):
        theta = math.radians(k * 72.0 + 18.0)
        x = r2 * math.cos(theta)
        y = r2 * math.sin(theta)
        centers.append([x, y])
        angles.append(0.0)
    return centers, angles, (0.0, 0.0), outer_rot


def topology_objective(params, topology, outer_rot, beta=40.0):
    """Objective: outer side + SAT gap penalty for topology-constrained config."""
    if topology == 'ring12':
        scale, inner_rot = params
        if scale < 1.0 or scale > 4.0:
            return 1000.0
        centers, angles, _, _ = build_ring12(scale, inner_rot, outer_rot)
    elif topology == 'hex6+6':
        scale1, scale2, rot1, rot2 = params
        if scale1 < 0.5 or scale2 < 0.5 or scale2 < scale1 + 0.5:
            return 1000.0
        centers, angles, _, _ = build_hex6_plus6(scale1, scale2, rot1, rot2, outer_rot)
    elif topology == 'rect3x4':
        sx, sy, rot = params
        if sx < 1.5 or sy < 1.5:
            return 1000.0
        centers, angles, _, _ = build_rect3x4(sx, sy, rot, outer_rot)
    elif topology == 'hex_lattice':
        r1, r2 = params
        if r1 < 1.5 or r2 < r1 + 1.0:
            return 1000.0
        centers, angles, _, _ = build_hex_lattice(r1, r2, outer_rot)
    elif topology == 'linear':
        spacing, rot = params
        if spacing < 1.5:
            return 1000.0
        centers, angles, _, _ = build_linear_chain(spacing, rot, outer_rot)
    elif topology == 'hex_cluster':
        r1, r2 = params
        if r1 < 1.5 or r2 < r1 + 1.0:
            return 1000.0
        centers, angles, _, _ = build_hexagonal_cluster(r1, r2, outer_rot)
    else:
        return 1000.0

    side = compute_min_outer_side_smooth(centers, angles, 0.0, 0.0, outer_rot, beta)
    gap_pen = sat_gap_penalty(centers, angles, margin=0.001, beta=20.0)
    return side + 50.0 * gap_pen


def optimize_topology(topology, x0, bounds, max_iters=200, beta=40.0, outer_rot=0.0):
    """Optimize a topology-constrained configuration."""
    try:
        result = minimize(
            lambda p: topology_objective(p, topology, outer_rot, beta),
            x0,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': max_iters, 'ftol': 1e-10}
        )
        if result.fun < 1000:
            params = result.x
            if topology == 'ring12':
                centers, angles, _, _ = build_ring12(params[0], params[1], outer_rot)
            elif topology == 'hex6+6':
                centers, angles, _, _ = build_hex6_plus6(params[0], params[1], params[2], params[3], outer_rot)
            elif topology == 'rect3x4':
                centers, angles, _, _ = build_rect3x4(params[0], params[1], params[2], outer_rot)
            elif topology == 'hex_lattice':
                centers, angles, _, _ = build_hex_lattice(params[0], params[1], outer_rot)
            elif topology == 'linear':
                centers, angles, _, _ = build_linear_chain(params[0], params[1], outer_rot)
            elif topology == 'hex_cluster':
                centers, angles, _, _ = build_hexagonal_cluster(params[0], params[1], outer_rot)
            else:
                return None
            side = compute_min_outer_side(centers, angles, 0.0, 0.0, outer_rot)
            return centers, angles, 0.0, 0.0, outer_rot, side
    except Exception:
        pass
    return None


def polish_outer(centers, angles, initial_ocx, initial_ocy, initial_otheta):
    """High-precision polish of outer hexagon parameters."""
    best_side = compute_min_outer_side(centers, angles, initial_ocx, initial_ocy, initial_otheta)
    best_ocx, best_ocy, best_otheta = initial_ocx, initial_ocy, initial_otheta

    scales = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    for scale in scales:
        improved = True
        while improved and time.time() < DEADLINE - 5:
            improved = False
            for dx in [-scale, 0, scale]:
                for dy in [-scale, 0, scale]:
                    for da_deg in [-scale * 10, 0, scale * 10]:
                        otheta = best_otheta + da_deg
                        side = compute_min_outer_side(centers, angles, best_ocx + dx, best_ocx + dy, otheta)
                        if side < best_side - 1e-14:
                            best_side = side
                            best_ocx += dx
                            best_ocy += dy
                            best_otheta = otheta
                            improved = True

    return best_ocx, best_ocy, best_otheta, best_side


def free_placement_objective(x, beta=40.0):
    """Objective with SAT gap penalty for free placement optimization."""
    if time.time() > DEADLINE - 10:
        return 1e10
    n = N
    centers = [[x[3 * i], x[3 * i + 1]] for i in range(n)]
    angles = [x[3 * i + 2] for i in range(n)]
    ocx, ocy, otheta = x[3 * n], x[3 * n + 1], x[3 * n + 2]
    side = compute_min_outer_side_smooth(centers, angles, ocx, ocy, otheta, beta)
    gap_pen = sat_gap_penalty(centers, angles, margin=0.001, beta=20.0)
    return side + 50.0 * gap_pen


def optimize_free_placement(centers, angles, max_iters=100, beta=40.0):
    """Optimize from a given starting configuration using SLSQP."""
    n = N
    x0 = []
    for i in range(n):
        x0.extend([centers[i][0], centers[i][1], angles[i]])
    x0.extend([0.0, 0.0, 0.0])

    bounds = []
    for i in range(n):
        bounds.extend([(-5, 5), (-5, 5), (-30, 30)])
    bounds.extend([(-2, 2), (-2, 2), (-30, 30)])

    try:
        result = minimize(free_placement_objective, x0, method='SLSQP',
                          bounds=bounds, options={'maxiter': max_iters, 'ftol': 1e-12})
        if result.fun < 1e9:
            c = [[result.x[3 * i], result.x[3 * i + 1]] for i in range(n)]
            a = [result.x[3 * i + 2] for i in range(n)]
            ocx, ocy, otheta = result.x[3 * n], result.x[3 * n + 1], result.x[3 * n + 2]
            side = compute_min_outer_side(c, a, ocx, ocy, otheta)
            return c, a, ocx, ocy, otheta, side
    except Exception:
        pass
    return None


def construct():
    t0 = time.time()

    best_result = None
    best_side = float('inf')

    def try_result(centers, angles, ocx, ocy, otheta):
        nonlocal best_result, best_side
        side = compute_min_outer_side(centers, angles, ocx, ocy, otheta)
        if check_validity(centers, angles, ocx, ocy, side, otheta):
            if side < best_side:
                best_side = side
                best_result = (centers, angles, ocx, ocy, side, otheta)

    # --- Topology-constrained optimization ---
    topology_configs = [
        ('ring12', [2.0, 0.0], [(1.0, 4.0), (-30, 30)]),
        ('hex6+6', [2.0, 3.5, 0.0, 30.0], [(0.5, 4.0), (1.5, 5.0), (-30, 30), (-30, 30)]),
        ('rect3x4', [2.0, 2.0, 0.0], [(1.5, 4.0), (1.5, 4.0), (-30, 30)]),
        ('hex_lattice', [2.0, 4.0], [(1.5, 4.0), (3.0, 6.0)]),
        ('linear', [2.0, 0.0], [(1.5, 4.0), (-30, 30)]),
        ('hex_cluster', [2.0, 4.0], [(1.5, 4.0), (3.0, 6.0)]),
    ]

    for outer_rot in [0.0, 15.0, 30.0]:
        if time.time() > DEADLINE - 60:
            break
        for topology, x0, bounds in topology_configs:
            if time.time() > DEADLINE - 55:
                break
            result = optimize_topology(topology, x0, bounds, max_iters=150, beta=40.0, outer_rot=outer_rot)
            if result is not None:
                c, a, ocx, ocy, otheta, side = result
                # Polish outer hexagon
                ocx, ocy, otheta, side = polish_outer(c, a, ocx, ocy, otheta)
                try_result(c, a, ocx, ocy, otheta)

    # --- Free placement from good topology starts ---
    if time.time() < DEADLINE - 40:
        # Start from best topology result or a good default
        if best_result is not None:
            c0, a0, _, _, _, _ = best_result
        else:
            c0, a0, _, _ = build_hex_lattice(2.0, 4.0, 0.0)

        result = optimize_free_placement(c0, a0, max_iters=100, beta=40.0)
        if result is not None:
            c, a, ocx, ocy, otheta, side = result
            ocx, ocy, otheta, side = polish_outer(c, a, ocx, ocy, otheta)
            try_result(c, a, ocx, ocy, otheta)

    # --- Additional topology variations with different rotations ---
    if time.time() < DEADLINE - 30:
        for outer_rot in [5.0, 10.0, 20.0, 25.0]:
            if time.time() > DEADLINE - 25:
                break
            for topology, x0, bounds in topology_configs[:3]:  # top 3 topologies
                if time.time() > DEADLINE - 25:
                    break
                result = optimize_topology(topology, x0, bounds, max_iters=100, beta=40.0, outer_rot=outer_rot)
                if result is not None:
                    c, a, ocx, ocy, otheta, side = result
                    ocx, ocy, otheta, side = polish_outer(c, a, ocx, ocy, otheta)
                    try_result(c, a, ocx, ocy, otheta)

    # If we found a good result, return it
    if best_result is not None:
        c, a, ocx, ocy, side, otheta = best_result
        inner = [(c[i][0], c[i][1], a[i]) for i in range(N)]
        outer_center = (ocx, ocy)
        return inner, outer_center, side, otheta

    # Fallback: simple hexagonal cluster with slight expansion to ensure disjointness
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
    outer_center = (0.0, 0.0)
    outer_side = compute_min_outer_side(centers, angles, 0.0, 0.0, 0.0)

    inner = [(centers[i][0], centers[i][1], angles[i]) for i in range(N)]
    return inner, outer_center, outer_side, 0.0
