"""Unit regular hexagons inside a regular hexagon – SAT-based packing optimizer."""
import numpy as np
import math
import time
import random

N = 11
DEADLINE = 150.0


def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon."""
    angles = np.radians(angle_deg) + np.arange(6) * np.pi / 3
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])


def _normal_vectors(angle_deg):
    """Return the 6 outward unit normal vectors of a hexagon with given rotation.
    For angle_deg=0, vertex at angle 0 → normals at 30°, 90°, 150°, 210°, 270°, 330°."""
    base = np.deg2rad(30) + np.deg2rad(angle_deg)
    angles = base + np.arange(6) * np.pi / 3
    return np.column_stack([np.cos(angles), np.sin(angles)])


def _support(proj_a, proj_b, side):
    """For a hexagon with side `side` and normals `proj_a` (projected onto some axis),
    return the support (max extent) of the hexagon along that axis.
    Actually, we compute: for each vertex, dot with axis; take max and min.
    Here we use the 6 vertices implicitly via the 6 normals' dot products with axis.
    For a regular hexagon with circumradius R=side, the support function along unit vector u
    is R * max_i cos(theta_i - phi) where theta_i are vertex angles, phi is angle of u.
    Equivalent: max over the 6 vertex directions.
    But simpler: just compute dot products of the 6 vertices with u and take max.
    Since we only have normals, let's precompute vertex directions instead.
    """
    pass  # We'll use a different approach below.


def _hexagon_support(cx, cy, side, angle_deg, ax, ay):
    """Return (min_proj, max_proj) of the hexagon's vertices onto axis (ax, ay)."""
    angles = np.radians(angle_deg) + np.arange(6) * np.pi / 3
    vx = cx + side * np.cos(angles)
    vy = cy + side * np.sin(angles)
    proj = vx * ax + vy * ay
    return proj.min(), proj.max()


def _sat_overlap(A, B):
    """
    SAT test for two unit hexagons.
    A = (cx1, cy1, ang1), B = (cx2, cy2, ang2)
    Returns (overlap: bool, min_penetration: float, contact_normal: (nx, ny) or None)
    """
    cx1, cy1, a1 = A
    cx2, cy2, a2 = B

    # Quick reject
    dist = math.hypot(cx2 - cx1, cy2 - cy1)
    if dist > 2.0 + 1e-9:
        return False, 0.0, None

    # Gather all candidate axes: 6 normals from A, 6 from B
    normals_a = _normal_vectors(a1)
    normals_b = _normal_vectors(a2)

    min_pen = float('inf')
    best_normal = None

    for i in range(6):
        for (nx, ny) in [normals_a[i], normals_b[i]]:
            # Project both hexagons onto this axis
            min_a, max_a = _hexagon_support(cx1, cy1, 1.0, a1, nx, ny)
            min_b, max_b = _hexagon_support(cx2, cy2, 1.0, a2, nx, ny)

            # Check overlap
            if max_a < min_b or max_b < min_a:
                return False, 0.0, None  # Separating axis found

            # Compute penetration depth
            overlap_a = min_a - max_b  # negative means A starts before B ends
            overlap_b = min_b - max_a  # negative means B starts before A ends
            pen = -min(overlap_a, overlap_b)  # positive penetration
            if pen < min_pen:
                min_pen = pen
                best_normal = (nx, ny)

    # Determine direction of contact normal (from A toward B or vice versa)
    dx = cx2 - cx1
    dy = cy2 - cy1
    if best_normal is not None:
        nx, ny = best_normal
        if nx * dx + ny * dy < 0:
            nx, ny = -nx, -ny

    return True, min_pen, (best_normal[0], best_normal[1]) if best_normal else None


def _point_in_hex(px, py, cx, cy, side, angle_deg):
    """Check if point is inside hexagon."""
    angle = np.radians(angle_deg)
    dx, dy = px - cx, py - cy
    ca, sa = math.cos(-angle), math.sin(-angle)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    # Check 6 half-planes
    verts = np.column_stack([np.cos(np.arange(6) * np.pi / 3) * side,
                              np.sin(np.arange(6) * np.pi / 3) * side])
    for i in range(6):
        v1 = verts[i]
        v2 = verts[(i + 1) % 6]
        ex, ey = v2[0] - v1[0], v2[1] - v1[1]
        nx, ny = -ey, ex
        if (lx - v1[0]) * nx + (ly - v1[1]) * ny < -1e-10:
            return False
    return True


def _hex_in_hex(inner_cx, inner_cy, inner_side, inner_angle,
                outer_cx, outer_cy, outer_side, outer_angle):
    """Check if inner hexagon fits entirely inside outer hexagon."""
    iv = _hex_vertices(inner_cx, inner_cy, inner_side, inner_angle)
    for i in range(6):
        if not _point_in_hex(iv[i, 0], iv[i, 1], outer_cx, outer_cy, outer_side, outer_angle):
            return False
    return True


def _min_outer_side(positions, center, outer_angle):
    """Binary search for minimum outer side length that contains all inner hexagons."""
    lo, hi = 1.0, 30.0
    for _ in range(25):
        mid = (lo + hi) / 2
        fits = True
        for cx, cy, ang in positions:
            if not _hex_in_hex(cx, cy, 1.0, ang, center[0], center[1], mid, outer_angle):
                fits = False
                break
        if fits:
            hi = mid
        else:
            lo = mid
    return hi


def _compute_center(positions):
    """Compute center of bounding box of all hexagon vertices."""
    all_v = np.vstack([_hex_vertices(cx, cy, 1.0, ang) for cx, cy, ang in positions])
    return (float(np.mean(all_v[:, 0])), float(np.mean(all_v[:, 1])))


def _initial_packing(n):
    """Hexagonal packing: center + ring1 (6) + partial ring2."""
    positions = [(0.0, 0.0, 0.0)]
    for k in range(6):
        ang = k * np.pi / 3
        positions.append((math.sqrt(3) * math.cos(ang), math.sqrt(3) * math.sin(ang), 0.0))
    for k in range(4):
        ang = k * 2 * np.pi / 3
        positions.append((3.0 * math.cos(ang), 3.0 * math.sin(ang), 0.0))
    return positions[:n]


def construct():
    start = time.time()
    deadline = start + DEADLINE - 10.0  # 10s safety margin

    random.seed(42)
    np.random.seed(42)

    positions = _initial_packing(N)
    center = _compute_center(positions)
    best_side = _min_outer_side(positions, center, 0.0)
    best_positions = [list(p) for p in positions]
    best_center = center

    # Simulated Annealing
    T = 0.3
    T_min = 0.001
    cooling = 0.997

    while time.time() < deadline and T > T_min:
        # Generate candidate perturbation
        i = random.randint(0, len(positions) - 1)
        cx, cy, ang = positions[i]
        dcx = (random.random() - 0.5) * 0.4
        dcy = (random.random() - 0.5) * 0.4
        dang = (random.random() - 0.5) * 20.0
        new_positions = [list(p) for p in positions]
        new_positions[i] = [cx + dcx, cy + dcy, ang + dang]

        # Check for overlaps with SAT
        has_overlap = False
        max_pen = 0.0
        for a in range(len(new_positions)):
            for b in range(a + 1, len(new_positions)):
                ov, pen, _ = _sat_overlap(tuple(new_positions[a]), tuple(new_positions[b]))
                if ov and pen > 1e-8:
                    has_overlap = True
                    max_pen = max(max_pen, pen)
                    break
            if has_overlap:
                break

        if has_overlap:
            # Reject overlapping configurations (or accept with high penalty)
            penalty = max_pen * 10.0
            # Compute outer side for this config anyway
            nc = _compute_center(new_positions)
            ns = _min_outer_side(new_positions, nc, 0.0)
            new_cost = ns + penalty
            delta_cost = new_cost - best_side
            if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
                positions = new_positions
                center = nc
                best_side = ns
                best_positions = [list(p) for p in positions]
                best_center = center
            T *= cooling
            continue

        # No overlap – compute new outer side
        nc = _compute_center(new_positions)
        ns = _min_outer_side(new_positions, nc, 0.0)
        delta_cost = ns - best_side
        if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
            positions = new_positions
            center = nc
            best_side = ns
            best_positions = [list(p) for p in positions]
            best_center = center

        T *= cooling

    # Final local refinement: gradient-free fine-tune
    for _ in range(200):
        if time.time() > deadline:
            break
        improved = False
        for i in range(len(best_positions)):
            cx, cy, ang = best_positions[i]
            for dx, dy in [(0.02, 0), (-0.02, 0), (0, 0.02), (0, -0.02)]:
                trial = [list(p) for p in best_positions]
                trial[i][0] += dx
                trial[i][1] += dy
                # Quick overlap check
                ok = True
                for j in range(len(trial)):
                    if i != j:
                        ov, _, _ = _sat_overlap(tuple(trial[i]), tuple(trial[j]))
                        if ov:
                            ok = False
                            break
                if not ok:
                    continue
                tc = _compute_center(trial)
                ts = _min_outer_side(trial, tc, 0.0)
                if ts < best_side - 1e-4:
                    best_positions = trial
                    best_center = tc
                    best_side = ts
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    # Final precise computation
    all_v = np.vstack([_hex_vertices(cx, cy, 1.0, ang) for cx, cy, ang in best_positions])
    cx_final = float(np.mean(all_v[:, 0]))
    cy_final = float(np.mean(all_v[:, 1]))

    lo, hi = 1.0, 20.0
    for _ in range(30):
        mid = (lo + hi) / 2
        fits = True
        for cx, cy, ang in best_positions:
            if not _hex_in_hex(cx, cy, 1.0, ang, cx_final, cy_final, mid, 0.0):
                fits = False
                break
        if fits:
            hi = mid
        else:
            lo = mid

    return [list(p) for p in best_positions], (cx_final, cy_final), hi, 0.0
# EVOLVE-BLOCK-END
