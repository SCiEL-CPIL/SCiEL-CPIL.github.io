"""Packing unit hexagons inside a regular hexagon."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)
DEADLINE = 135.0  # safety margin before 150s kill


def logsumexp(a, b=1.0):
    """Stable log-sum-exp for array a with temperature b."""
    a = np.asarray(a, dtype=np.float64)
    amax = np.max(a)
    if not np.isfinite(amax):
        return amax
    return amax + math.log(np.sum(np.exp((a - amax) / b))) * b


def get_hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon."""
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        vx = cx + side * math.cos(theta)
        vy = cy + side * math.sin(theta)
        verts.append((vx, vy))
    return verts


def compute_required_side(inner, outer_center, outer_angle_deg):
    """
    Compute the minimum outer side length to contain all inner hexagons.
    inner: list of (x, y, angle_deg)
    outer_center: (x, y)
    outer_angle_deg: rotation of the outer hexagon
    """
    if not inner:
        return 0.0

    ox, oy = outer_center
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))

    max_projection = 0.0

    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))

    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a

            for nx, ny in normals:
                proj = lx * nx + ly * ny
                if proj > max_projection:
                    max_projection = proj

    return 2.0 * max_projection / SQRT3


def sat_overlap_check(v1, v2, eps=1e-9):
    """
    Check if two convex polygons (defined by their vertices) overlap using SAT.
    v1, v2: lists of (x, y) tuples.
    Returns True if they overlap (including touching).
    """
    axes = []

    for i in range(len(v1)):
        p1 = v1[i]
        p2 = v1[(i + 1) % len(v1)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0] ** 2 + normal[1] ** 2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))

    for i in range(len(v2)):
        p1 = v2[i]
        p2 = v2[(i + 1) % len(v2)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0] ** 2 + normal[1] ** 2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))

    for axis in axes:
        min1 = min(p[0] * axis[0] + p[1] * axis[1] for p in v1)
        max1 = max(p[0] * axis[0] + p[1] * axis[1] for p in v1)
        min2 = min(p[0] * axis[0] + p[1] * axis[1] for p in v2)
        max2 = max(p[0] * axis[0] + p[1] * axis[1] for p in v2)

        if max1 < min2 + eps or max2 < min1 + eps:
            return False

    return True


def check_all_disjoint(inner, eps=1e-6):
    """Check if all inner hexagons are pairwise disjoint using SAT."""
    n = len(inner)
    for i in range(n):
        for j in range(i + 1, n):
            vi = get_hex_vertices(inner[i][0], inner[i][1], 1.0, math.radians(inner[i][2]))
            vj = get_hex_vertices(inner[j][0], inner[j][1], 1.0, math.radians(inner[j][2]))
            if sat_overlap_check(vi, vj, eps):
                return False
    return True


def check_inside_outer(inner, outer_center, outer_side, outer_angle_deg, eps=1e-6):
    """Check all inner hexagons are inside the outer hexagon."""
    ox, oy = outer_center
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))

    apothem = outer_side * SQRT3 / 2.0

    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a

            for i in range(6):
                theta = math.radians(30 + i * 60)
                nx = math.cos(theta)
                ny = math.sin(theta)
                if lx * nx + ly * ny > apothem + eps:
                    return False
    return True


def axial_to_xy(q, r, scale=1.0):
    """Convert axial coordinates to Cartesian for unit hexagons."""
    x = 1.5 * scale * q
    y = SQRT3 * scale * (r + q / 2.0)
    return x, y


def generate_polyhexes_topologies():
    """
    Generate several distinct 11-hexagon topologies (polyhexes) as sets of axial coords.
    We focus on compact shapes that minimize the bounding hexagon.
    """
    topologies = []

    # Topology 1: Standard hexagonal cluster (radius 2 minus 5 corners)
    # Center + 6 ring + 4 from second ring
    t1 = set()
    t1.add((0, 0))
    for i in range(6):
        dirs = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
        t1.add(dirs[i])
    # Add 4 more from second ring (choose to be compact)
    t1.add((2, 0))
    t1.add((1, 1))
    t1.add((0, 2))
    t1.add((-1, 2))
    topologies.append(t1)

    # Topology 2: More linear/elongated
    t2 = set()
    t2.add((0, 0))
    t2.add((1, 0))
    t2.add((2, 0))
    t2.add((3, 0))
    t2.add((4, 0))
    t2.add((0, 1))
    t2.add((1, 1))
    t2.add((2, 1))
    t2.add((3, 1))
    t2.add((1, -1))
    t2.add((2, -1))
    topologies.append(t2)

    # Topology 3: Compact diamond-like
    t3 = set()
    t3.add((0, 0))
    t3.add((1, 0))
    t3.add((0, 1))
    t3.add((-1, 1))
    t3.add((-1, 0))
    t3.add((0, -1))
    t3.add((1, -1))
    t3.add((2, 0))
    t3.add((1, 1))
    t3.add((0, 2))
    t3.add((-1, 2))
    topologies.append(t3)

    # Topology 4: Two rows of 5 + 1
    t4 = set()
    for q in range(5):
        t4.add((q, 0))
    for q in range(5):
        t4.add((q, 1))
    t4.add((2, -1))
    topologies.append(t4)

    # Topology 5: Cross shape
    t5 = set()
    t5.add((0, 0))
    t5.add((1, 0))
    t5.add((2, 0))
    t5.add((3, 0))
    t5.add((1, 1))
    t5.add((2, 1))
    t5.add((1, -1))
    t5.add((2, -1))
    t5.add((0, 1))
    t5.add((3, 1))
    t5.add((0, -1))
    topologies.append(t5)

    # Topology 6: Full hexagon radius 2 (19 cells) minus 8 to get 11
    # Actually let's do a different compact shape
    t6 = set()
    # 3-4-3-1 pattern
    for q in range(3):
        t6.add((q, 0))
    for q in range(4):
        t6.add((q, 1))
    for q in range(3):
        t6.add((q, 2))
    t6.add((1, -1))
    topologies.append(t6)

    # Topology 7: 4-3-4 pattern
    t7 = set()
    for q in range(4):
        t7.add((q, 0))
    for q in range(3):
        t7.add((q, 1))
    for q in range(4):
        t7.add((q, 2))
    # That's 11
    topologies.append(t7)

    # Topology 8: 5-3-3 pattern
    t8 = set()
    for q in range(5):
        t8.add((q, 0))
    for q in range(3):
        t8.add((q, 1))
    for q in range(3):
        t8.add((q, 2))
    topologies.append(t8)

    return topologies


def compute_optimal_enclosing_hex(inner_list, n_angles=72):
    """
    For a fixed set of inner hexagons (with fixed rotations), find the optimal
    outer hexagon center and rotation angle that minimizes the required side length.
    
    Uses a 1D sweep over outer angle, and for each angle, computes the optimal center
    as the centroid of the projected extremes.
    """
    ox_best = 0.0
    oy_best = 0.0
    angle_best = 0.0
    side_best = float('inf')

    # Precompute all vertices
    all_verts = []
    for (ix, iy, iangle) in inner_list:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        all_verts.extend(verts)

    # Sweep over outer hexagon rotation angle
    for angle_deg in range(0, 180, 180 // n_angles):
        if time.time() - DEADLINE < 5:
            break

        cos_a = math.cos(math.radians(angle_deg))
        sin_a = math.sin(math.radians(angle_deg))

        # Project all vertices into the rotated frame
        # The outer hexagon has normals at 30, 90, 150, 210, 270, 330 degrees in its local frame
        # In the rotated frame, we need to find the center that minimizes the max projection

        # For a regular hexagon with apothem a, the constraint is:
        # For each of 6 directions (normals), the projection of all points must be within [-a, a]
        # The optimal center minimizes the maximum over all (point, direction) of |proj|

        # Actually, for a hexagon, the six half-planes are:
        # n_i · (p - c) <= apothem for i = 0..5
        # where n_i are the outward normals at 30+60*i degrees

        # To minimize the apothem (and thus side length), we want to minimize:
        # max over all vertices v and normals n of (n · (v - c))
        # = max over all (v, n) of (n·v - n·c)

        # For a given angle, this is a linear programming problem in c.
        # But we can solve it more simply: the optimal c is the one that minimizes
        # the maximum support function.

        # For a regular hexagon, the six constraints in the rotated frame are:
        # Let's work in the frame where the hexagon is axis-aligned.
        # The normals in the hexagon's local frame are at 30, 90, 150, 210, 270, 330 degrees.

        normals_local = []
        for i in range(6):
            theta = math.radians(30 + i * 60)
            normals_local.append((math.cos(theta), math.sin(theta)))

        # Project all vertices into the outer hexagon's local frame
        # Local frame: rotate by -angle_deg
        local_verts = []
        for (vx, vy) in all_verts:
            lx = vx * cos_a + vy * sin_a
            ly = -vx * sin_a + vy * cos_a
            local_verts.append((lx, ly))

        # For each normal, find the max and min projection
        max_projs = []
        min_projs = []
        for (nx, ny) in normals_local:
            max_p = max(lx * nx + ly * ny for (lx, ly) in local_verts)
            min_p = min(lx * nx + ly * ny for (lx, ly) in local_verts)
            max_projs.append(max_p)
            min_projs.append(min_p)

        # The hexagon must contain all points, so:
        # For each normal i: max_p_i - c_local_i <= apothem AND -min_p_i + c_local_i <= apothem
        # i.e., c_local_i >= max_p_i - apothem AND c_local_i <= -min_p_i + apothem
        # So we need: max_p_i - (-min_p_i) <= 2 * apothem for all i
        # apothem >= max over i of (max_p_i - min_p_i) / 2

        # But wait, the center c_local is shared across all constraints.
        # The constraint is: for all i, max_p_i - c_i <= a AND c_i - min_p_i <= a
        # where c_i = c_local · n_i
        # This means: c_local must satisfy for all i:
        #   max_p_i - a <= c_local · n_i <= min_p_i + a  ... wait, let me redo this

        # Actually: n_i · (v - c) <= a for all v, all i
        # => n_i · v - n_i · c <= a for all v, all i
        # => n_i · v <= a + n_i · c for all v, all i
        # => max_v (n_i · v) <= a + n_i · c for all i
        # => n_i · c >= max_v (n_i · v) - a for all i

        # Also: -n_i · (v - c) <= a for all v, all i (the opposite face)
        # => -n_i · v + n_i · c <= a
        # => n_i · c <= a + n_i · v for all v
        # => n_i · c <= a + min_v (n_i · v) for all i

        # So: max_v(n_i·v) - a <= n_i·c <= min_v(n_i·v) + a for all i

        # Let M_i = max_v(n_i·v), m_i = min_v(n_i·v)
        # We need: M_i - a <= n_i·c <= m_i + a for all i
        # This requires M_i - a <= m_i + a, i.e., a >= (M_i - m_i)/2 for all i
        # And c must be in the intersection of half-planes.

        # The minimum a is max_i (M_i - m_i) / 2
        # And c exists if the half-planes are consistent (they always are for a >= a_min)

        a_min = max((max_projs[i] - min_projs[i]) / 2.0 for i in range(6))

        # Now find c_local that satisfies all constraints with a = a_min
        # We can use the center of the feasible region, or just use the midpoint
        # For simplicity, use c_local such that n_i · c_local = (M_i + m_i) / 2 for all i
        # This is an overdetermined system; solve by least squares

        # Build the system: n_i · c_local = (M_i + m_i) / 2 for i = 0..5
        A = np.array([[nx, ny] for (nx, ny) in normals_local], dtype=np.float64)
        b = np.array([(max_projs[i] + min_projs[i]) / 2.0 for i in range(6)], dtype=np.float64)

        # Least squares solution
        c_local, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        cx_local, cy_local = c_local

        # Convert back to global frame
        cx_global = cx_local * cos_a - cy_local * sin_a
        cy_global = cx_local * sin_a + cy_local * cos_a

        side = 2.0 * a_min / SQRT3

        if side < side_best:
            side_best = side
            ox_best = cx_global
            oy_best = cy_global
            angle_best = float(angle_deg)

    return ox_best, oy_best, angle_best, side_best


def construct():
    start_time = time.time()

    # Generate multiple topologies
    topologies = generate_polyhexes_topologies()

    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0

    for topo in topologies:
        if time.time() - start_time > DEADLINE - 20:
            break

        # Convert axial coords to Cartesian, all with 0 rotation
        inner_list = []
        for (q, r) in sorted(topo):
            x, y = axial_to_xy(q, r, scale=1.0)
            inner_list.append([x, y, 0.0])

        # Find optimal enclosing hexagon
        try:
            ox, oy, angle, side = compute_optimal_enclosing_hex(inner_list, n_angles=36)

            # Verify
            if check_all_disjoint(inner_list) and check_inside_outer(inner_list, (ox, oy), side, angle):
                if side < best_side:
                    best_side = side
                    best_inner = [list(x) for x in inner_list]
                    best_center = (ox, oy)
                    best_angle = angle
        except Exception:
            pass

    # Fallback: use the original hexagonal cluster if nothing worked
    if best_inner is None or not np.isfinite(best_side):
        axial_coords = [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, 0), (1, 1), (0, 2), (-1, 2),
        ]
        centers = []
        for q, r in axial_coords:
            x, y = axial_to_xy(q, r, scale=1.0)
            centers.append((x, y))
        best_inner = [[x, y, 0.0] for x, y in centers]
        ox, oy, angle, side = compute_optimal_enclosing_hex(best_inner, n_angles=36)
        best_center = (ox, oy)
        best_angle = angle
        best_side = side

    # Final polish: try to optimize the outer hexagon parameters more finely
    if time.time() - start_time < DEADLINE - 5:
        try:
            def obj(p):
                return compute_required_side(best_inner, (p[0], p[1]), p[2])

            res = minimize(
                obj,
                [best_center[0], best_center[1], best_angle],
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'maxfev': 2000}
            )
            if res.fun < best_side:
                best_center = (res.x[0], res.x[1])
                best_angle = res.x[2]
                best_side = res.fun
        except Exception:
            pass

    # Final verification
    if not check_all_disjoint(best_inner) or not check_inside_outer(best_inner, best_center, best_side, best_angle):
        # Fallback with scale
        axial_coords = [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, 0), (1, 1), (0, 2), (-1, 2),
        ]
        centers = []
        for q, r in axial_coords:
            x, y = axial_to_xy(q, r, scale=1.05)
            centers.append((x, y))
        best_inner = [[x, y, 0.0] for x, y in centers]
        best_center = (0.0, 0.0)
        best_angle = 0.0
        best_side = compute_required_side(best_inner, best_center, best_angle)

    return best_inner, best_center, best_side, best_angle
