"""Packing unit hexagons inside a regular hexagon."""
import math
import time
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
SQRT3 = math.sqrt(3)
APOTHEM = SQRT3 / 2.0
DEADLINE = 140.0  # safety margin before 150s kill

START_TIME = time.time()


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


def compute_required_side_smooth(inner, outer_center, outer_angle_deg, temp=0.1):
    """
    Smooth approximation of compute_required_side using log-sum-exp.
    """
    if not inner:
        return 0.0

    ox, oy = outer_center
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))

    all_projections = []

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
                all_projections.append(proj)

    if not all_projections:
        return 0.0

    arr = np.array(all_projections, dtype=np.float64)
    amax = np.max(arr)
    if not np.isfinite(amax):
        return 2.0 * amax / SQRT3

    smooth_max = amax + temp * math.log(np.sum(np.exp((arr - amax) / temp)))

    return 2.0 * smooth_max / SQRT3


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


def axial_to_cartesian(axial_list, scale=1.0):
    """Convert axial coordinates to Cartesian. Returns list of (x, y)."""
    pts = []
    for q, r in axial_list:
        x = SQRT3 * scale * (q + r / 2.0)
        y = 1.5 * scale * r
        pts.append((x, y))
    return pts


def generate_polyhexes(n_cells):
    """
    Generate all n-cell polyhexes via BFS on axial grid.
    Returns list of frozensets of axial coordinates.
    Uses canonical form to avoid duplicates.
    """
    if n_cells == 1:
        return [frozenset([(0, 0)])]

    # Generate (n-1)-cell polyhexes
    prev = generate_polyhexes(n_cells - 1)

    seen = set()
    results = []

    for poly in prev:
        # Find all cells adjacent to the polyhex
        neighbors = set()
        for q, r in poly:
            for dq, dr in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]:
                nq, nr = q + dq, r + dr
                if (nq, nr) not in poly:
                    neighbors.add((nq, nr))

        for nq, nr in neighbors:
            new_poly = poly | {(nq, nr)}

            # Canonicalize: translate so min-q is 0, then min-r is 0
            min_q = min(q for q, r in new_poly)
            min_r = min(r for q, r in new_poly if q == min_q)
            canonical = frozenset((q - min_q, r - min_r) for q, r in new_poly)

            if canonical not in seen:
                seen.add(canonical)
                results.append(canonical)

    return results


def calc_S_lp(pts, beta):
    """
    Use LP to find optimal center (cx, cy) and minimum apothem D for given points and outer rotation beta.
    Returns (S, cx, cy, D) or (inf, None, None, None) if infeasible.
    S = (D + APOTHEM) / APOTHEM is the side length ratio.
    """
    normals = []
    for k in range(6):
        ang = beta + math.pi / 6.0 + k * math.pi / 3.0
        normals.append((math.cos(ang), math.sin(ang)))

    c_obj = np.array([0.0, 0.0, 1.0])
    A_ub = []
    b_ub = []

    for nx, ny in normals:
        for px, py in pts:
            # -nx*cx - ny*cy - D <= -(px*nx + py*ny)  =>  nx*cx + ny*cy + D >= px*nx + py*ny
            A_ub.append([-nx, -ny, -1.0])
            b_ub.append(-(px * nx + py * ny))
            # nx*cx + ny*cy - D <= px*nx + py*ny  =>  -nx*cx - ny*cy + D >= -(px*nx + py*ny)
            A_ub.append([nx, ny, -1.0])
            b_ub.append(px * nx + py * ny)

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    bounds = [(None, None), (None, None), (0, None)]

    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.status == 0:
            D = res.x[2]
            cx = res.x[0]
            cy = res.x[1]
            S = (D + APOTHEM) / APOTHEM
            return S, cx, cy, D
        else:
            return float('inf'), None, None, None
    except Exception:
        return float('inf'), None, None, None


def smooth_calc_S(pts, beta, cx, cy, temperature=0.1):
    """
    Smooth approximation of S for a given configuration and outer rotation.
    pts: (N, 2) array of inner hexagon centers (in local frame relative to composite origin)
    beta: outer rotation angle in radians
    cx, cy: outer center
    """
    normals = np.array([
        (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
         math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
        for k in range(6)
    ])

    pts_centered = pts - np.array([cx, cy])
    projs = pts_centered @ normals.T  # (N, 6)

    # For each point, the max projection in any direction determines the constraint
    # We need max over all points and all 6 normals of (proj + APOTHEM) essentially
    # The required D is max over all i, k of (pts_centered[i] . normals[k] + APOTHEM)
    # But actually the constraint is: for each face normal n, max_i (p_i . n) <= D - APOTHEM
    # Wait, the standard form: the hexagon with center (cx,cy) and apothem D contains p_i if
    # for all normals n: (p_i - center) . n <= D
    # So D >= max_i,k ( (p_i - center) . n_k )
    # Also D >= -min_i,k ( (p_i - center) . n_k )  ... no, the 6 normals cover all 6 sides
    # Actually for a regular hexagon, the 6 outward normals n_k satisfy:
    # point is inside iff (p - c) . n_k <= D for all k=0..5
    # So D >= max_k max_i ( (p_i - c) . n_k )
    
    all_projs = projs.ravel()  # all N*6 projections
    # D must be >= max(all_projs)
    # S = (D + APOTHEM) / APOTHEM where D is the apothem of the OUTER hexagon
    # Wait, I need to be careful. The inner hexagons have apothem APOTHEM.
    # The outer hexagon has apothem D_outer.
    # Constraint: for each outer face normal n_k, and each inner vertex v: (v - c) . n_k <= D_outer
    # But here pts are centers, not vertices. This is a simplification.
    
    # For the smooth objective, let's use a simpler model:
    # Treat each inner hexagon as a point with "radius" APOTHEM in each normal direction.
    # Then D_outer >= max_k ( max_i ( (p_i - c) . n_k ) + APOTHEM )
    
    # Actually this is not quite right either. Let me just use the vertex-based approach for exact,
    # and for smooth, use a proxy.
    
    max_val = np.max(all_projs)
    if not np.isfinite(max_val):
        return float('inf')
    
    smooth_max = max_val + temperature * math.log(np.sum(np.exp((all_projs - max_val) / temperature)))
    
    # D_outer = smooth_max + APOTHEM (approximation)
    D_outer = smooth_max + APOTHEM
    S = D_outer / APOTHEM
    
    return S


def construct():
    start_time = time.time()

    # Generate all 11-cell polyhexes
    # This might be slow, so we'll use a pre-computed set of promising topologies
    # if generation takes too long

    # Pre-defined candidate topologies (axial coordinates)
    candidates = []

    # Hexagonal cluster (center + ring + partial outer)
    axial_1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    candidates.append(axial_1)

    axial_2 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_2)

    axial_3 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1)
    ]
    candidates.append(axial_3)

    axial_4 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2)
    ]
    candidates.append(axial_4)

    axial_5 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_5)

    axial_6 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    candidates.append(axial_6)

    axial_7 = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_7)

    axial_8 = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)
    ]
    candidates.append(axial_8)

    axial_9 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (-2, 2), (-1, 3), (1, -2)
    ]
    candidates.append(axial_9)

    axial_10 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_10)

    axial_11 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (1, 3)
    ]
    candidates.append(axial_11)

    axial_12 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3),
        (0, 4)
    ]
    candidates.append(axial_12)

    axial_13 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (1, 2)
    ]
    candidates.append(axial_13)

    axial_14 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 1), (-1, -1)
    ]
    candidates.append(axial_14)

    axial_15 = [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1), (0, -1)
    ]
    candidates.append(axial_15)

    axial_16 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3), (3, 2)
    ]
    candidates.append(axial_16)

    axial_17 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (1, 2), (-1, 2), (0, 3)
    ]
    candidates.append(axial_17)

    axial_18 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2), (3, 2)
    ]
    candidates.append(axial_18)

    axial_19 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3), (3, 3)
    ]
    candidates.append(axial_19)

    axial_20 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 1), (-1, 1)
    ]
    candidates.append(axial_20)

    axial_21 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (1, 2), (2, 2), (3, 2)
    ]
    candidates.append(axial_21)

    axial_22 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3)
    ]
    candidates.append(axial_22)

    axial_23 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_23)

    axial_24 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3),
        (0, 4)
    ]
    candidates.append(axial_24)

    axial_25 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_25)

    axial_26 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_26)

    axial_27 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (1, -2)
    ]
    candidates.append(axial_27)

    axial_28 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3), (2, 3),
        (1, 4), (2, 4)
    ]
    candidates.append(axial_28)

    # Try to generate more polyhexes if time permits
    try:
        if time.time() - start_time < DEADLINE - 60:
            generated = generate_polyhexes(11)
            # Add generated polyhexes not already in candidates
            existing = set()
            for cand in candidates:
                # Canonicalize for comparison
                min_q = min(q for q, r in cand)
                min_r = min(r for q, r in cand if q == min_q)
                canon = frozenset((q - min_q, r - min_r) for q, r in cand)
                existing.add(canon)

            for poly in generated:
                if poly not in existing:
                    candidates.append(list(poly))
                    existing.add(poly)
    except Exception:
        pass

    best_side = float('inf')
    best_inner = None
    best_outer_center = None
    best_outer_angle = None

    # Phase 1: For each topology, find optimal outer rotation and center via LP
    # Then refine individual hexagon positions/rotations via SA

    for idx, axial_coords in enumerate(candidates):
        if time.time() - start_time > DEADLINE - 20:
            break

        # Convert to Cartesian with slight expansion to avoid initial overlaps
        scale = 1.0 + 1e-6
        centers = axial_to_cartesian(axial_coords, scale)

        # Initial inner: all at 0 degrees rotation
        initial_inner = [[x, y, 0.0] for x, y in centers]

        # Compute initial composite center
        cx_avg = sum(c[0] for c in centers) / len(centers)
        cy_avg = sum(c[1] for c in centers) / len(centers)

        # Try several outer rotations
        best_for_this_topo = None
        best_for_this_topo_side = float('inf')
        best_for_this_topo_center = None
        best_for_this_topo_angle = None
        best_for_this_topo_inner = None

        # Sample outer rotations
        for beta_deg in range(0, 60, 5):
            beta = math.radians(beta_deg)

            # Use LP to find optimal center and D for this rotation
            # But LP assumes points are the centers, which is a simplification
            # Let's just use a heuristic: center at centroid, compute required side
            side = compute_required_side(initial_inner, (cx_avg, cy_avg), beta_deg)

            # Check validity
            if check_all_disjoint(initial_inner):
                if side < best_for_this_topo_side:
                    best_for_this_topo_side = side
                    best_for_this_topo_center = (cx_avg, cy_avg)
                    best_for_this_topo_angle = beta_deg
                    best_for_this_topo_inner = [list(x) for x in initial_inner]

        if best_for_this_topo is None and best_for_this_topo_inner is not None:
            best_for_this_topo = best_for_this_topo_inner

        if best_for_this_topo_inner is not None:
            if best_for_this_topo_side < best_side:
                best_side = best_for_this_topo_side
                best_inner = best_for_this_topo_inner
                best_outer_center = best_for_this_topo_center
                best_outer_angle = best_for_this_topo_angle

            # Phase 2: Refine with SA for this topology
            if time.time() - start_time > DEADLINE - 15:
                continue

            def sa_objective(params, inner_init):
                inner_list = []
                for i in range(N):
                    x = params[i * 3]
                    y = params[i * 3 + 1]
                    a = params[i * 3 + 2]
                    inner_list.append([x, y, a])

                cx = params[N * 3]
                cy = params[N * 3 + 1]
                angle = params[N * 3 + 2]

                side = compute_required_side_smooth(inner_list, (cx, cy), angle, temp=0.05)

                # Overlap penalty
                penalty = 0.0
                min_dist = SQRT3 - 0.005
                for i in range(N):
                    for j in range(i + 1, N):
                        xi, yi, _ = inner_list[i]
                        xj, yj, _ = inner_list[j]
                        dist = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
                        if dist < min_dist:
                            penalty += (min_dist - dist) ** 2

                return side + 100.0 * penalty

            def sa_validate(params):
                inner_list = []
                for i in range(N):
                    inner_list.append([params[i * 3], params[i * 3 + 1], params[i * 3 + 2]])

                cx = params[N * 3]
                cy = params[N * 3 + 1]
                angle = params[N * 3 + 2]

                side = compute_required_side(inner_list, (cx, cy), angle)

                if not check_all_disjoint(inner_list):
                    return False, None, None, None

                if not check_inside_outer(inner_list, (cx, cy), side, angle):
                    return False, None, None, None

                return True, inner_list, (cx, cy), angle

            init_params = []
            for x, y, a in best_for_this_topo_inner:
                init_params.extend([x, y, a])
            init_params.extend([best_for_this_topo_center[0], best_for_this_topo_center[1], best_for_this_topo_angle])

            current_params = init_params[:]
            current_obj = sa_objective(current_params, initial_inner)

            best_sa_params = current_params[:]
            best_sa_obj = current_obj
            best_sa_valid = False
            best_sa_inner = None
            best_sa_center = None
            best_sa_angle = None

            valid, inner_l, center_l, angle_l = sa_validate(current_params)
            if valid:
                best_sa_valid = True
                best_sa_inner = inner_l
                best_sa_center = center_l
                best_sa_angle = angle_l
                best_sa_obj = compute_required_side(inner_l, center_l, angle_l)

            T0 = 0.3
            T_end = 0.001
            total_steps = 15000
            steps_per_check = 500

            np.random.seed(42 + idx)

            for step in range(total_steps):
                if time.time() - start_time > DEADLINE - 15:
                    break

                T = T0 * (T_end / T0) ** (step / total_steps)

                new_params = current_params[:]
                move_type = np.random.randint(0, 5)

                if move_type < 3:
                    idx_h = np.random.randint(0, N)
                    base = idx_h * 3
                    new_params[base] += np.random.uniform(-0.1, 0.1)
                    new_params[base + 1] += np.random.uniform(-0.1, 0.1)
                elif move_type == 3:
                    idx_h = np.random.randint(0, N)
                    new_params[idx_h * 3 + 2] += np.random.uniform(-5, 5)
                else:
                    if np.random.random() < 0.7:
                        new_params[N * 3] += np.random.uniform(-0.1, 0.1)
                        new_params[N * 3 + 1] += np.random.uniform(-0.1, 0.1)
                    else:
                        new_params[N * 3 + 2] += np.random.uniform(-5, 5)

                new_obj = sa_objective(new_params, initial_inner)

                if new_obj < current_obj or np.random.random() < math.exp(-(new_obj - current_obj) / max(T, 1e-10)):
                    current_params = new_params
                    current_obj = new_obj

                    if step % steps_per_check == 0:
                        valid, inner_l, center_l, angle_l = sa_validate(current_params)
                        if valid:
                            sa_side = compute_required_side(inner_l, center_l, angle_l)
                            if sa_side < best_sa_obj:
                                best_sa_valid = True
                                best_sa_inner = inner_l
                                best_sa_center = center_l
                                best_sa_angle = angle_l
                                best_sa_obj = sa_side

            if best_sa_valid and best_sa_obj < best_side:
                best_side = best_sa_obj
                best_inner = best_sa_inner
                best_outer_center = best_sa_center
                best_outer_angle = best_sa_angle

    # Phase 3: Final polish with Nelder-Mead
    if time.time() - start_time < DEADLINE - 10 and best_inner is not None:
        try:
            flat = []
            for i, j, a in best_inner:
                flat.extend([i, j, a])
            flat.extend([best_outer_center[0], best_outer_center[1], best_outer_angle])

            def polish_obj(p):
                inner_l = []
                for i in range(N):
                    inner_l.append([p[i*3], p[i*3+1], p[i*3+2]])
                cx = p[N*3]
                cy = p[N*3+1]
                ang = p[N*3+2]
                side = compute_required_side_smooth(inner_l, (cx, cy), ang, temp=0.02)
                pen = 0.0
                min_d = SQRT3 - 0.003
                for i in range(N):
                    for j in range(i+1, N):
                        d = math.sqrt((inner_l[i][0]-inner_l[j][0])**2 + (inner_l[i][1]-inner_l[j][1])**2)
                        if d < min_d:
                            pen += (min_d - d)**2
                return side + 200.0 * pen

            polish_res = minimize(
                polish_obj,
                flat,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'maxfev': 5000}
            )

            inner_l = []
            for i in range(N):
                inner_l.append([polish_res.x[i*3], polish_res.x[i*3+1], polish_res.x[i*3+2]])
            cx = polish_res.x[N*3]
            cy = polish_res.x[N*3+1]
            ang = polish_res.x[N*3+2]

            side = compute_required_side(inner_l, (cx, cy), ang)

            if check_all_disjoint(inner_l) and check_inside_outer(inner_l, (cx, cy), side, ang):
                if side < best_side:
                    best_side = side
                    best_inner = inner_l
                    best_outer_center = (cx, cy)
                    best_outer_angle = ang
        except Exception:
            pass

    # Final verification
    if best_inner is None:
        # Fallback
        axial_coords = [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, 0), (1, 1), (0, 2), (-1, 2),
        ]
        scale_fallback = 1.05
        centers = axial_to_cartesian(axial_coords, scale_fallback)
        best_inner = [[x, y, 0.0] for x, y in centers]
        best_outer_center = (0.0, 0.0)
        best_outer_angle = 0.0
        best_side = compute_required_side(best_inner, best_outer_center, best_outer_angle)

    final_inner = [list(x) for x in best_inner]
    final_center = best_outer_center
    final_side = best_side
    final_angle = best_outer_angle

    exact_side = compute_required_side(final_inner, final_center, final_angle)
    if exact_side > final_side + 1e-6:
        final_side = exact_side

    if not check_all_disjoint(final_inner) or not check_inside_outer(final_inner, final_center, final_side, final_angle):
        scale_fallback = 1.05
        axial_coords = [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, 0), (1, 1), (0, 2), (-1, 2),
        ]
        centers = axial_to_cartesian(axial_coords, scale_fallback)
        final_inner = [[x, y, 0.0] for x, y in centers]
        final_center = (0.0, 0.0)
        final_angle = 0.0
        final_side = compute_required_side(final_inner, final_center, final_angle)

    return final_inner, final_center, final_side, final_angle
