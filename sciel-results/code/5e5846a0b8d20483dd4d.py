import numpy as np
import time
import math
from itertools import combinations
from scipy.optimize import linprog

N = 11
DEADLINE = 143.0  # safety margin before 150s kill

# Precompute the 6 unit edge-normal directions for angle 0 (flat-top hexagon).
# A regular hexagon's 6 outward edge normals are 60 deg apart.
_UNIT_NORMALS = np.array([[math.cos(math.pi/6 + math.pi*i/3),
                           math.sin(math.pi/6 + math.pi*i/3)] for i in range(6)],
                         dtype=float)
# 60 deg rotation matrix
_ROT60 = np.array([[math.cos(math.pi/3), -math.sin(math.pi/3)],
                   [math.sin(math.pi/3),  math.cos(math.pi/3)]], dtype=float)


def _rot_normals(angle_deg):
    """6 outward unit edge normals of a regular hexagon at given rotation."""
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    R = np.array([[c, -s], [s, c]])
    return _UNIT_NORMALS @ R.T


def _hex_vertices(cx, cy, side, angle_deg):
    angle = math.radians(angle_deg)
    verts = np.empty((6, 2))
    for i in range(6):
        theta = angle + math.pi * i / 3.0
        verts[i, 0] = cx + side * math.cos(theta)
        verts[i, 1] = cy + side * math.sin(theta)
    return verts


def _edge_axes(verts):
    """6 unit edge-normal axes of a hexagon from its 6 vertices."""
    axes = np.empty((6, 2))
    for i in range(6):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % 6]
        ex, ey = x2 - x1, y2 - y1
        nrm = math.hypot(ex, ey)
        axes[i] = (-ey, ex) / nrm
    return axes


def _sat_gap(v1, v2):
    """Minimum signed gap between two hexagons (positive => separated).
    Returns (gap, axis) where axis is the separating/penetrating direction."""
    axes = np.vstack([_edge_axes(v1), _edge_axes(v2)])
    p1 = v1 @ axes.T
    p2 = v2 @ axes.T
    lo1, hi1 = p1.min(axis=0), p1.max(axis=0)
    lo2, hi2 = p2.min(axis=0), p2.max(axis=0)
    gap = np.minimum(hi2 - hi1, hi1 - hi2)
    k = int(np.argmin(gap))
    return float(gap[k]), axes[k]


def _hex_overlap(c1, a1, c2, a2, tol=1e-9):
    v1 = _hex_vertices(c1[0], c1[1], 1.0, a1)
    v2 = _hex_vertices(c2[0], c2[1], 1.0, a2)
    gap, _ = _sat_gap(v1, v2)
    return gap <= tol


def _min_enclosing_side_lp(centers, inner_angles, outer_angle_deg):
    sqrt3 = math.sqrt(3)
    normals = _rot_normals(outer_angle_deg)
    all_points = np.vstack([_hex_vertices(centers[i][0], centers[i][1], 1.0, inner_angles[i])
                            for i in range(len(centers))])
    A = (all_points @ normals.T).max(axis=0)

    c_obj = np.array([0.0, 0.0, 1.0])
    A_mat = np.zeros((6, 3))
    b_vec = np.zeros(6)
    for i in range(6):
        A_mat[i, 0] = normals[i, 0]
        A_mat[i, 1] = normals[i, 1]
        A_mat[i, 2] = -1.0
        b_vec[i] = -A[i]
    res = linprog(c_obj, A_ub=A_mat, b_ub=b_vec, method='highs')
    if res.status != 0:
        return float('inf'), np.array([0.0, 0.0])
    cx, cy, r = res.x
    return 2 * r / sqrt3, np.array([cx, cy])


def _outer_side_for_angle(centers, inner_angles, outer_angle_deg):
    """Cheaper side-length-only evaluation (no LP) for angle sweeps."""
    normals = _rot_normals(outer_angle_deg)
    all_points = np.vstack([_hex_vertices(centers[i][0], centers[i][1], 1.0, inner_angles[i])
                            for i in range(len(centers))])
    A = (all_points @ normals.T).max(axis=0)
    # Minimize r s.t. n_i . c - r <= -A_i  =>  r >= max_i (n_i . c + A_i).
    # For fixed angle the optimal center minimizes max_i(n_i.c+A_i) + min_i(-n_i.c+A_i).
    # Use LP but only need r; reuse linprog quickly.
    c_obj = np.array([0.0, 0.0, 1.0])
    A_mat = np.zeros((6, 3)); b_vec = np.zeros(6)
    for i in range(6):
        A_mat[i, 0] = normals[i, 0]; A_mat[i, 1] = normals[i, 1]; A_mat[i, 2] = -1.0
        b_vec[i] = -A[i]
    res = linprog(c_obj, A_ub=A_mat, b_ub=b_vec, method='highs')
    if res.status != 0:
        return float('inf')
    return 2 * res.x[2] / math.sqrt(3)


def _optimize_outer(centers, inner_angles):
    """Find near-optimal outer hexagon (angle + center) for fixed inner layout."""
    best_side = float('inf'); best_center = np.array([0.0, 0.0]); best_angle = 0.0
    coarse = np.linspace(0.0, 60.0, 61)
    coarse_sides = []
    for ang in coarse:
        s = _outer_side_for_angle(centers, inner_angles, ang)
        coarse_sides.append(s)
        if s < best_side:
            best_side = s; best_angle = ang
    coarse_sides = np.array(coarse_sides)
    k = int(np.argmin(coarse_sides))
    lo = coarse[max(0, k - 1)]; hi = coarse[min(len(coarse) - 1, k + 1)]
    # Golden-section refinement of the angle
    gr = (math.sqrt(5) - 1) / 2
    a = lo + (1 - gr) * (hi - lo); b = lo + gr * (hi - lo)
    fa = _outer_side_for_angle(centers, inner_angles, a)
    fb = _outer_side_for_angle(centers, inner_angles, b)
    for _ in range(30):
        if fa < fb:
            hi = b; b = a; fb = fa; a = lo + (1 - gr) * (hi - lo); fa = _outer_side_for_angle(centers, inner_angles, a)
        else:
            lo = a; a = b; fa = fb; b = lo + gr * (hi - lo); fb = _outer_side_for_angle(centers, inner_angles, b)
    best_angle = (lo + hi) / 2
    best_side, best_center = _min_enclosing_side_lp(centers, inner_angles, best_angle)
    return best_side, best_center, best_angle


def _check_validity(centers, inner_angles, outer_center, outer_side, outer_angle_deg, tol=1e-7):
    outer_normals = _rot_normals(outer_angle_deg)
    inradius = outer_side * math.sqrt(3) / 2
    for i in range(len(centers)):
        verts = _hex_vertices(centers[i][0], centers[i][1], 1.0, inner_angles[i])
        for vx, vy in verts:
            for nx, ny in outer_normals:
                if (vx - outer_center[0]) * nx + (vy - outer_center[1]) * ny > inradius + tol:
                    return False
    for i in range(N):
        for j in range(i + 1, N):
            if _hex_overlap(centers[i], inner_angles[i], centers[j], inner_angles[j]):
                return False
    return True


def _try_arrangement(centers, inner_angles=None):
    if inner_angles is None:
        inner_angles = [0.0] * N
    for i in range(N):
        for j in range(i + 1, N):
            if _hex_overlap(centers[i], inner_angles[i], centers[j], inner_angles[j]):
                return float('inf'), None
    side, center, angle = _optimize_outer(centers, inner_angles)
    if _check_validity(centers, inner_angles, center, side, angle):
        return side, (centers, inner_angles, center, side, angle)
    return float('inf'), None


def _get_valid_fallback():
    centers = []
    for i in range(N):
        row = i // 5
        col = i % 5
        centers.append((col * 4.0, row * 4.0))
    angles = [0.0] * N
    side, center, angle = _optimize_outer(centers, angles)
    return centers, angles, center, side, angle


def _contact_optimize(centers, angles, max_time=25.0, tol=1e-6):
    """Iteratively push the closest disjoint pair to contact, compacting the hull.
    Returns (centers, angles, improved_bool)."""
    start = time.time()
    centers = [list(c) for c in centers]
    angles = list(angles)
    verts = [_hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i]) for i in range(N)]

    def recompute_verts():
        nonlocal verts
        verts = [_hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i]) for i in range(N)]

    # contact target distance: gap just below 0
    for _ in range(4000):
        if time.time() - start > max_time:
            break
        # find closest pair by center distance (cheap proxy)
        best = None; best_d = float('inf')
        for i in range(N):
            for j in range(i + 1, N):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                d = dx * dx + dy * dy
                if d < best_d:
                    best_d = d; best = (i, j)
        i, j = best
        gap, axis = _sat_gap(verts[i], verts[j])
        if gap >= tol:
            # move pair together by gap/2 along axis
            step = gap * 0.5
            centers[i][0] += axis[0] * step; centers[i][1] += axis[1] * step
            centers[j][0] -= axis[0] * step; centers[j][1] -= axis[1] * step
            recompute_verts()
        else:
            # already in contact; nudge apart slightly then re-compact via other pairs
            break
    # final overlap guard: separate any overlaps
    for _ in range(500):
        if time.time() - start > max_time:
            break
        any_overlap = False
        for i in range(N):
            for j in range(i + 1, N):
                gap, axis = _sat_gap(verts[i], verts[j])
                if gap < 1e-6:
                    any_overlap = True
                    push = (1e-6 - gap) * 0.5 + 1e-4
                    centers[i][0] -= axis[0] * push; centers[i][1] -= axis[1] * push
                    centers[j][0] += axis[0] * push; centers[j][1] += axis[1] * push
        if not any_overlap:
            break
        recompute_verts()
    return centers, angles, True


def _simulate_annealing(initial_centers, initial_angles, max_time=30.0, seed=42):
    start = time.time()
    centers = [list(c) for c in initial_centers]
    angles = list(initial_angles)
    rng = np.random.default_rng(seed)
    current_side, current_center, current_angle = _optimize_outer(centers, angles)
    if not np.isfinite(current_side):
        return initial_centers, initial_angles, float('inf'), None, None
    best_side = current_side
    best_centers = [list(c) for c in centers]
    best_angles = list(angles)
    best_center = current_center; best_angle = current_angle
    T = 0.4; T_min = 0.01; cool = 0.9995

    while time.time() - start < max_time:
        if T < T_min:
            break
        new_centers = [list(c) for c in centers]
        new_angles = list(angles)
        idx = int(rng.integers(N))
        new_centers[idx][0] += rng.normal(0, 0.12)
        new_centers[idx][1] += rng.normal(0, 0.12)
        if rng.random() < 0.5:
            r = int(rng.integers(N))
            new_angles[r] += rng.normal(0, 5.0)
        if rng.random() < 0.1:
            dx = rng.normal(0, 0.05); dy = rng.normal(0, 0.05)
            for i in range(N):
                new_centers[i][0] += dx; new_centers[i][1] += dy

        valid = True
        for i in range(N):
            for j in range(i + 1, N):
                if _hex_overlap(new_centers[i], new_angles[i], new_centers[j], new_angles[j]):
                    valid = False; break
            if not valid:
                break
        if not valid:
            T *= cool; continue
        new_side, new_center, new_angle = _optimize_outer(new_centers, new_angles)
        if not np.isfinite(new_side):
            T *= cool; continue
        delta = new_side - current_side
        if delta < 0 or rng.random() < math.exp(-delta / T):
            centers = new_centers; angles = new_angles
            current_side = new_side; current_center = new_center; current_angle = new_angle
            if current_side < best_side:
                best_side = current_side
                best_centers = [list(c) for c in centers]; best_angles = list(angles)
                best_center = current_center; best_angle = current_angle
        T *= cool
    return best_centers, best_angles, best_side, best_center, best_angle


def _generate_initial_configurations():
    configs = []
    d1 = 2 * math.sqrt(3)
    d2 = 4 * math.sqrt(3)
    first_ring = [(d1 * math.cos(math.pi/3 * i), d1 * math.sin(math.pi/3 * i)) for i in range(6)]
    second_ring = []
    for i in range(6):
        second_ring.append((d2 * math.cos(math.pi/3 * i), d2 * math.sin(math.pi/3 * i)))
        second_ring.append((d2 * math.cos(math.pi/3 * i + math.pi/6), d2 * math.sin(math.pi/3 * i + math.pi/6)))
    for subset in [(0, 2, 4, 6), (1, 3, 5, 7), (0, 3, 6, 9), (0, 1, 2, 3)]:
        configs.append(([(0.0, 0.0)] + first_ring + [second_ring[i] for i in subset], [0.0] * N))
    # spiral
    centers = [(0.0, 0.0)]
    for ring in range(1, 4):
        for k in range(6 * ring):
            if len(centers) >= N:
                break
            a = 2 * math.pi * k / (6 * ring)
            centers.append((ring * d1 * math.cos(a), ring * d1 * math.sin(a)))
    configs.append((centers[:N], [0.0] * N))
    # random
    rng = np.random.default_rng(0)
    for _ in range(4):
        centers = [(float(rng.uniform(-5, 5)), float(rng.uniform(-5, 5))) for _ in range(N)]
        angles = [float(rng.uniform(0, 60)) for _ in range(N)]
        configs.append((centers, angles))
    return configs


def construct():
    start_time = time.time()
    best_side = float('inf')
    best_result = None

    initial_configs = _generate_initial_configurations()
    best_init_centers = None; best_init_angles = None; best_init_side = float('inf')

    for centers, angles in initial_configs:
        if time.time() - start_time > 12:
            break
        side, result = _try_arrangement(centers, angles)
        if side < best_init_side:
            best_init_side = side
            best_init_centers = [list(c) for c in centers]
            best_init_angles = list(angles)
            best_result = result

    # lattice combinations
    d1 = 2 * math.sqrt(3); d2 = 4 * math.sqrt(3)
    first_ring = [(d1 * math.cos(math.pi/3 * i), d1 * math.sin(math.pi/3 * i)) for i in range(6)]
    second_ring = []
    for i in range(6):
        second_ring.append((d2 * math.cos(math.pi/3 * i), d2 * math.sin(math.pi/3 * i)))
        second_ring.append((d2 * math.cos(math.pi/3 * i + math.pi/6), d2 * math.sin(math.pi/3 * i + math.pi/6)))
    for subset in combinations(range(12), 4):
        if time.time() - start_time > 22:
            break
        centers = [(0.0, 0.0)] + first_ring + [second_ring[i] for i in subset]
        inner_angles = [0.0] * N
        side, result = _try_arrangement(centers, inner_angles)
        if side < best_init_side:
            best_init_side = side
            best_init_centers = [list(c) for c in centers]
            best_init_angles = list(inner_angles)
            best_result = result

    if best_result is None:
        fb = _get_valid_fallback()
        best_result = fb
        best_init_centers, best_init_angles = fb[0], fb[1]
        best_init_side = fb[3]

    # Contact optimization (new structural lever)
    if best_init_centers and time.time() - start_time < DEADLINE - 35:
        cc, aa, _ = _contact_optimize(best_init_centers, best_init_angles,
                                      max_time=min(20.0, DEADLINE - time.time() - 35))
        valid = True
        for i in range(N):
            for j in range(i + 1, N):
                if _hex_overlap(cc[i], aa[i], cc[j], aa[j]):
                    valid = False; break
            if not valid:
                break
        if valid:
            side, center, angle = _optimize_outer(cc, aa)
            if np.isfinite(side) and _check_validity(cc, aa, center, side, angle):
                if side < best_side:
                    best_side = side
                    best_result = (cc, aa, center, side, angle)
                    best_init_centers, best_init_angles = cc, aa

    # Simulated annealing
    if best_init_centers and time.time() - start_time < DEADLINE - 8:
        sa = _simulate_annealing(best_init_centers, best_init_angles,
                                 max_time=DEADLINE - time.time() - 8)
        sa_centers, sa_angles, sa_side, sa_center, sa_angle = sa
        if np.isfinite(sa_side) and _check_validity(sa_centers, sa_angles, sa_center, sa_side, sa_angle):
            if sa_side < best_side:
                best_side = sa_side
                best_result = (sa_centers, sa_angles, sa_center, sa_side, sa_angle)

    if best_result is None:
        fb = _get_valid_fallback()
        best_result = fb

    centers, inner_angles, outer_center, outer_side, outer_angle = best_result
    if not _check_validity(centers, inner_angles, outer_center, outer_side, outer_angle):
        fb = _get_valid_fallback()
        centers, inner_angles, outer_center, outer_side, outer_angle = fb

    inner = [[float(centers[i][0]), float(centers[i][1]), float(inner_angles[i])] for i in range(N)]
    return inner, (float(outer_center[0]), float(outer_center[1])), float(outer_side), float(outer_angle)
