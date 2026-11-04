"""
Optimized packing of 12 unit hexagons in a minimal outer hexagon.
Structural idea: use a fast, vectorized validity/containment representation
combined with scipy.optimize.minimize (Nelder-Mead / Powell) for refinement,
plus a simulated-annealing phase for global exploration.
"""
import time
import math
import random
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0  # safety margin before the 150s kill
INRADIUS = math.sqrt(3.0) / 2.0
MIN_DIST_CIRCLE = 2.0 * INRADIUS  # minimum center distance for non-overlap (circle packing)
MIN_DIST_CIRCLE_SQ = MIN_DIST_CIRCLE * MIN_DIST_CIRCLE


def _hex_vertices(cx, cy, angle_rad):
    """Regular hexagon vertices, side 1, centered at (cx,cy), rotated by angle_rad."""
    ca = math.cos(angle_rad)
    sa = math.sin(angle_rad)
    verts = []
    for k in range(6):
        theta = k * math.pi / 3.0
        x = ca * math.cos(theta) - sa * math.sin(theta)
        y = sa * math.cos(theta) + ca * math.sin(theta)
        verts.append((cx + x, cy + y))
    return np.array(verts)


def _hex_axes(angle_rad):
    """6 edge-normal directions (unit vectors) of a hexagon oriented by angle_rad."""
    axes = []
    for k in range(6):
        a = angle_rad + math.pi / 6.0 + k * math.pi / 3.0
        axes.append((math.cos(a), math.sin(a)))
    return np.array(axes)


def _project_polygon(verts, axes):
    """Project polygon vertices onto given axes. Returns array of shape (len(axes), 2) = (min, max)."""
    P = verts @ axes.T
    return np.vstack([P.min(axis=0), P.max(axis=0)]).T


def _sat_overlap(A, B):
    """SAT overlap test between two hexagons. Returns True if overlapping."""
    EPS = 1e-9
    axesA = _hex_axes_from_verts(A)
    axesB = _hex_axes_from_verts(B)
    all_axes = np.vstack([axesA, axesB])

    projA = _project_polygon(A, all_axes)
    projB = _project_polygon(B, all_axes)

    for i in range(len(all_axes)):
        if projA[i, 1] < projB[i, 0] - EPS or projB[i, 1] < projA[i, 0] - EPS:
            return False
    return True


def _hex_axes_from_verts(verts):
    """Compute edge normals from polygon vertices."""
    axes = []
    for k in range(6):
        p1 = verts[k]
        p2 = verts[(k + 1) % 6]
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])
        nl = np.linalg.norm(normal)
        if nl > 1e-9:
            axes.append(normal / nl)
    return np.array(axes)


def _is_valid_fast(cx_arr, cy_arr, angle_arr):
    """
    Fast validity check: first circle-packing pre-check, then SAT for close pairs.
    Returns True if no overlaps.
    """
    n = len(cx_arr)

    # Fast pre-check: circle packing
    for i in range(n):
        dx = cx_arr[i] - cx_arr[i + 1:]
        dy = cy_arr[i] - cy_arr[i + 1:]
        dist_sq = dx * dx + dy * dy
        if np.any(dist_sq < MIN_DIST_CIRCLE_SQ - 1e-9):
            # Circle pre-check failed; need exact SAT
            pass
        else:
            # All remaining pairs are far enough that circle packing guarantees no overlap?
            # Actually circle packing is a sufficient condition for non-overlap only if
            # the hexagons are axis-aligned. For rotated hexagons, we still need SAT
            # but we can skip pairs that are very far apart.
            pass

    # Exact check with early termination
    verts_list = []
    for i in range(n):
        verts_list.append(_hex_vertices(cx_arr[i], cy_arr[i], angle_arr[i]))

    for i in range(n):
        vi = verts_list[i]
        for j in range(i + 1, n):
            vj = verts_list[j]
            # Quick distance check to skip far pairs
            d = np.linalg.norm(vi.mean(axis=0) - vj.mean(axis=0))
            if d > 2.0 * 1.0 + 1e-6:  # circumradius is 1, so max extent is 2
                continue
            if _sat_overlap(vi, vj):
                return False

    return True


def _outer_hex_side_for_orientation(cx_arr, cy_arr, angle_arr, outer_theta, outer_center):
    """
    Compute the side length of the outer hexagon (given orientation and center)
    needed to contain all inner hexagons.
    """
    axes = _hex_axes(outer_theta)
    all_pts = []
    for i in range(N):
        all_pts.append(_hex_vertices(cx_arr[i], cy_arr[i], angle_arr[i]))
    all_pts = np.vstack(all_pts)

    P = all_pts @ axes.T
    oc = np.array(outer_center) @ axes.T
    max_dev = np.max(np.abs(P - oc))
    return 2.0 * max_dev / math.sqrt(3.0)


def _compute_best_outer(cx_arr, cy_arr, angle_arr):
    """Find best outer hexagon orientation and center for given set of points."""
    all_pts = []
    for i in range(N):
        all_pts.append(_hex_vertices(cx_arr[i], cy_arr[i], angle_arr[i]))
    all_pts = np.vstack(all_pts)

    best_side = float('inf')
    best_center = np.array([0.0, 0.0])
    best_angle = 0.0

    # Sweep over 60-degree range (hexagon symmetry)
    for ang in np.linspace(0.0, math.pi / 3.0, 181):
        axes = _hex_axes(ang)
        P = all_pts @ axes.T
        c = (P.min(axis=0) + P.max(axis=0)) / 2.0
        side = 2.0 * np.max(P.max(axis=0) - c) / math.sqrt(3.0)
        if side < best_side - 1e-12:
            best_side = side
            best_center = axes.T @ c
            best_angle = float(ang)

    return best_side, best_center, best_angle


def _objective(params, outer_theta, outer_center):
    """
    Objective function: outer side length + overlap penalty + containment penalty.
    params: flat array of [cx0, cy0, a0, cx1, cy1, a1, ..., cx11, cy11, a11]
    """
    cx_arr = params[0::3]
    cy_arr = params[1::3]
    angle_arr = params[2::3]

    # Compute bounding hexagon side
    side = _outer_hex_side_for_orientation(cx_arr, cy_arr, angle_arr, outer_theta, outer_center)

    # Overlap penalty
    verts_list = []
    for i in range(N):
        verts_list.append(_hex_vertices(cx_arr[i], cy_arr[i], angle_arr[i]))

    overlap_pen = 0.0
    for i in range(N):
        vi = verts_list[i]
        for j in range(i + 1, N):
            vj = verts_list[j]
            d = np.linalg.norm(vi.mean(axis=0) - vj.mean(axis=0))
            if d > 2.5:
                continue
            if _sat_overlap(vi, vj):
                # Compute minimum separation depth
                axesA = _hex_axes_from_verts(vi)
                axesB = _hex_axes_from_verts(vj)
                all_axes = np.vstack([axesA, axesB])
                projA = _project_polygon(vi, all_axes)
                projB = _project_polygon(vj, all_axes)
                min_sep = float('inf')
                for k in range(len(all_axes)):
                    sep = min(projA[k, 1] - projB[k, 0], projB[k, 1] - projA[k, 0])
                    min_sep = min(min_sep, sep)
                if min_sep < 0:
                    overlap_pen += 10.0 * min_sep * min_sep

    return side + overlap_pen


def _construct_initial_config(seed_type=0):
    """Generate initial configuration based on seed type."""
    if seed_type == 0:
        # Hexagonal cluster
        d = 2.0 * math.sqrt(3.0) + 0.001
        a1 = np.array([d, 0.0])
        a2 = np.array([d / 2.0, d * math.sqrt(3.0) / 2.0])
        coords = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1),
                  (2, 0), (-2, 0), (0, 2), (0, -2), (2, -2)]
        pts = []
        for (i, j) in coords:
            pts.append(i * a1 + j * a2)
        pts = np.array(pts)
        c = pts.mean(axis=0)
        pts = pts - c
        angles = np.zeros(N)
        return pts, angles

    elif seed_type == 1:
        # Triangular lattice
        d = 2.0 * math.sqrt(3.0) + 0.001
        a1 = np.array([d, 0.0])
        a2 = np.array([d / 2.0, d * math.sqrt(3.0) / 2.0])
        pts = []
        for i in range(-1, 3):
            for j in range(-1, 2):
                pts.append(i * a1 + j * a2)
        pts = np.array(pts)
        c = pts.mean(axis=0)
        pts = pts - c
        angles = np.zeros(N)
        return pts, angles

    elif seed_type == 2:
        # Random in disk
        pts = []
        for _ in range(N):
            r = random.uniform(0, 4.0)
            theta = random.uniform(0, 2 * math.pi)
            pts.append([r * math.cos(theta), r * math.sin(theta)])
        pts = np.array(pts)
        c = pts.mean(axis=0)
        pts = pts - c
        angles = np.array([random.uniform(-math.pi / 6, math.pi / 6) for _ in range(N)])
        return pts, angles

    else:
        # Ring + center
        pts = [np.array([0.0, 0.0])]
        for k in range(11):
            ang = 2 * math.pi * k / 11
            r = 2.5 * math.sqrt(3.0)
            pts.append([r * math.cos(ang), r * math.sin(ang)])
        pts = np.array(pts)
        c = pts.mean(axis=0)
        pts = pts - c
        angles = np.zeros(N)
        return pts, angles


def _simulated_annealing(centers, angles, outer_theta, outer_center, max_iters=3000):
    """Simulated annealing to find better packings."""
    best_centers = centers.copy()
    best_angles = angles.copy()
    best_obj = _objective(np.concatenate([centers[:, 0], centers[:, 1], angles]), outer_theta, outer_center)

    current_obj = best_obj
    T = 1.0
    T_min = 0.01
    cooling = 0.9995

    for it in range(max_iters):
        if time.time() > DEADLINE:
            break

        # Propose move
        i = random.randint(0, N - 1)
        move_type = random.random()

        if move_type < 0.7:
            # Move center
            delta = np.array([random.gauss(0, T), random.gauss(0, T)])
            new_centers = centers.copy()
            new_centers[i] += delta
            new_angles = angles.copy()
        else:
            # Rotate
            new_centers = centers.copy()
            new_angles = angles.copy()
            new_angles[i] += random.gauss(0, T * 0.5)

        params = np.concatenate([new_centers[:, 0], new_centers[:, 1], new_angles])
        new_obj = _objective(params, outer_theta, outer_center)
        delta_E = new_obj - current_obj

        # Accept/reject
        if delta_E < 0 or random.random() < math.exp(-delta_E / max(T, 1e-10)):
            centers = new_centers
            angles = new_angles
            current_obj = new_obj
            if new_obj < best_obj:
                best_obj = new_obj
                best_centers = centers.copy()
                best_angles = angles.copy()

        T = max(T * cooling, T_min)

    return best_centers, best_angles, best_obj


def construct():
    start = time.time()
    random.seed(42)
    np.random.seed(42)

    best_side = float('inf')
    best_centers = None
    best_angles = None
    best_center = (0.0, 0.0)
    best_angle = 0.0

    def _consider(centers, angles, side, center, angle):
        nonlocal best_side, best_centers, best_angles, best_center, best_angle
        if side < best_side:
            best_side = side
            best_centers = centers.copy()
            best_angles = angles.copy()
            best_center = center
            best_angle = angle

    seed_types = [0, 1, 2, 3]

    for seed_type in seed_types:
        if time.time() > DEADLINE - 15:
            break

        centers, angles = _construct_initial_config(seed_type)

        # Compute initial best outer
        side, center, angle = _compute_best_outer(centers, angles)
        _consider(centers, angles, side, center, angle)

        # Simulated annealing
        bc, ba, bobj = _simulated_annealing(centers, angles, angle, center, max_iters=2000)
        all_pts = np.vstack([_hex_vertices(ba[k], bc[k][0], bc[k][1]) for k in range(N)])
        side2, center2, angle2 = _compute_best_outer(bc, ba)
        _consider(bc, ba, side2, center2, angle2)

        # SciPy Nelder-Mead refinement
        if time.time() > DEADLINE - 5:
            break

        params = np.concatenate([bc[:, 0], bc[:, 1], ba])
        result = minimize(
            _objective,
            params,
            args=(angle2, center2),
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6}
        )

        if result.success or result.fun < bobj:
            new_params = result.x
            new_centers = np.column_stack([new_params[0::3], new_params[1::3]])
            new_angles = new_params[2::3]
            side3, center3, angle3 = _compute_best_outer(new_centers, new_angles)
            _consider(new_centers, new_angles, side3, center3, angle3)

    # Final refinement on best configuration
    if best_centers is not None and time.time() < DEADLINE - 5:
        # Fine-tune outer angle
        all_pts = np.vstack([_hex_vertices(best_angles[k], best_centers[k][0], best_centers[k][1]) for k in range(N)])
        side, center, angle = _compute_best_outer(best_centers, best_angles)

        # Final polish with scipy
        params = np.concatenate([best_centers[:, 0], best_centers[:, 1], best_angles])
        result = minimize(
            _objective,
            params,
            args=(angle, center),
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-7, 'fatol': 1e-7}
        )

        if result.success or result.fun < side:
            new_params = result.x
            new_centers = np.column_stack([new_params[0::3], new_params[1::3]])
            new_angles = new_params[2::3]
            side_final, center_final, angle_final = _compute_best_outer(new_centers, new_angles)
            _consider(new_centers, new_angles, side_final, center_final, angle_final)

    # Fallback
    if best_centers is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Format output
    inner = []
    for k in range(N):
        inner.append([float(best_centers[k][0]), float(best_centers[k][1]), float(math.degrees(best_angles[k]))])

    return inner, best_center, best_side, math.degrees(best_angle)
