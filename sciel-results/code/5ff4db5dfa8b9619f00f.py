import numpy as np
import time
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0
_TIME_START = time.time()


def get_hex_vertices_batch(xys_rots):
    """Batch compute vertices for all hexagons. xys_rots is (N, 3) array."""
    n_pts = len(xys_rots)
    base_angles = np.radians(np.array([0, 60, 120, 180, 240, 300]))
    rots = np.radians(xys_rots[:, 2])
    angles = base_angles[None, :] + rots[:, None]
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    cx = xys_rots[:, 0]
    cy = xys_rots[:, 1]
    x = cx[:, None] + cos_a
    y = cy[:, None] + sin_a
    return np.stack((x, y), axis=-1)


def sat_overlap_penalty(vertices_list):
    """
    Smooth SAT overlap penalty for all pairs.
    Returns scalar penalty (sum of soft-min penetration depths).
    """
    n_pts = len(vertices_list)
    # Precompute edge normals for each hex
    axes_all = []
    for v in vertices_list:
        hex_axes = []
        for i in range(6):
            p1 = v[i]
            p2 = v[(i + 1) % 6]
            e = p2 - p1
            n = np.array([-e[1], e[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-12:
                n = n / norm
                hex_axes.append(n)
        axes_all.append(np.array(hex_axes))

    penalty = 0.0
    for i in range(n_pts):
        for j in range(i + 1, n_pts):
            v1 = vertices_list[i]
            v2 = vertices_list[j]
            all_axes = np.vstack([axes_all[i], axes_all[j]])

            min_overlap = float('inf')
            for n in all_axes:
                proj1 = v1 @ n
                min1, max1 = np.min(proj1), np.max(proj1)
                proj2 = v2 @ n
                min2, max2 = np.min(proj2), np.max(proj2)

                if max1 < min2 or max2 < min1:
                    min_overlap = 0.0
                    break

                overlap = min(max1, max2) - max(min1, min2)
                min_overlap = min(min_overlap, overlap)

            if min_overlap > 0:
                penalty += min_overlap ** 2

    return penalty


def outer_hex_side_centered(points, cx, cy, outer_rot_deg):
    """
    Minimum side length of regular hexagon centered at (cx,cy) with given rotation
    that encloses all points.
    """
    pts = points - np.array([cx, cy])
    angles = np.radians(np.array([90, 150, 210, 270, 330, 30]) + outer_rot_deg)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    proj = pts @ normals.T
    max_proj = np.max(np.abs(proj))
    return (2.0 / np.sqrt(3.0)) * max_proj


def make_hex_grid(rows, cols, rot_offset=0.0, gap=0.05):
    pts = []
    sqrt3 = np.sqrt(3)
    spacing_x = sqrt3 + gap
    spacing_y = 1.5 + gap * 0.5
    for r in range(rows):
        y = r * spacing_y
        for c in range(cols):
            x = c * spacing_x
            if r % 2 == 1:
                x += spacing_x / 2.0
            pts.append([x, y, rot_offset])
    pts = np.array(pts)
    pts[:, :2] -= np.mean(pts[:, :2], axis=0)
    return pts


def make_ring(gap=0.05):
    pts = []
    sqrt3 = np.sqrt(3)
    d1 = sqrt3 + gap
    d2 = 2 * sqrt3 + gap
    pts.append([0, 0, 0.0])
    for k in range(6):
        ang = np.radians(60 * k)
        pts.append([d1 * np.cos(ang), d1 * np.sin(ang), 0.0])
    for k in range(5):
        ang = np.radians(60 * k + 30)
        pts.append([d2 * np.cos(ang), d2 * np.sin(ang), 0.0])
    pts = np.array(pts)
    pts[:, :2] -= np.mean(pts[:, :2], axis=0)
    return pts


def make_cluster(gap=0.05):
    """Two rows of 3+3+3 staggered, but tighter."""
    pts = []
    sqrt3 = np.sqrt(3)
    sx = sqrt3 + gap
    sy = 1.5 + gap * 0.3
    # Row 0: 3 hexes
    for c in range(3):
        pts.append([c * sx, 0.0, 0.0])
    # Row 1: 3 hexes offset
    for c in range(3):
        pts.append([c * sx + sx / 2, sy, 0.0])
    # Row 2: 3 hexes
    for c in range(3):
        pts.append([c * sx, 2 * sy, 0.0])
    # Row 3: 3 hexes offset
    for c in range(3):
        pts.append([c * sx + sx / 2, 3 * sy, 0.0])
    pts = np.array(pts)
    pts[:, :2] -= np.mean(pts[:, :2], axis=0)
    return pts


def make_tight_ring(gap=0.02):
    """Tighter ring configuration."""
    pts = []
    sqrt3 = np.sqrt(3)
    d1 = sqrt3 + gap
    d2 = 1.8 * sqrt3 + gap
    pts.append([0, 0, 0.0])
    for k in range(6):
        ang = np.radians(60 * k)
        pts.append([d1 * np.cos(ang), d1 * np.sin(ang), 0.0])
    for k in range(5):
        ang = np.radians(60 * k + 30)
        pts.append([d2 * np.cos(ang), d2 * np.sin(ang), 0.0])
    pts = np.array(pts)
    pts[:, :2] -= np.mean(pts[:, :2], axis=0)
    return pts


def construct():
    start_time = time.time()

    def time_left():
        return DEADLINE - (time.time() - start_time)

    def objective(params):
        if time.time() - start_time > DEADLINE - 5:
            return 1e6

        xys_rots = params[:3 * N].reshape(N, 3)
        cx = params[3 * N]
        cy = params[3 * N + 1]
        outer_rot = params[3 * N + 2]

        vertices_list = get_hex_vertices_batch(xys_rots)

        penalty = sat_overlap_penalty(vertices_list)

        all_points = np.vstack(vertices_list)
        L = outer_hex_side_centered(all_points, cx, cy, outer_rot)

        return L + 500.0 * penalty

    # Generate diverse initial configurations
    configs = []
    gap = 0.05
    configs.append(make_hex_grid(4, 3, gap=gap))
    configs.append(make_hex_grid(3, 4, gap=gap))
    configs.append(make_hex_grid(2, 6, gap=gap))
    configs.append(make_hex_grid(6, 2, gap=gap))
    configs.append(make_ring(gap=gap))
    configs.append(make_cluster(gap=gap))
    configs.append(make_tight_ring(gap=0.02))

    # Add rotated versions
    for base_cfg in configs[:4]:
        rotated = base_cfg.copy()
        rotated[:, 2] = 30.0
        configs.append(rotated)

    outer_rots = [0.0, 30.0, 15.0, 45.0, 60.0]

    best_L = float('inf')
    best_params = None

    for cfg in configs:
        if time.time() - start_time > DEADLINE - 15:
            break

        for orot in outer_rots:
            if time.time() - start_time > DEADLINE - 15:
                break

            init_params = np.concatenate((cfg.flatten(), [0.0, 0.0, orot]))

            xys_rots_init = init_params[:3 * N].reshape(N, 3)
            vertices_list_init = get_hex_vertices_batch(xys_rots_init)

            penalty_init = sat_overlap_penalty(vertices_list_init)
            if penalty_init < 1e-6:
                all_pts = np.vstack(vertices_list_init)
                L_val = outer_hex_side_centered(all_pts, 0.0, 0.0, orot)
                if L_val < best_L:
                    best_L = L_val
                    best_params = init_params.copy()

            # Nelder-Mead optimization
            try:
                res = minimize(
                    objective,
                    init_params,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 200,
                        'xatol': 1e-5,
                        'fatol': 1e-5,
                        'adaptive': True
                    }
                )

                xys_rots = res.x[:3 * N].reshape(N, 3)
                cx_val = res.x[3 * N]
                cy_val = res.x[3 * N + 1]
                outer_rot_val = res.x[3 * N + 2]

                vertices_list = get_hex_vertices_batch(xys_rots)
                penalty = sat_overlap_penalty(vertices_list)

                if penalty < 1e-6:
                    all_pts = np.vstack(vertices_list)
                    L_val = outer_hex_side_centered(all_pts, cx_val, cy_val, outer_rot_val)

                    if L_val < best_L:
                        best_L = L_val
                        best_params = res.x.copy()

            except Exception:
                pass

    # Second stage: finer optimization from best
    if best_params is not None and time.time() - start_time < DEADLINE - 25:
        try:
            res = minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 400,
                    'xatol': 1e-7,
                    'fatol': 1e-7,
                    'adaptive': True
                }
            )

            xys_rots = res.x[:3 * N].reshape(N, 3)
            cx_val = res.x[3 * N]
            cy_val = res.x[3 * N + 1]
            outer_rot_val = res.x[3 * N + 2]

            vertices_list = get_hex_vertices_batch(xys_rots)
            penalty = sat_overlap_penalty(vertices_list)

            if penalty < 1e-6:
                all_pts = np.vstack(vertices_list)
                L_val = outer_hex_side_centered(all_pts, cx_val, cy_val, outer_rot_val)

                if L_val < best_L:
                    best_L = L_val
                    best_params = res.x.copy()

        except Exception:
            pass

    # Third stage: high-precision polish
    if best_params is not None and time.time() - start_time < DEADLINE - 12:
        try:
            res = minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 300,
                    'xatol': 1e-9,
                    'fatol': 1e-9,
                    'adaptive': True
                }
            )

            xys_rots = res.x[:3 * N].reshape(N, 3)
            cx_val = res.x[3 * N]
            cy_val = res.x[3 * N + 1]
            outer_rot_val = res.x[3 * N + 2]

            vertices_list = get_hex_vertices_batch(xys_rots)
            penalty = sat_overlap_penalty(vertices_list)

            if penalty < 1e-6:
                all_pts = np.vstack(vertices_list)
                L_val = outer_hex_side_centered(all_pts, cx_val, cy_val, outer_rot_val)

                if L_val < best_L:
                    best_L = L_val
                    best_params = res.x.copy()

        except Exception:
            pass

    # Final: try a few more random perturbations around best
    if best_params is not None and time.time() - start_time < DEADLINE - 8:
        for attempt in range(5):
            if time.time() - start_time > DEADLINE - 8:
                break
            perturbed = best_params + np.random.normal(0, 0.05, size=best_params.shape)
            try:
                res = minimize(
                    objective,
                    perturbed,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 150,
                        'xatol': 1e-7,
                        'fatol': 1e-7,
                        'adaptive': True
                    }
                )

                xys_rots = res.x[:3 * N].reshape(N, 3)
                cx_val = res.x[3 * N]
                cy_val = res.x[3 * N + 1]
                outer_rot_val = res.x[3 * N + 2]

                vertices_list = get_hex_vertices_batch(xys_rots)
                penalty = sat_overlap_penalty(vertices_list)

                if penalty < 1e-6:
                    all_pts = np.vstack(vertices_list)
                    L_val = outer_hex_side_centered(all_pts, cx_val, cy_val, outer_rot_val)

                    if L_val < best_L:
                        best_L = L_val
                        best_params = res.x.copy()

            except Exception:
                pass

    if best_params is None:
        # Fallback: simple grid
        inner = []
        for k in range(N):
            inner.append([3.0 * k, 0.0, 0.0])
        pts = np.array([[x, y] for x, y, r in inner])
        L_val = outer_hex_side_centered(pts, 0.0, 0.0, 0.0)
        return inner, (0.0, 0.0), L_val, 0.0

    xys_rots = best_params[:3 * N].reshape(N, 3)
    cx_val = best_params[3 * N]
    cy_val = best_params[3 * N + 1]
    outer_rot_val = best_params[3 * N + 2]

    inner = [[float(x), float(y), float(r)] for x, y, r in xys_rots]
    center = [float(cx_val), float(cy_val)]
    side = float(best_L)
    angle = float(outer_rot_val)

    return inner, center, side, angle
