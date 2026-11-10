# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _hex_vertices(cx, cy, d, angle_deg):
    a = np.radians(angle_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


def _pairwise_min_gap(v1, v2):
    e1 = np.roll(v1, -1, axis=0) - v1
    e2 = np.roll(v2, -1, axis=0) - v2
    n1 = np.column_stack([-e1[:, 1], e1[:, 0]])
    n2 = np.column_stack([-e2[:, 1], e2[:, 0]])

    min_gap = np.inf
    for n in n1:
        p1 = v1 @ n
        p2 = v2 @ n
        gap = max(p2.min() - p1.max(), p1.min() - p2.max(), 0)
        if gap < min_gap:
            min_gap = gap
    for n in n2:
        p1 = v1 @ n
        p2 = v2 @ n
        gap = max(p2.min() - p1.max(), p1.min() - p2.max(), 0)
        if gap < min_gap:
            min_gap = gap
    return min_gap


def _compute_s_and_penalty(params, deadline):
    if time.time() > deadline:
        return None

    n = 12
    inner_xy = params[:n * 2].reshape(n, 2)
    inner_ang = params[n * 2:n * 3]
    outer_cx, outer_cy = params[n * 3:n * 3 + 2]
    outer_ang = params[n * 3 + 2]

    a_rad = np.radians(outer_ang)
    edge_angles = a_rad + np.radians(30 + 60 * np.arange(6))
    outward_normals = np.column_stack([np.cos(edge_angles), np.sin(edge_angles)])

    all_inner_vertices = []
    for i in range(n):
        v = _hex_vertices(inner_xy[i, 0], inner_xy[i, 1], 1.0, inner_ang[i])
        all_inner_vertices.append(v)
    all_inner_vertices = np.vstack(all_inner_vertices)

    diffs = all_inner_vertices - np.array([outer_cx, outer_cy])
    dots = diffs @ outward_normals.T  # (72, 6)
    s_vals = 2.0 / np.sqrt(3.0) * dots  # (72, 6)

    # Use soft-max to smooth the max
    beta = 50.0
    s_soft = np.log(np.sum(np.exp(beta * s_vals), axis=(0, 1))) / beta
    s_hard = np.max(s_vals)
    s = max(s_soft, 0.0)

    # Overlap penalty using smooth approximation
    overlap_penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            v1 = _hex_vertices(inner_xy[i, 0], inner_xy[i, 1], 1.0, inner_ang[i])
            v2 = _hex_vertices(inner_xy[j, 0], inner_xy[j, 1], 1.0, inner_ang[j])
            gap = _pairwise_min_gap(v1, v2)
            if gap < 0:
                # Smooth penalty: use a smooth max(0, -gap)
                # -gap > 0, so penalty = -gap
                overlap_penalty += (-gap) ** 2

    penalty_weight = 50.0
    objective = s + penalty_weight * overlap_penalty
    return objective


def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 580

    n = 12
    inner_init = np.array(
        [
            [0, 0], [-2.5, 0], [2.5, 0],
            [-1.25, 2.17], [1.25, 2.17], [-1.25, -2.17], [1.25, -2.17],
            [-3.75, 2.17], [3.75, 2.17], [-3.75, -2.17], [3.75, -2.17],
            [0, -4],
        ]
    )
    inner_ang_init = np.zeros(n)
    outer_cx_init, outer_cy_init = 0.0, 0.0
    outer_ang_init = 0.0

    params_init = np.zeros(n * 3 + 3)
    for i in range(n):
        params_init[i * 3] = inner_init[i, 0]
        params_init[i * 3 + 1] = inner_init[i, 1]
        params_init[i * 3 + 2] = inner_ang_init[i]
    params_init[n * 3] = outer_cx_init
    params_init[n * 3 + 1] = outer_cy_init
    params_init[n * 3 + 2] = outer_ang_init

    best_params = params_init.copy()
    best_s = np.inf

    np.random.seed(42)
    n_restarts = 4

    for restart in range(n_restarts):
        if time.time() > deadline - 80:
            break

        if restart == 0:
            params = params_init.copy()
        else:
            params = params_init.copy()
            params[:n * 2] += np.random.normal(0, 0.3, n * 2)
            params[n * 2:n * 3] += np.random.uniform(-20, 20, n)

        result = minimize(
            _compute_s_and_penalty,
            params,
            args=(deadline,),
            method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10},
        )

        if result.fun is not None and np.isfinite(result.fun):
            p = result.x
            inner_xy = p[:n * 2].reshape(n, 2)
            inner_ang = p[n * 2:n * 3]
            outer_cx, outer_cy = p[n * 3:n * 3 + 2]
            outer_ang = p[n * 3 + 2]

            a_rad = np.radians(outer_ang)
            edge_angles = a_rad + np.radians(30 + 60 * np.arange(6))
            outward_normals = np.column_stack([np.cos(edge_angles), np.sin(edge_angles)])

            all_inner_vertices = []
            for i in range(n):
                v = _hex_vertices(inner_xy[i, 0], inner_xy[i, 1], 1.0, inner_ang[i])
                all_inner_vertices.append(v)
            all_inner_vertices = np.vstack(all_inner_vertices)

            diffs = all_inner_vertices - np.array([outer_cx, outer_cy])
            dots = diffs @ outward_normals.T
            s = 2.0 / np.sqrt(3.0) * np.max(dots)
            s = max(s, 0.0)

            has_overlap = False
            for i in range(n):
                for j in range(i + 1, n):
                    v1 = _hex_vertices(inner_xy[i, 0], inner_xy[i, 1], 1.0, inner_ang[i])
                    v2 = _hex_vertices(inner_xy[j, 0], inner_xy[j, 1], 1.0, inner_ang[j])
                    gap = _pairwise_min_gap(v1, v2)
                    if gap < -1e-6:
                        has_overlap = True
                        break
                if has_overlap:
                    break

            if not has_overlap and s < best_s:
                best_s = s
                best_params = result.x.copy()

    p = best_params
    inner_xy = p[:n * 2].reshape(n, 2)
    inner_ang = p[n * 2:n * 3]
    outer_cx, outer_cy = p[n * 3:n * 3 + 2]
    outer_ang = p[n * 3 + 2]

    a_rad = np.radians(outer_ang)
    edge_angles = a_rad + np.radians(30 + 60 * np.arange(6))
    outward_normals = np.column_stack([np.cos(edge_angles), np.sin(edge_angles)])

    all_inner_vertices = []
    for i in range(n):
        v = _hex_vertices(inner_xy[i, 0], inner_xy[i, 1], 1.0, inner_ang[i])
        all_inner_vertices.append(v)
    all_inner_vertices = np.vstack(all_inner_vertices)

    diffs = all_inner_vertices - np.array([outer_cx, outer_cy])
    dots = diffs @ outward_normals.T
    s = 2.0 / np.sqrt(3.0) * np.max(dots)
    s = max(s, np.sqrt(12.0))

    inner_hex_data = np.zeros((n, 3))
    for i in range(n):
        inner_hex_data[i] = [inner_xy[i, 0], inner_xy[i, 1], inner_ang[i]]

    outer_hex_data = np.array([outer_cx, outer_cy, outer_ang])

    return inner_hex_data, outer_hex_data, s


# EVOLVE-BLOCK-END
