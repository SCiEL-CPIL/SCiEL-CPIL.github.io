import numpy as np
from scipy.optimize import minimize
import time

N = 12
SQRT3 = np.sqrt(3.0)


def _hex_vertices(cx, cy, s, angle_deg):
    angle = np.radians(angle_deg)
    base_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0
    angles = base_angles + angle
    xs = cx + s * np.cos(angles)
    ys = cy + s * np.sin(angles)
    return np.column_stack([xs, ys])


def _point_hex_violation(x, y, s):
    v1 = abs(x + y / SQRT3) - s
    v2 = abs(y) - s * SQRT3 / 2
    v3 = abs(x - y / SQRT3) - s
    return max(v1, v2, v3)


def _hex_min_distance(h1, h2):
    def _project(vertices, axis):
        dots = vertices @ axis
        return dots.min(), dots.max()

    def _get_axes(hex_verts):
        axes = []
        for i in range(6):
            edge = hex_verts[(i + 1) % 6] - hex_verts[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-12:
                axes.append(normal / norm)
        return axes

    axes = _get_axes(h1) + _get_axes(h2)

    min_dist = float('inf')
    for axis in axes:
        min1, max1 = _project(h1, axis)
        min2, max2 = _project(h2, axis)
        if max1 < min2:
            dist = min2 - max1
        elif max2 < min1:
            dist = min1 - max2
        else:
            dist = 0.0
        min_dist = min(min_dist, dist)

    return min_dist


def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    # Initial hexagonal packing (3 rows x 4 cols honeycomb)
    positions = []
    angles = []
    for row in range(3):
        y = row * SQRT3
        offset = 1.0 if row % 2 == 1 else 0.0
        for col in range(4):
            x = col * 2.0 + offset
            positions.append([x, y])
            angles.append(0.0)

    pos_arr = np.array(positions)
    center = pos_arr.mean(axis=0)
    pos_arr -= center
    positions = pos_arr.tolist()

    initial_L = 6.0

    def _objective(params):
        elapsed = time.time() - start_time
        if elapsed > deadline:
            return 1e10

        L = max(params[0], 1.0)

        all_hexes = []
        for i in range(N):
            x = params[1 + 3 * i]
            y = params[2 + 3 * i]
            a = params[3 + 3 * i]
            verts = _hex_vertices(x, y, 1.0, a)
            all_hexes.append(verts)

        # Containment violation
        max_contain_viol = 0.0
        for verts in all_hexes:
            for (vx, vy) in verts:
                viol = _point_hex_violation(vx, vy, L)
                max_contain_viol = max(max_contain_viol, viol)

        # Overlap violation
        max_overlap = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dist = _hex_min_distance(all_hexes[i], all_hexes[j])
                if dist < 0:
                    max_overlap = max(max_overlap, -dist)

        return L + 5.0 * max_contain_viol + 5.0 * max_overlap

    init_params = [initial_L]
    for pos, ang in zip(positions, angles):
        init_params.extend([pos[0], pos[1], ang])

    best_params = None
    best_score = float('inf')

    n_restarts = 8
    for restart in range(n_restarts):
        if time.time() > deadline - 5:
            break

        if restart == 0:
            params0 = init_params.copy()
        else:
            params0 = init_params.copy()
            params0[0] = initial_L + np.random.uniform(-1.0, 1.0)
            for i in range(1, len(params0), 3):
                params0[i] += np.random.uniform(-0.5, 0.5)
                params0[i + 1] += np.random.uniform(-0.5, 0.5)
                params0[i + 2] += np.random.uniform(-30, 30)

        result = minimize(_objective, params0, method='L-BFGS-B',
                          options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})

        if result.fun < best_score:
            best_score = result.fun
            best_params = result.x.copy()

        # Local refinement from current best
        if best_params is not None and restart > 0:
            if time.time() > deadline - 10:
                break
            perturbed = best_params + np.random.normal(0, 0.01, len(best_params))
            result2 = minimize(_objective, perturbed, method='L-BFGS-B',
                               options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            if result2.fun < best_score:
                best_score = result2.fun
                best_params = result2.x.copy()

    if best_params is None:
        return [[pos[0], pos[1], ang] for pos, ang in zip(positions, angles)], (0.0, 0.0), initial_L, 0.0

    L = best_params[0]
    inner = []
    for i in range(N):
        x = best_params[1 + 3 * i]
        y = best_params[2 + 3 * i]
        a = best_params[3 + 3 * i]
        inner.append([x, y, a])

    return inner, (0.0, 0.0), L, 0.0
