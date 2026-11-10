import numpy as np
import time
from scipy.optimize import minimize
import itertools

def hexagon_packing_11():
    start_time = time.time()
    deadline = 630.0

    # Unit hexagon vertices (angle 0): vertices at 0,60,...,300 degrees
    hex_verts = np.array(
        [[np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))] for a in range(0, 360, 60)]
    )

    def get_vertices(center, angle_deg):
        a = np.deg2rad(angle_deg)
        rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        return np.asarray(center) + hex_verts @ rot.T

    def get_normals(vertices):
        normals = []
        for i in range(6):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % 6]
            e = v2 - v1
            n = np.array([e[1], -e[0]])
            n = n / (np.linalg.norm(n) + 1e-15)
            normals.append(n)
        return np.array(normals)

    def max_penetration(vertices1, vertices2):
        # SAT: returns max penetration depth (0 if disjoint)
        normals1 = get_normals(vertices1)
        normals2 = get_normals(vertices2)
        all_normals = np.vstack([normals1, normals2])
        max_pen = 0.0
        for n in all_normals:
            p1 = vertices1 @ n
            p2 = vertices2 @ n
            if np.max(p1) < np.min(p2) or np.max(p2) < np.min(p1):
                return 0.0  # gap on this normal -> disjoint
            overlap = min(np.max(p1), np.max(p2)) - max(np.min(p1), np.min(p2))
            if overlap > max_pen:
                max_pen = overlap
        return max_pen

    def compute_side(centers, angles, theta_outer_rad):
        # support in each of 6 outer normal directions = max center proj + hex support
        normals = np.array(
            [[np.cos(theta_outer_rad + np.deg2rad(60 * i)),
              np.sin(theta_outer_rad + np.deg2rad(60 * i))] for i in range(6)]
        )
        center_proj = centers @ normals.T
        max_center = np.max(center_proj, axis=0)
        hex_proj = hex_verts @ normals.T
        max_hex = np.max(hex_proj, axis=1)
        support = max_center + max_hex
        return (2.0 / np.sqrt(3.0)) * np.max(support)

    def objective(params):
        if time.time() > deadline - 15:
            return 1e10
        data = params[:33].reshape(11, 3)
        theta_outer_deg = params[33]
        centers = data[:, :2]
        angles = data[:, 2]

        all_vertices = [get_vertices(centers[i], angles[i]) for i in range(11)]

        pen_sum = 0.0
        for i in range(11):
            for j in range(i + 1, 11):
                pen = max_penetration(all_vertices[i], all_vertices[j])
                if pen > 0:
                    pen_sum += pen * pen

        side = compute_side(centers, angles, np.deg2rad(theta_outer_deg))
        return side + 1000.0 * pen_sum

    # ---- Build tight honeycomb base: center + ring1 (6) ----
    d = np.sqrt(3.0)
    base_centers = [np.array([0.0, 0.0])]
    base_angles = [0.0]
    for i in range(6):
        a = np.deg2rad(30 + 60 * i)
        base_centers.append([d * np.cos(a), d * np.sin(a)])
        base_angles.append(0.0)
    base_centers = np.array(base_centers)
    base_angles = np.array(base_angles)

    # ---- Ring2 candidate points (12 positions) ----
    v = np.array([d * np.cos(np.deg2rad(60 * i)), d * np.sin(np.deg2rad(60 * i))]
                 for i in range(6))
    ring2 = []
    for i in range(6):
        ring2.append(2 * v[i])          # even index: 0,60,...,300 deg
        ring2.append(v[i] + v[(i + 1) % 6])  # odd index: 30,90,...,330 deg
    ring2 = np.array(ring2)

    # Symmetric 4-subsets of ring2 (indices into the 12 ring2 points)
    subsets = [
        [0, 3, 6, 9],
        [0, 2, 6, 8],
        [1, 4, 7, 10],
        [0, 5, 6, 11],
        [1, 3, 7, 9],
        [2, 4, 8, 10],
        [0, 3, 8, 11],
        [1, 2, 7, 10],
        [1, 5, 7, 11],
        [0, 4, 6, 10],
    ]

    candidates = []
    for subset in subsets:
        r2 = ring2[subset]
        centers = np.vstack([base_centers, r2])
        angles = np.concatenate([base_angles, [0.0] * 4])
        init = list(centers.ravel()) + list(angles) + [0.0]
        candidates.append(np.array(init))

    # A few randomized perturbations for diversity
    np.random.seed(42)
    for _ in range(6):
        centers = base_centers.copy()
        angles = base_angles.copy()
        for i in range(4):
            a = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(2.6, 3.6)
            centers = np.vstack([centers, [r * np.cos(a), r * np.sin(a)]])
            angles = np.concatenate([angles, [np.random.uniform(0, 60)]])
        init = list(centers.ravel()) + list(angles) + [np.random.uniform(-30, 30)]
        candidates.append(np.array(init))

    best_result = None
    best_side = float('inf')

    for init in candidates:
        if time.time() > deadline - 70:
            break
        try:
            res = minimize(objective, init, method='Nelder-Mead',
                           options={'maxiter': 1500, 'xatol': 1e-7,
                                    'fatol': 1e-7, 'disp': False})
            data = res.x[:33].reshape(11, 3)
            theta_outer_deg = res.x[33]
            centers = data[:, :2]
            angles = data[:, 2]
            all_vertices = [get_vertices(centers[i], angles[i]) for i in range(11)]

            valid = True
            for i in range(11):
                for j in range(i + 1, 11):
                    if max_penetration(all_vertices[i], all_vertices[j]) > 1e-6:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                continue

            # Fine-tune outer angle to minimize side
            theta0 = np.deg2rad(theta_outer_deg)
            thetas = theta0 + np.linspace(-np.pi / 6, np.pi / 6, 240)
            best_theta = theta0
            min_side = float('inf')
            for th in thetas:
                s = compute_side(centers, angles, th)
                if s < min_side:
                    min_side = s
                    best_theta = th

            if min_side < best_side:
                best_side = min_side
                inner_hex_data = np.column_stack([centers, angles])
                outer_hex_data = np.array([0.0, 0.0, np.rad2deg(best_theta)])
                best_result = (inner_hex_data, outer_hex_data, min_side)
        except Exception:
            pass

    if best_result is not None:
        return best_result

    # Fallback: loose grid (known valid, side 8)
    inner_hex_data = np.array(
        [
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.17, 0],
            [1.25, 2.17, 0],
            [-1.25, -2.17, 0],
            [1.25, -2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, -2.17, 0],
            [3.75, -2.17, 0],
        ]
    )
    outer_hex_data = np.array([0, 0, 0])
    return inner_hex_data, outer_hex_data, 8.0
