# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _hex_vertices(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(np.arange(6) * 60)
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _hex_normals(vertices):
    n = len(vertices)
    normals = np.zeros((n, 2))
    for i in range(n):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % n]
        edge = p2 - p1
        normal = np.array([edge[1], -edge[0]])
        norm = np.linalg.norm(normal)
        if norm > 1e-15:
            normals[i] = normal / norm
    return normals

def _check_inner_separation(inner_data):
    n = len(inner_data)
    for i in range(n):
        v1 = _hex_vertices(inner_data[i, 0], inner_data[i, 1], 1.0, inner_data[i, 2])
        n1 = _hex_normals(v1)
        for j in range(i + 1, n):
            v2 = _hex_vertices(inner_data[j, 0], inner_data[j, 1], 1.0, inner_data[j, 2])
            n2 = _hex_normals(v2)
            all_normals = np.vstack([n1, n2])
            found_separation = False
            for k in range(12):
                nrm = all_normals[k]
                proj1 = v1 @ nrm
                proj2 = v2 @ nrm
                if max(proj1) < min(proj2) - 1e-6 or max(proj2) < min(proj1) - 1e-6:
                    found_separation = True
                    break
            if not found_separation:
                return False
    return True

def _validate(inner, outer, side):
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)):
        return False
    if side < np.sqrt(11) - 1e-12:
        return False
    if not _check_inner_separation(inner):
        return False
    outer_v = _hex_vertices(outer[0], outer[1], side, outer[2])
    n_outer = len(outer_v)
    for i in range(11):
        inner_v = _hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2])
        for p in inner_v:
            for e in range(n_outer):
                p1 = outer_v[e]
                p2 = outer_v[(e + 1) % n_outer]
                cross_val = np.cross(p2 - p1, p - p1)
                if cross_val < -1e-6:
                    return False
    return True

def _compute_outer_side(inner_positions):
    """Compute minimum side length of a regular hexagon (centered at origin, angle 0) that contains all inner hexagons."""
    sqrt3 = np.sqrt(3)
    max_s = 0
    for pos in inner_positions:
        cx, cy, angle = pos
        v = _hex_vertices(cx, cy, 1.0, angle)
        for px, py in v:
            s1 = (sqrt3/2) * px + 0.5 * py
            s2 = py / (sqrt3/2)
            s3 = (-sqrt3/2) * px + 0.5 * py
            s4 = (-sqrt3/2) * px - 0.5 * py
            s5 = -py / (sqrt3/2)
            s6 = (sqrt3/2) * px - 0.5 * py
            max_s = max(max_s, s1, s2, s3, s4, s5, s6)
    return max_s

def _overlap_penalty(inner):
    """Compute penalty for overlapping inner hexagons."""
    penalty = 0
    for i in range(11):
        v1 = _hex_vertices(inner[i,0], inner[i,1], 1.0, inner[i,2])
        n1 = _hex_normals(v1)
        for j in range(i+1, 11):
            v2 = _hex_vertices(inner[j,0], inner[j,1], 1.0, inner[j,2])
            n2 = _hex_normals(v2)
            all_normals = np.vstack([n1, n2])
            separated = False
            for k in range(12):
                nrm = all_normals[k]
                proj1 = v1 @ nrm
                proj2 = v2 @ nrm
                if max(proj1) < min(proj2) - 1e-6 or max(proj2) < min(proj1) - 1e-6:
                    separated = True
                    break
            if not separated:
                for k in range(12):
                    nrm = all_normals[k]
                    proj1 = v1 @ nrm
                    proj2 = v2 @ nrm
                    gap = min(proj2) - max(proj1)
                    if gap < 0:
                        penalty += (-gap) ** 2
                break
    return penalty

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 600

    sqrt3 = np.sqrt(3)
    # Initial: compact hexagonal cluster
    init_positions = [
        [0, 0, 0],
        [sqrt3, 0, 0],
        [-sqrt3, 0, 0],
        [sqrt3/2, 1.5, 0],
        [-sqrt3/2, 1.5, 0],
        [sqrt3/2, -1.5, 0],
        [-sqrt3/2, -1.5, 0],
        [sqrt3, 3.0, 0],
        [-sqrt3, 3.0, 0],
        [sqrt3, -3.0, 0],
        [-sqrt3, -3.0, 0],
    ]

    best_inner = np.array(init_positions)
    best_outer = np.array([0.0, 0.0, 0.0])
    best_side = _compute_outer_side(best_inner)

    def objective(params):
        inner = params.reshape(11, 3)
        side = _compute_outer_side(inner)
        penalty = _overlap_penalty(inner)
        return side + 1000 * penalty

    x0 = np.array(init_positions).flatten()

    # Main optimization
    try:
        res = minimize(objective, x0, method='Powell',
                      options={'maxiter': 10000, 'maxfev': 100000, 'xtol': 1e-10, 'ftol': 1e-10})
        if res.fun < best_side:
            cand = res.x.reshape(11, 3)
            cand_side = _compute_outer_side(cand)
            if cand_side < best_side:
                best_inner = cand
                best_side = cand_side
    except Exception:
        pass

    # Random restarts
    np.random.seed(123)
    for _ in range(30):
        if time.time() > deadline - 30:
            break
        x0_r = x0 + np.random.randn(33) * 0.3
        try:
            res_r = minimize(objective, x0_r, method='Powell',
                           options={'maxiter': 3000, 'maxfev': 30000, 'xtol': 1e-10, 'ftol': 1e-10})
            if res_r.fun < best_side:
                cand = res_r.x.reshape(11, 3)
                cand_side = _compute_outer_side(cand)
                if cand_side < best_side:
                    best_inner = cand
                    best_side = cand_side
        except Exception:
            continue

    # Center
    center = np.mean(best_inner[:, :2], axis=0)
    best_inner[:, 0] -= center[0]
    best_inner[:, 1] -= center[1]
    best_side = _compute_outer_side(best_inner)

    if not _validate(best_inner, best_outer, best_side):
        best_inner = np.array(init_positions)
        best_outer = np.array([0.0, 0.0, 0.0])
        best_side = _compute_outer_side(best_inner)

    return best_inner, best_outer, best_side

# EVOLVE-BLOCK-END
