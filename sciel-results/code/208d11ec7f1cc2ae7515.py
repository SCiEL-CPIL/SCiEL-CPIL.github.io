# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution


def _hex_vertices(cx, cy, d, angle_deg):
    angles = np.deg2rad(angle_deg) + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


def _min_containing_hex_side(inner_verts, outer_center, outer_angle_deg):
    oc = np.array(outer_center)
    oa = np.deg2rad(outer_angle_deg)
    base_angles = oa + np.deg2rad(60.0 * np.arange(6))
    unit_verts = np.column_stack([np.cos(base_angles), np.sin(base_angles)])
    rel = inner_verts - oc
    edge_vecs = np.roll(unit_verts, -1, axis=0) - unit_verts
    A = edge_vecs[:, 0:1] * rel[1:2, :] - edge_vecs[:, 1:2] * rel[0:1, :]
    s_min = np.max(-2.0 * A / np.sqrt(3.0))
    return max(s_min, 0.0)


def _get_configurations():
    sqrt3 = np.sqrt(3.0)
    configs = []
    axial_sets = [
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(1,1),(0,2),(-1,2),(-2,1)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(1,1),(0,2),(-2,0),(-2,1)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(1,1),(0,2),(-2,0),(-1,-1)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(1,1),(0,2),(-1,2),(-2,2)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(1,1),(0,2),(-1,2),(-2,0)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(1,1),(0,2),(-1,2),(0,-2)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(1,1),(0,2),(-1,2),(2,-1)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(1,1),(0,2),(-1,2),(1,-2)],
    ]
    for axial in axial_sets:
        pos = []
        for q, r in axial:
            x = 1.5 * q
            y = sqrt3 * (r + 0.5 * q)
            pos.append([x, y, 0.0])
        configs.append(np.array(pos))
    return configs


def _check_separation_fast(inner_positions):
    n = len(inner_positions)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(inner_positions[i, :2] - inner_positions[j, :2])
            if dist < np.sqrt(3.0) - 1e-9:
                return False
    return True


def _validate_result(inner_hex_data, outer_hex_data, s):
    if not np.all(np.isfinite(inner_hex_data)) or not np.all(np.isfinite(outer_hex_data)):
        return False
    if s < np.sqrt(12.0):
        return False
    outer_verts = _hex_vertices(outer_hex_data[0], outer_hex_data[1], s, outer_hex_data[2])
    for i in range(6):
        for j in range(i + 1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-8:
                return False
    oc = outer_hex_data[:2]
    shifted_inner = inner_hex_data.copy()
    shifted_inner[:, :2] -= oc
    for pos in shifted_inner:
        verts = _hex_vertices(pos[0], pos[1], 1.0, pos[2])
        for i in range(6):
            edge_len = np.linalg.norm(verts[(i + 1) % 6] - verts[i])
            if abs(edge_len - 1.0) > 1e-8:
                return False
    all_verts = []
    for pos in shifted_inner:
        verts = _hex_vertices(pos[0], pos[1], 1.0, pos[2])
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    ov = _hex_vertices(0, 0, s, outer_hex_data[2])
    for v in all_verts:
        for i in range(6):
            p1 = ov[i]
            p2 = ov[(i + 1) % 6]
            edge = p2 - p1
            to_v = v - p1
            cross_val = edge[0] * to_v[1] - edge[1] * to_v[0]
            if cross_val < -1e-6:
                return False
    return True


def _free_body_objective(params, n_hex=12):
    """Objective for free-body optimization: minimize outer hexagon side length."""
    inner = params[:n_hex * 3].reshape(n_hex, 3)
    outer_cx, outer_cy, outer_angle = params[n_hex * 3], params[n_hex * 3 + 1], params[n_hex * 3 + 2]
    
    all_verts = []
    for pos in inner:
        verts = _hex_vertices(pos[0], pos[1], 1.0, pos[2])
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    s = _min_containing_hex_side(all_verts, [outer_cx, outer_cy], outer_angle)
    
    # Separation penalty
    sep_penalty = 0.0
    for i in range(n_hex):
        for j in range(i + 1, n_hex):
            dist = np.linalg.norm(inner[i, :2] - inner[j, :2])
            if dist < np.sqrt(3.0) - 1e-6:
                sep_penalty += (np.sqrt(3.0) - dist) ** 2
    return s + 100.0 * sep_penalty


def _free_body_optimize(config, deadline):
    """Optimize a configuration with free-body optimization."""
    n_hex = len(config)
    x0 = np.concatenate([config.ravel(), [0.0, 0.0, 0.0]])
    
    def objective(params):
        return _free_body_objective(params, n_hex)
    
    best_s = np.inf
    best_params = None
    
    # Nelder-Mead from multiple starts
    angle_starts = np.linspace(0, 60, 7)
    for a0 in angle_starts:
        if time.time() > deadline - 5:
            break
        x0_try = x0.copy()
        x0_try[-1] = a0
        try:
            result = minimize(objective, x0_try, method='Nelder-Mead',
                             options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-11})
            if result.fun < best_s:
                best_s = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Refine best
    if best_params is not None and time.time() < deadline - 2:
        try:
            result = minimize(objective, best_params, method='Nelder-Mead',
                             options={'maxiter': 10000, 'xatol': 1e-11, 'fatol': 1e-13})
            if result.fun < best_s:
                best_s = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        inner = config.copy()
        outer = np.array([0.0, 0.0, 0.0])
        all_verts = np.vstack([_hex_vertices(p[0], p[1], 1.0, p[2]) for p in inner])
        best_s = _min_containing_hex_side(all_verts, [0.0, 0.0], 0.0)
        best_params = np.concatenate([config.ravel(), [0.0, 0.0, 0.0]])
    
    return best_s, best_params


def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 600.0
    
    configs = _get_configurations()
    
    best_s = np.inf
    best_inner = None
    best_outer = None
    
    for idx, config in enumerate(configs):
        if time.time() > deadline - 30:
            break
        if not _check_separation_fast(config):
            continue
        s, params = _free_body_optimize(config, deadline)
        if s < best_s:
            best_s = s
            best_inner = params[:36].reshape(12, 3).copy()
            best_outer = params[36:39].copy()
    
    if best_inner is None:
        config = configs[0]
        all_verts = np.vstack([_hex_vertices(p[0], p[1], 1.0, p[2]) for p in config])
        best_s = _min_containing_hex_side(all_verts, [0.0, 0.0], 0.0)
        best_inner = config.copy()
        best_outer = np.array([0.0, 0.0, 0.0])
    
    if not _validate_result(best_inner, best_outer, best_s):
        centroid = np.mean(best_inner[:, :2], axis=0)
        best_outer = np.array([centroid[0], centroid[1], 0.0])
        all_verts = np.vstack([_hex_vertices(p[0], p[1], 1.0, p[2]) for p in best_inner])
        best_s = _min_containing_hex_side(all_verts, [centroid[0], centroid[1]], 0.0)
        if not _validate_result(best_inner, best_outer, best_s):
            best_inner = configs[0].copy()
            best_outer = np.array([0.0, 0.0, 0.0])
            all_verts = np.vstack([_hex_vertices(p[0], p[1], 1.0, p[2]) for p in best_inner])
            best_s = _min_containing_hex_side(all_verts, [0.0, 0.0], 0.0)
    
    return best_inner, best_outer, best_s
# EVOLVE-BLOCK-END
