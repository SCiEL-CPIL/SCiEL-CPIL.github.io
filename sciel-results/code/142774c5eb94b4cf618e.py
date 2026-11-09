# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _hex_vertices(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    ang = a + np.deg2rad(60 * np.arange(6))
    return np.stack([cx + s * np.cos(ang), cy + s * np.sin(ang)], axis=1)


def _hex_vertices_batch(inner_data, outer_data=None, side=1.0, outer_side=None):
    """Compute vertices for all inner hexagons at once.
    inner_data: (11, 3) array of (x, y, angle_deg)
    Returns (11, 6, 2) array of vertices.
    """
    x = inner_data[:, 0]
    y = inner_data[:, 1]
    a = np.deg2rad(inner_data[:, 2])
    ang = a[:, np.newaxis] + np.deg2rad(60 * np.arange(6))[np.newaxis, :]
    verts = np.stack([
        x[:, np.newaxis] + 1.0 * np.cos(ang),
        y[:, np.newaxis] + 1.0 * np.sin(ang)
    ], axis=2)
    return verts


def _hex_normals(a_deg):
    a = np.deg2rad(a_deg)
    base = a + np.deg2rad(60 * np.arange(6))
    normals = base + np.deg2rad(30)
    dirs = np.stack([np.cos(normals), np.sin(normals)], axis=1)
    return np.vstack([dirs, -dirs])


def _separation_ok(v1, n1, v2, n2):
    for d in n1:
        p1 = v1 @ d
        p2 = v2 @ d
        gap = min(p1) - max(p2)
        if gap > 1e-6:
            return True
        gap = min(p2) - max(p1)
        if gap > 1e-6:
            return True
    for d in n2:
        p1 = v1 @ d
        p2 = v2 @ d
        gap = min(p1) - max(p2)
        if gap > 1e-6:
            return True
        gap = min(p2) - max(p1)
        if gap > 1e-6:
            return True
    return False


def _hexagons_separated(inner_data):
    verts = []
    norms = []
    for i in range(len(inner_data)):
        x, y, a = inner_data[i]
        v = _hex_vertices(x, y, 1.0, a)
        n = _hex_normals(a)
        verts.append(v)
        norms.append(n)
    for i in range(len(inner_data)):
        for j in range(i + 1, len(inner_data)):
            if not _separation_ok(verts[i], norms[i], verts[j], norms[j]):
                return False
    return True


def _contains_in_hex(inner_data, outer_data, side):
    ox, oy, oa = outer_data
    outer_v = _hex_vertices(ox, oy, side, oa)
    tol = -1e-6
    for i in range(len(inner_data)):
        x, y, a = inner_data[i]
        v = _hex_vertices(x, y, 1.0, a)
        for p in v:
            for e in range(6):
                p1 = outer_v[e]
                p2 = outer_v[(e + 1) % 6]
                cross = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
                if cross < tol:
                    return False
    return True


def _valid(inner_data, outer_data, side):
    if not np.all(np.isfinite(inner_data)) or not np.all(np.isfinite(outer_data)):
        return False
    if side < np.sqrt(11) - 1e-12:
        return False
    if not _hexagons_separated(inner_data):
        return False
    if not _contains_in_hex(inner_data, outer_data, side):
        return False
    for i in range(len(inner_data)):
        x, y, a = inner_data[i]
        v = _hex_vertices(x, y, 1.0, a)
        for e in range(6):
            p1 = v[e]
            p2 = v[(e + 1) % 6]
            if abs(np.linalg.norm(p2 - p1) - 1.0) > 1e-8:
                return False
    return True


def _compute_min_side(inner, outer_angle_deg):
    """Compute minimum outer hex side to contain all inner hexagons."""
    a_rad = np.deg2rad(outer_angle_deg)
    normals = []
    for k in range(6):
        ang = a_rad + np.deg2rad(30 + 60 * k)
        normals.append([np.cos(ang), np.sin(ang)])
    normals = np.array(normals)
    
    verts = _hex_vertices_batch(inner)
    # verts shape: (11, 6, 2)
    # Project all vertices onto all 6 normals
    # projs[i, j, k] = vertex j of hex i projected onto normal k
    # Using 'ijx,kx->ijk' to handle the 3D verts array correctly
    projs = np.abs(np.einsum('ijx,kx->ijk', verts, normals))
    max_proj = np.max(projs)
    
    return max_proj / (np.sqrt(3) / 2)


def _build_inner_symmetric(r1, r2, gangle):
    """Build 11 hexagons with 6-fold symmetry: 1 center + 6 ring + 4 outer."""
    inner = [[0.0, 0.0, gangle]]
    for k in range(6):
        ang = np.deg2rad(60 * k)
        x = r1 * np.cos(ang)
        y = r1 * np.sin(ang)
        inner.append([x, y, gangle])
    for k in [0, 2, 3, 5]:
        ang = np.deg2rad(60 * k + 30)
        x = r2 * np.cos(ang)
        y = r2 * np.sin(ang)
        inner.append([x, y, gangle])
    return np.array(inner)


def _objective(params):
    """Objective: minimize outer hex side length.
    params = [r1, r2, gangle_deg]
    """
    r1, r2, gangle = params
    if r1 < 1.5 or r2 < r1 or r1 > 10 or r2 > 10:
        return 100.0 + abs(r1) + abs(r2)
    
    inner = _build_inner_symmetric(r1, r2, gangle)
    side = _compute_min_side(inner, gangle)
    
    outer = np.array([0.0, 0.0, gangle])
    if not _valid(inner, outer, side):
        return side + 50.0
    
    return side


def _objective_smooth(params, beta=100.0):
    """Smooth surrogate for the min-side objective using log-sum-exp.
    params = [r1, r2, gangle_deg]
    """
    r1, r2, gangle = params
    if r1 < 1.5 or r2 < r1 or r1 > 10 or r2 > 10:
        return 100.0 + abs(r1) + abs(r2)
    
    inner = _build_inner_symmetric(r1, r2, gangle)
    a_rad = np.deg2rad(gangle)
    normals = []
    for k in range(6):
        ang = a_rad + np.deg2rad(30 + 60 * k)
        normals.append([np.cos(ang), np.sin(ang)])
    normals = np.array(normals)
    
    verts = _hex_vertices_batch(inner)
    # Compute max projection per hex per normal
    # Using 'ijx,kx->ijk' to handle the 3D verts array correctly
    projs = np.abs(np.einsum('ijx,kx->ijk', verts, normals))
    # For each hex and each normal, get max projection
    # We want the max over all hexes and normals
    # Use log-sum-exp for smooth max
    all_projs = projs.flatten()
    m = np.max(all_projs)
    log_sum_exp = m + np.log(np.sum(np.exp(all_projs - m)))
    smooth_max_proj = log_sum_exp / beta
    side = smooth_max_proj / (np.sqrt(3) / 2)
    
    # Add penalty if invalid
    outer = np.array([0.0, 0.0, gangle])
    if not _valid(inner, outer, side):
        return side + 50.0
    
    return side


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon.
    Returns (inner_hex_data, outer_hex_data, outer_hex_side_length).
    """
    t0 = time.time()
    deadline = t0 + 615.0
    
    best_side = float('inf')
    best_inner = None
    best_outer = None
    best_params = None
    
    # Phase 1: Multi-start Nelder-Mead with smooth surrogate
    initial_guesses = [
        [2.5, 4.5, 0.0],
        [2.5, 4.5, 30.0],
        [2.6, 4.6, 0.0],
        [2.4, 4.4, 0.0],
        [2.5, 4.8, 15.0],
        [2.7, 4.7, 0.0],
        [2.5, 5.0, 0.0],
        [3.0, 5.0, 0.0],
        [2.5, 4.5, 60.0],
        [2.6, 4.9, 30.0],
    ]
    
    for guess in initial_guesses:
        if time.time() > deadline - 90:
            break
        try:
            # Use smooth objective for better convergence
            res = minimize(
                _objective_smooth,
                guess,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-7, 'fatol': 1e-7, 'adaptive': True}
            )
            if res.fun < best_side and res.fun < 10.0:
                best_side = res.fun
                best_params = res.x.copy()
        except Exception:
            continue
    
    # Phase 2: Polish with exact objective from best params
    if best_params is not None and time.time() < deadline - 60:
        try:
            res2 = minimize(
                _objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True}
            )
            if res2.fun < best_side and res2.fun < 10.0:
                best_side = res2.fun
                best_params = res2.x.copy()
        except Exception:
            pass
    
    # Phase 3: Fine grid search around best solution
    if best_params is not None and time.time() < deadline - 30:
        r1_best, r2_best, ga_best = best_params
        for dr1 in np.arange(-0.15, 0.16, 0.05):
            for dr2 in np.arange(-0.15, 0.16, 0.05):
                for dga in [-10.0, 0.0, 10.0]:
                    if time.time() > deadline - 20:
                        break
                    cand = np.array([r1_best + dr1, r2_best + dr2, ga_best + dga])
                    side = _objective(cand)
                    if side < best_side and side < 10.0:
                        best_side = side
                        best_params = cand
    
    # Final validation
    if best_params is not None:
        r1, r2, gangle = best_params
        inner = _build_inner_symmetric(r1, r2, gangle)
        outer = np.array([0.0, 0.0, gangle])
        side = _compute_min_side(inner, gangle)
        
        if _valid(inner, outer, side):
            best_inner = inner
            best_outer = outer
            best_side = side
    
    if best_inner is None:
        # Fallback: known valid arrangement
        inner = np.array([
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.165, 0],
            [1.25, 2.165, 0],
            [-1.25, -2.165, 0],
            [1.25, -2.165, 0],
            [-3.75, 2.165, 0],
            [3.75, 2.165, 0],
            [-3.75, -2.165, 0],
            [3.75, -2.165, 0],
        ])
        outer = np.array([0, 0, 0])
        side = _compute_min_side(inner, 0.0)
        if not _valid(inner, outer, side):
            side = 8.0
            if not _valid(inner, outer, side):
                inner = np.array([
                    [0, 0, 0],
                    [-2.5, 0, 0],
                    [2.5, 0, 0],
                    [-1.25, 2.165, 0],
                    [1.25, 2.165, 0],
                    [-1.25, -2.165, 0],
                    [1.25, -2.165, 0],
                    [-3.75, 2.165, 0],
                    [3.75, 2.165, 0],
                    [-3.75, -2.165, 0],
                    [3.75, -2.165, 0],
                ])
                outer = np.array([0, 0, 0])
                side = 9.0
        
        return inner, outer, side
    
    return best_inner, best_outer, best_side


# EVOLVE-BLOCK-END
