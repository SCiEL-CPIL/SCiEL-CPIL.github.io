# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull


def _hex_vertices(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    ang = a + np.deg2rad(60 * np.arange(6))
    return np.stack([cx + s * np.cos(ang), cy + s * np.sin(ang)], axis=1)


def _hex_vertices_batch(inner_data):
    """Compute vertices for all inner hexagons at once."""
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
    normals = np.array([
        [np.cos(a_rad + np.deg2rad(30 + 60 * k)), np.sin(a_rad + np.deg2rad(30 + 60 * k))]
        for k in range(6)
    ])
    verts = _hex_vertices_batch(inner)
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


def _pairwise_min_gap(inner_data):
    """Compute minimum gap between any pair of hexagons (negative if overlapping)."""
    n = len(inner_data)
    min_gap = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1, a1 = inner_data[i]
            x2, y2, a2 = inner_data[j]
            v1 = _hex_vertices(x1, y1, 1.0, a1)
            v2 = _hex_vertices(x2, y2, 1.0, a2)
            n1 = _hex_normals(a1)
            n2 = _hex_normals(a2)
            all_normals = np.vstack([n1, n2])
            for d in all_normals:
                p1 = v1 @ d
                p2 = v2 @ d
                gap1 = min(p1) - max(p2)
                gap2 = min(p2) - max(p1)
                min_gap = min(min_gap, gap1, gap2)
    return min_gap


def _soft_min_side_smooth(inner_data, outer_angle_deg, beta=200.0):
    """Smooth approximation of the minimum outer hex side using log-sum-exp."""
    a_rad = np.deg2rad(outer_angle_deg)
    normals = np.array([
        [np.cos(a_rad + np.deg2rad(30 + 60 * k)), np.sin(a_rad + np.deg2rad(30 + 60 * k))]
        for k in range(6)
    ])
    verts = _hex_vertices_batch(inner_data)
    projs = np.abs(np.einsum('ijx,kx->ijk', verts, normals))
    all_projs = projs.flatten()
    m = np.max(all_projs)
    log_sum_exp = m + np.log(np.sum(np.exp(all_projs - m)))
    smooth_max_proj = log_sum_exp / beta
    return smooth_max_proj / (np.sqrt(3) / 2)


def _smooth_objective(params, beta=200.0):
    """Smooth objective for optimization.
    params: 34 values = 11*(x, y, angle_deg) + 1 (outer_angle_deg)
    """
    if len(params) != 34:
        return 1e6
    inner = params[:33].reshape(11, 3)
    outer_angle = params[33]
    
    if not np.all(np.isfinite(inner)) or not np.isfinite(outer_angle):
        return 1e6
    
    if np.max(np.abs(inner[:, :2])) > 20:
        return 1e6
    
    side = _soft_min_side_smooth(inner, outer_angle, beta)
    
    # Penalty for overlapping hexagons
    min_gap = _pairwise_min_gap(inner)
    if min_gap < 0:
        penalty = (-min_gap) ** 2
    else:
        penalty = 0.0
    
    return side + 10.0 * penalty


def _exact_objective(params):
    """Exact objective for final polishing."""
    if len(params) != 34:
        return 1e6
    inner = params[:33].reshape(11, 3)
    outer_angle = params[33]
    
    if not np.all(np.isfinite(inner)) or not np.isfinite(outer_angle):
        return 1e6
    
    side = _compute_min_side(inner, outer_angle)
    outer = np.array([0.0, 0.0, outer_angle])
    
    if not _valid(inner, outer, side):
        return side + 50.0
    
    return side


def _gradient_free_search(inner_init, outer_angle_init, n_iter=5000, step=0.1):
    """Custom gradient-free optimization using simulated annealing-like approach."""
    best_params = np.append(inner_init.flatten(), outer_angle_init)
    best_val = _smooth_objective(best_params)
    current_params = best_params.copy()
    current_val = best_val
    
    T = 1.0
    T_min = 0.001
    cooling = (T_min / T) ** (1.0 / n_iter)
    
    for i in range(n_iter):
        if i % 1000 == 0 and i > 0:
            T *= cooling ** 1000
            if T < T_min:
                T = T_min
        
        # Random perturbation
        delta = np.random.randn(34) * step
        new_params = current_params + delta
        
        new_val = _smooth_objective(new_params)
        
        # Accept if better or with probability exp(-delta/T)
        if new_val < current_val or np.random.rand() < np.exp(-(new_val - current_val) / max(T, 1e-10)):
            current_params = new_params
            current_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_params = new_params.copy()
        
        T *= cooling
    
    return best_params, best_val


def _build_inner_7fold(r1, r2, gangle):
    """Build 11 hexagons with 7-fold symmetry: 1 center + 6 ring + 4 at 7-fold positions."""
    inner = [[0.0, 0.0, gangle]]
    for k in range(6):
        ang = np.deg2rad(60 * k)
        x = r1 * np.cos(ang)
        y = r1 * np.sin(ang)
        inner.append([x, y, gangle])
    # 4 outer hexagons at angles that break 6-fold symmetry slightly
    for k in [0, 1, 3, 4]:
        ang = np.deg2rad(60 * k + 15)
        x = r2 * np.cos(ang)
        y = r2 * np.sin(ang)
        inner.append([x, y, gangle])
    return np.array(inner)


def _build_inner_5fold(r1, r2, gangle):
    """Build 11 hexagons: 1 center + 5 ring + 5 outer."""
    inner = [[0.0, 0.0, gangle]]
    for k in range(5):
        ang = np.deg2rad(72 * k)
        x = r1 * np.cos(ang)
        y = r1 * np.sin(ang)
        inner.append([x, y, gangle])
    for k in range(5):
        ang = np.deg2rad(72 * k + 36)
        x = r2 * np.cos(ang)
        y = r2 * np.sin(ang)
        inner.append([x, y, gangle])
    return np.array(inner)


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
    
    # Phase 1: Optimize symmetric family with multiple starting points
    initial_guesses = [
        [2.5, 4.5, 0.0],
        [2.5, 4.5, 30.0],
        [2.6, 4.6, 0.0],
        [2.4, 4.4, 0.0],
        [2.5, 4.8, 15.0],
        [2.7, 4.7, 0.0],
        [2.5, 4.5, 45.0],
        [2.5, 4.5, 60.0],
    ]
    
    for guess in initial_guesses:
        if time.time() > deadline - 180:
            break
        try:
            inner_init = _build_inner_symmetric(*guess)
            params_init = np.append(inner_init.flatten(), guess[2])
            
            res = minimize(
                _smooth_objective,
                params_init,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-7, 'adaptive': True}
            )
            if res.fun < best_side and res.fun < 10.0:
                best_side = res.fun
                best_params = res.x.copy()
        except Exception:
            continue
    
    # Phase 2: Try different symmetry families
    if time.time() < deadline - 150:
        alt_guesses = [
            ('7fold', 2.5, 4.5, 0.0),
            ('7fold', 2.5, 4.5, 30.0),
            ('5fold', 2.5, 4.5, 0.0),
            ('5fold', 2.5, 4.5, 36.0),
        ]
        
        for sym_type, r1, r2, gangle in alt_guesses:
            if time.time() > deadline - 120:
                break
            try:
                if sym_type == '7fold':
                    inner_init = _build_inner_7fold(r1, r2, gangle)
                else:
                    inner_init = _build_inner_5fold(r1, r2, gangle)
                params_init = np.append(inner_init.flatten(), gangle)
                
                res = minimize(
                    _smooth_objective,
                    params_init,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-7, 'fatol': 1e-7, 'adaptive': True}
                )
                if res.fun < best_side and res.fun < 10.0:
                    best_side = res.fun
                    best_params = res.x.copy()
            except Exception:
                continue
    
    # Phase 3: Polish with exact objective
    if best_params is not None and time.time() < deadline - 60:
        try:
            res2 = minimize(
                _exact_objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True}
            )
            if res2.fun < best_side and res2.fun < 10.0:
                best_side = res2.fun
                best_params = res2.x.copy()
        except Exception:
            pass
    
    # Phase 4: Gradient-free search from best point
    if best_params is not None and time.time() < deadline - 30:
        try:
            gf_params, gf_val = _gradient_free_search(
                best_params[:33].reshape(11, 3),
                best_params[33],
                n_iter=3000,
                step=0.05
            )
            if gf_val < best_side and gf_val < 10.0:
                best_side = gf_val
                best_params = gf_params.copy()
        except Exception:
            pass
    
    # Phase 5: Additional multi-start from perturbed configs
    if time.time() < deadline - 20:
        np.random.seed(42)
        for attempt in range(3):
            if time.time() > deadline - 15:
                break
            base_guess = [2.5, 4.5, 0.0]
            base_inner = _build_inner_symmetric(*base_guess)
            perturbed = base_inner + np.random.randn(11, 3) * 0.3
            perturbed[:, 2] = np.degrees(np.arctan2(np.sin(np.deg2rad(perturbed[:, 2])), np.cos(np.deg2rad(perturbed[:, 2]))))
            params = np.append(perturbed.flatten(), 0.0)
            try:
                res3 = minimize(
                    _smooth_objective,
                    params,
                    method='Nelder-Mead',
                    options={'maxiter': 1500, 'xatol': 1e-7, 'fatol': 1e-7, 'adaptive': True}
                )
                if res3.fun < best_side and res3.fun < 10.0:
                    best_side = res3.fun
                    best_params = res3.x.copy()
            except Exception:
                continue
    
    # Final validation
    if best_params is not None:
        inner = best_params[:33].reshape(11, 3)
        outer_angle = best_params[33]
        outer = np.array([0.0, 0.0, outer_angle])
        side = _compute_min_side(inner, outer_angle)
        
        if _valid(inner, outer, side):
            best_inner = inner
            best_outer = outer
            best_side = side
        else:
            for ds in [0.01, 0.05, 0.1, 0.5]:
                test_side = side + ds
                if _valid(inner, outer, test_side):
                    best_inner = inner
                    best_outer = outer
                    best_side = test_side
                    break
    
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
                side = 9.0
        
        return inner, outer, side
    
    return best_inner, best_outer, best_side


# EVOLVE-BLOCK-END
