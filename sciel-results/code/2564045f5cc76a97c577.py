# EVOLVE-BLOCK-START
import numpy as np
import time


def _hex_vertices(cx, cy, s, a_deg):
    """Vertices of a regular hexagon centered at (cx,cy) with side s and angle a_deg."""
    a = np.deg2rad(a_deg)
    ang = a + np.deg2rad(60.0 * np.arange(6))
    return np.stack([cx + s * np.cos(ang), cy + s * np.sin(ang)], axis=1)


def _hex_normals(a_deg):
    """Return 12 edge normals (6 outward + 6 inward) for a hexagon with given angle."""
    a = np.deg2rad(a_deg)
    base = a + np.deg2rad(30 + 60 * np.arange(6))
    dirs = np.stack([np.cos(base), np.sin(base)], axis=1)
    return np.vstack([dirs, -dirs])


def _separation_ok(v1, n1, v2, n2):
    """Check if two hexagons are separated on at least one of 12 normals."""
    for d in n1:
        p1 = v1 @ d
        p2 = v2 @ d
        if (p1.min() - p2.max()) > 1e-6 or (p2.min() - p1.max()) > 1e-6:
            return True
    for d in n2:
        p1 = v1 @ d
        p2 = v2 @ d
        if (p1.min() - p2.max()) > 1e-6 or (p2.min() - p1.max()) > 1e-6:
            return True
    return False


def _all_separated(inner_data):
    """Check all pairs of inner hexagons are separated."""
    n = len(inner_data)
    verts = []
    norms = []
    for i in range(n):
        x, y, a = inner_data[i]
        verts.append(_hex_vertices(x, y, 1.0, a))
        norms.append(_hex_normals(a))
    for i in range(n):
        for j in range(i + 1, n):
            if not _separation_ok(verts[i], norms[i], verts[j], norms[j]):
                return False
    return True


def _compute_min_side(inner_data, outer_angle_deg):
    """Compute minimum outer hex side to contain all inner hexagons, given outer angle."""
    a_rad = np.deg2rad(outer_angle_deg)
    normals = []
    for k in range(6):
        ang = a_rad + np.deg2rad(30 + 60 * k)
        normals.append([np.cos(ang), np.sin(ang)])
    normals = np.array(normals)
    
    max_proj = 0.0
    for i in range(len(inner_data)):
        x, y, a = inner_data[i]
        v = _hex_vertices(x, y, 1.0, a)
        projs = np.abs(v @ normals.T)
        max_proj = max(max_proj, np.max(projs))
    
    return max_proj / (np.sqrt(3) / 2)


def _contains_in_hex(inner_data, outer_data, side):
    """Check all inner vertices are inside the outer hexagon."""
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
    """Full validity check."""
    if not np.all(np.isfinite(inner_data)) or not np.all(np.isfinite(outer_data)):
        return False
    if side < np.sqrt(11) - 1e-12:
        return False
    if not _all_separated(inner_data):
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


def _build_inner_ring(r_inner, r_outer, gangle):
    """Build 11 hexagons: 1 center + 6 inner ring + 4 outer ring at 30deg offsets."""
    inner = [[0.0, 0.0, gangle]]
    for k in range(6):
        ang = np.deg2rad(60 * k)
        x = r_inner * np.cos(ang)
        y = r_inner * np.sin(ang)
        inner.append([x, y, gangle])
    for k in range(6):
        ang = np.deg2rad(60 * k + 30)
        x = r_outer * np.cos(ang)
        y = r_outer * np.sin(ang)
        inner.append([x, y, gangle])
    # Remove last one to get 11 total (1+6+4)
    inner = inner[:11]
    return np.array(inner)


def _build_inner_hex_cluster(r1, r2, r3, gangle):
    """Build 11 hexagons: 1 center + 6 ring at r1 + 4 at r2 with 30deg offset."""
    inner = [[0.0, 0.0, gangle]]
    for k in range(6):
        ang = np.deg2rad(60 * k)
        x = r1 * np.cos(ang)
        y = r1 * np.sin(ang)
        inner.append([x, y, gangle])
    # 4 additional hexagons at different positions
    for k in [0, 1, 3, 4]:
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


def _objective_ring(params):
    """Objective for ring configuration."""
    r_inner, r_outer, gangle = params
    if r_inner < 1.5 or r_outer < r_inner or r_inner > 10 or r_outer > 10:
        return 100.0 + abs(r_inner) + abs(r_outer)
    
    inner = _build_inner_ring(r_inner, r_outer, gangle)
    if len(inner) != 11:
        return 100.0
    side = _compute_min_side(inner, gangle)
    
    outer = np.array([0.0, 0.0, gangle])
    if not _valid(inner, outer, side):
        return side + 50.0
    
    return side


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is (x, y, angle_degrees)
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree)
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    t_start = time.time()
    deadline = 620.0

    def _time_left():
        return deadline - (time.time() - t_start)

    best_side = float('inf')
    best_inner = None
    best_outer = None
    best_params = None

    from scipy.optimize import minimize

    # Multi-start Nelder-Mead optimization for symmetric configuration
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
        [2.5, 4.2, 0.0],
        [2.8, 4.8, 0.0],
        [2.4, 4.7, 30.0],
        [2.6, 4.4, 0.0],
        [2.55, 4.55, 0.0],
        [2.45, 4.65, 0.0],
        [2.5, 4.4, 10.0],
        [2.5, 4.6, 20.0],
    ]

    for guess in initial_guesses:
        if _time_left() < 60:
            break
        try:
            res = minimize(
                _objective,
                guess,
                method='Nelder-Mead',
                options={'maxiter': 8000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            if res.fun < best_side and res.fun < 10.0:
                best_side = res.fun
                best_params = res.x.copy()
        except Exception:
            continue

    # Grid search for robustness
    if _time_left() > 120:
        for r1 in np.arange(2.2, 3.2, 0.05):
            for r2 in np.arange(r1 + 0.3, 5.8, 0.05):
                for ga in [0.0, 30.0]:
                    if _time_left() < 60:
                        break
                    side = _objective([r1, r2, ga])
                    if side < best_side and side < 10.0:
                        best_side = side
                        best_params = np.array([r1, r2, ga])

    # Try ring configuration
    if _time_left() > 90:
        ring_guesses = [
            [2.5, 4.5, 0.0],
            [2.5, 4.5, 30.0],
            [2.6, 4.6, 0.0],
            [2.4, 4.4, 0.0],
            [2.5, 4.8, 0.0],
            [2.7, 4.7, 30.0],
        ]
        for guess in ring_guesses:
            if _time_left() < 45:
                break
            try:
                res = minimize(
                    _objective_ring,
                    guess,
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
                )
                if res.fun < best_side and res.fun < 10.0:
                    best_side = res.fun
                    best_params = res.x.copy()
            except Exception:
                continue

    # Fine-tune with Powell from best found
    if best_params is not None and _time_left() > 60:
        try:
            res = minimize(
                _objective,
                best_params,
                method='Powell',
                options={'maxiter': 4000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            if res.fun < best_side:
                best_side = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Final Nelder-Mead polish
    if best_params is not None and _time_left() > 30:
        try:
            res = minimize(
                _objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 8000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            if res.fun < best_side:
                best_side = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

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
                side = 9.0
        
        return inner, outer, side
    
    return best_inner, best_outer, best_side


# EVOLVE-BLOCK-END
