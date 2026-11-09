# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _hex_vertices(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    ang = a + np.deg2rad(60 * np.arange(6))
    return np.stack([cx + s * np.cos(ang), cy + s * np.sin(ang)], axis=1)


def _hex_normals(a_deg):
    a = np.deg2rad(a_deg)
    base = a + np.deg2rad(60 * np.arange(6))
    normals = base + np.deg2rad(30)
    dirs = np.stack([np.cos(normals), np.sin(normals)], axis=1)
    return np.vstack([dirs, -dirs])


def _separation_ok(v1, n1, v2, n2):
    """Check if two hexagons are separated on at least one normal."""
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
    """Check all pairs of inner hexagons are separated."""
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
    """Check all inner hex vertices are inside or on the outer hexagon."""
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
    if side < np.sqrt(11):
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


def _compute_min_enclosing_hex(inner_data, angle_deg):
    """
    Compute the minimum side length of a regular hexagon with given angle
    that contains all vertices of the inner hexagons, centered at origin.
    """
    a_rad = np.deg2rad(angle_deg)
    normals = []
    for k in range(6):
        ang = a_rad + np.deg2rad(30 + 60 * k)
        normals.append([np.cos(ang), np.sin(ang)])
    normals = np.array(normals)
    
    max_proj = 0.0
    for i in range(len(inner_data)):
        x, y, a = inner_data[i]
        v = _hex_vertices(x, y, 1.0, a)
        for p in v:
            projs = np.abs(p @ normals.T)
            max_proj = max(max_proj, np.max(projs))
    
    side = max_proj / (np.sqrt(3) / 2)
    return side


def _build_inner_hexagonal(r1, r2, gangle, ring2_angles):
    """Build inner hexagons in a hexagonal lattice arrangement."""
    inner = []
    # Center
    inner.append([0.0, 0.0, gangle])
    # First ring: 6 hexagons
    for k in range(6):
        ang = np.deg2rad(60 * k)
        x = r1 * np.cos(ang)
        y = r1 * np.sin(ang)
        inner.append([x, y, gangle])
    # Second ring: 4 hexagons
    for ang_deg in ring2_angles:
        ang = np.deg2rad(ang_deg)
        x = r2 * np.cos(ang)
        y = r2 * np.sin(ang)
        inner.append([x, y, gangle])
    return np.array(inner)


def _build_inner_grid(r1, r2, gangle, ring2_angles):
    """Build inner hexagons in a grid-like arrangement."""
    inner = []
    # Center
    inner.append([0.0, 0.0, gangle])
    # First ring: 6 hexagons at 60-degree intervals
    for k in range(6):
        ang = np.deg2rad(60 * k)
        x = r1 * np.cos(ang)
        y = r1 * np.sin(ang)
        inner.append([x, y, gangle])
    # Second ring: 4 hexagons at specified angles
    for ang_deg in ring2_angles:
        ang = np.deg2rad(ang_deg)
        x = r2 * np.cos(ang)
        y = r2 * np.sin(ang)
        inner.append([x, y, gangle])
    return np.array(inner)


def _optimize_configuration(r1_init, r2_init, gangle_init, ring2_angles, deadline, build_func):
    """Optimize r1, r2, gangle for a given ring2 angle configuration."""
    
    def objective(params):
        r1, r2, gangle = params
        if r1 < 0.1 or r2 < 0.5 or r1 > 20 or r2 > 20:
            return 1000.0
        inner = build_func(r1, r2, gangle, ring2_angles)
        outer = np.array([0.0, 0.0, gangle])
        side = _compute_min_enclosing_hex(inner, gangle)
        if not _valid(inner, outer, side):
            return side + 500.0
        return side
    
    try:
        res = minimize(
            objective,
            [r1_init, r2_init, gangle_init],
            method='Powell',
            options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-9}
        )
        if res.fun < 200:  # valid solution
            r1, r2, gangle = res.x
            inner = build_func(r1, r2, gangle, ring2_angles)
            outer = np.array([0.0, 0.0, gangle])
            side = _compute_min_enclosing_hex(inner, gangle)
            if _valid(inner, outer, side):
                return inner, outer, side
    except Exception:
        pass
    
    # Try Nelder-Mead as a second approach
    try:
        res2 = minimize(
            objective,
            [r1_init, r2_init, gangle_init],
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-9}
        )
        if res2.fun < 200:
            r1, r2, gangle = res2.x
            inner = build_func(r1, r2, gangle, ring2_angles)
            outer = np.array([0.0, 0.0, gangle])
            side = _compute_min_enclosing_hex(inner, gangle)
            if _valid(inner, outer, side):
                return inner, outer, side
    except Exception:
        pass
    
    return None


def _optimize_free_positions(positions_init, gangle, deadline):
    """
    Optimize free positions of 11 hexagons (all with same orientation gangle).
    positions_init: (11, 2) array of (x, y) for each hexagon.
    """
    def objective(params):
        pos = params[:22].reshape(11, 2)
        inner = np.column_stack([pos, np.full(11, gangle)])
        outer = np.array([0.0, 0.0, gangle])
        side = _compute_min_enclosing_hex(inner, gangle)
        if not _valid(inner, outer, side):
            return side + 500.0
        return side
    
    try:
        res = minimize(
            objective,
            positions_init.flatten(),
            method='Powell',
            options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-9}
        )
        if res.fun < 200:
            pos = res.x[:22].reshape(11, 2)
            inner = np.column_stack([pos, np.full(11, gangle)])
            outer = np.array([0.0, 0.0, gangle])
            side = _compute_min_enclosing_hex(inner, gangle)
            if _valid(inner, outer, side):
                return inner, outer, side
    except Exception:
        pass
    
    return None


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon.
    Returns (inner, outer, side) where inner is (11,3), outer is (3,), side is float.
    """
    deadline = time.time() + 600.0  # safety margin before 630s kill
    
    best_result = None
    best_side = float('inf')
    
    # Try multiple ring2 angle configurations
    ring2_configs = [
        [30, 150, 210, 330],
        [0, 120, 180, 300],
        [60, 180, 240, 300],
        [90, 210, 270, 30],
        [150, 270, 330, 90],
        [30, 90, 150, 210],
        [60, 120, 240, 300],
        [0, 60, 180, 240],
    ]
    
    # Initial guesses for each configuration
    for ring2_angles in ring2_configs:
        if time.time() > deadline - 60:
            break
        
        initial_guesses = [
            [2.5, 4.5, 0.0],
            [2.5, 4.5, 30.0],
            [2.5, 4.5, 60.0],
            [2.0, 4.0, 0.0],
            [3.0, 5.0, 0.0],
            [2.2, 4.2, 15.0],
            [2.8, 4.8, 45.0],
            [2.5, 5.0, 0.0],
            [2.5, 5.0, 30.0],
        ]
        
        for guess in initial_guesses:
            if time.time() > deadline - 60:
                break
            
            result = _optimize_configuration(guess[0], guess[1], guess[2], ring2_angles, deadline, _build_inner_hexagonal)
            if result is not None:
                inner, outer, side = result
                if side < best_side:
                    best_side = side
                    best_result = (inner, outer, side)
    
    # Try free position optimization from the best structured result
    if best_result is not None and time.time() < deadline - 30:
        inner, outer, side = best_result
        gangle = inner[0, 2]
        pos_init = inner[:, :2].copy()
        
        result = _optimize_free_positions(pos_init, gangle, deadline)
        if result is not None:
            inner2, outer2, side2 = result
            if side2 < best_side:
                best_side = side2
                best_result = (inner2, outer2, side2)
    
    # If we found a good result, do a final high-precision polish pass
    if best_result is not None and time.time() < deadline - 20:
        inner, outer, side = best_result
        gangle = inner[0, 2]
        pos_init = inner[:, :2].copy()
        
        # Polish with tighter tolerance
        def polish_objective(params):
            pos = params[:22].reshape(11, 2)
            inner_p = np.column_stack([pos, np.full(11, gangle)])
            outer_p = np.array([0.0, 0.0, gangle])
            side_p = _compute_min_enclosing_hex(inner_p, gangle)
            if not _valid(inner_p, outer_p, side_p):
                return side_p + 500.0
            return side_p
        
        try:
            res_polish = minimize(
                polish_objective,
                pos_init.flatten(),
                method='Powell',
                options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            if res_polish.fun < 200:
                pos = res_polish.x[:22].reshape(11, 2)
                inner_p = np.column_stack([pos, np.full(11, gangle)])
                outer_p = np.array([0.0, 0.0, gangle])
                side_p = _compute_min_enclosing_hex(inner_p, gangle)
                if _valid(inner_p, outer_p, side_p) and side_p < best_side:
                    best_side = side_p
                    best_result = (inner_p, outer_p, side_p)
        except Exception:
            pass
    
    # Fallback to a known valid arrangement if nothing else works
    if best_result is None:
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
        outer_hex_side_length = 8.0
        return inner_hex_data, outer_hex_data, outer_hex_side_length
    
    return best_result


# EVOLVE-BLOCK-END
