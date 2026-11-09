# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution


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
    
    max_proj = 0.0
    for i in range(len(inner)):
        x, y, a = inner[i]
        v = _hex_vertices(x, y, 1.0, a)
        projs = np.abs(v @ normals.T)
        max_proj = max(max_proj, np.max(projs))
    
    return max_proj / (np.sqrt(3) / 2)


def _build_inner_fixed_core():
    """
    Builds the 7 core hexagons: 1 center + 6 ring.
    The ring hexagons touch the center and each other.
    Distance from center to ring centers is sqrt(3).
    """
    core = [[0.0, 0.0, 0.0]]
    r = np.sqrt(3)
    for k in range(6):
        ang = np.deg2rad(60 * k)
        x = r * np.cos(ang)
        y = r * np.sin(ang)
        core.append([x, y, 0.0])
    return np.array(core)


def _objective_asymmetric(x_vec):
    """
    Objective function for optimizing the positions of the 4 outer hexagons.
    x_vec: (8,) array of [x1, y1, x2, y2, x3, y3, x4, y4]
    Returns the side length of the enclosing hexagon.
    """
    core = _build_inner_fixed_core()
    
    # Construct the 4 outer hexagons
    outer_hexes = []
    for i in range(4):
        x = x_vec[2*i]
        y = x_vec[2*i+1]
        outer_hexes.append([x, y, 0.0])
    
    inner = np.vstack([core, np.array(outer_hexes)])
    
    # Check separation quickly to penalize invalid configurations
    # We can do a fast check of minimum distance between centers
    # If center distance < 2 * apothem - eps, they overlap.
    # Apothem = sqrt(3)/2. Min dist = sqrt(3).
    # But we need to check all pairs.
    # For speed in DE, we can just return a large penalty if obvious overlap.
    # However, DE calls the objective many times. 
    # Let's do a fast separation check.
    for i in range(11):
        for j in range(i+1, 11):
            dx = inner[i,0] - inner[j,0]
            dy = inner[i,1] - inner[j,1]
            dist = np.sqrt(dx*dx + dy*dy)
            # If dist < sqrt(3) - 1e-5, they definitely overlap (for aligned hexes)
            # Actually, min distance for touching is sqrt(3).
            if dist < np.sqrt(3) - 1e-4:
                return 100.0 + dist # Penalty
    
    side = _compute_min_side(inner, 0.0)
    
    # Check validity for the best candidates? 
    # DE doesn't know about validity, it just minimizes.
    # We should ensure the result is valid at the end.
    # To guide DE, we can add a penalty if not valid, but _valid is expensive.
    # Let's assume the separation check above is a good proxy.
    
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
    
    # 1. Try the symmetric solution from the parent as a baseline
    # Re-implementing the quick symmetric optimization to get a good starting point
    def _build_inner_symmetric(r1, r2, gangle):
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
        
    # Quick Nelder-Mead for symmetric case to get a baseline
    def _objective_sym(params):
        r1, r2, gangle = params
        inner = _build_inner_symmetric(r1, r2, gangle)
        side = _compute_min_side(inner, gangle)
        if not _hexagons_separated(inner):
            return side + 50.0
        return side

    try:
        res_sym = minimize(
            _objective_sym,
            [np.sqrt(3), 3.5, 0.0],
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6}
        )
        if res_sym.fun < 10.0:
            r1, r2, gangle = res_sym.x
            inner_sym = _build_inner_symmetric(r1, r2, gangle)
            outer_sym = np.array([0.0, 0.0, gangle])
            side_sym = _compute_min_side(inner_sym, gangle)
            if _valid(inner_sym, outer_sym, side_sym):
                best_side = side_sym
                best_inner = inner_sym
                best_outer = outer_sym
    except Exception:
        pass

    # 2. Try asymmetric optimization for the 4 outer hexes
    if time.time() < deadline - 100:
        # Use Differential Evolution
        # Bounds for x, y of the 4 outer hexes
        # They should be roughly in the corners of the hexagon.
        # Max radius approx 4-5.
        bounds = [(-5, 5), (-5, 5)] * 4
        
        # Initial population guess: use the symmetric positions
        core = _build_inner_fixed_core()
        # Symmetric positions for the 4 outer hexes (from parent logic)
        r2_init = 3.5
        angles_init = [np.deg2rad(30), np.deg2rad(150), np.deg2rad(210), np.deg2rad(330)]
        x0 = []
        for ang in angles_init:
            x0.append(r2_init * np.cos(ang))
            x0.append(r2_init * np.sin(ang))
            
        try:
            # DE can be slow, so we limit popsize and maxiter
            # We will run DE, then refine the best result with Nelder-Mead
            de_res = differential_evolution(
                _objective_asymmetric,
                bounds,
                seed=42,
                maxiter=200,
                popsize=15,
                tol=1e-4,
                update_interval=10,
                polish=False # We will polish manually
            )
            
            if de_res.fun < best_side:
                # Polish the result from DE using Nelder-Mead
                def _objective_refine(x_vec):
                    core = _build_inner_fixed_core()
                    outer_hexes = []
                    for i in range(4):
                        x = x_vec[2*i]
                        y = x_vec[2*i+1]
                        outer_hexes.append([x, y, 0.0])
                    inner = np.vstack([core, np.array(outer_hexes)])
                    if not _hexagons_separated(inner):
                        return 100.0
                    side = _compute_min_side(inner, 0.0)
                    return side

                res_ref = minimize(
                    _objective_refine,
                    de_res.x,
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                
                if res_ref.fun < best_side:
                    # Construct final inner
                    core = _build_inner_fixed_core()
                    outer_hexes = []
                    for i in range(4):
                        x = res_ref.x[2*i]
                        y = res_ref.x[2*i+1]
                        outer_hexes.append([x, y, 0.0])
                    inner_asym = np.vstack([core, np.array(outer_hexes)])
                    outer_asym = np.array([0.0, 0.0, 0.0])
                    side_asym = _compute_min_side(inner_asym, 0.0)
                    
                    if _valid(inner_asym, outer_asym, side_asym):
                        best_side = side_asym
                        best_inner = inner_asym
                        best_outer = outer_asym
                        
        except Exception:
            pass

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
