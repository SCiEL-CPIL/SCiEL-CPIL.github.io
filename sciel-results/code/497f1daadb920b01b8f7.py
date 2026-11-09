# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog


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
    Uses the fact that for a hexagon centered at origin with side s and angle a,
    a point p is inside if |p · n_k| <= s * sqrt(3)/2 for all 6 normals n_k.
    So s >= max over all vertices and all k of |p · n_k| / (sqrt(3)/2).
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


def _build_inner(r1, r2, gangle, ring2_angles):
    """Build inner hexagons with given parameters."""
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


def _optimize_configuration(r1_init, r2_init, gangle_init, ring2_angles, deadline):
    """Optimize r1, r2, gangle for a given ring2 angle configuration."""
    from scipy.optimize import minimize
    
    def objective(params):
        r1, r2, gangle = params
        inner = _build_inner(r1, r2, gangle, ring2_angles)
        outer = np.array([0.0, 0.0, gangle])
        side = _compute_min_enclosing_hex(inner, gangle)
        if not _valid(inner, outer, side):
            return side + 1000.0
        return side
    
    try:
        res = minimize(
            objective,
            [r1_init, r2_init, gangle_init],
            method='Powell',
            options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6}
        )
        if res.fun < 100:  # valid solution
            r1, r2, gangle = res.x
            inner = _build_inner(r1, r2, gangle, ring2_angles)
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
    deadline = time.time() + 590.0  # safety margin
    
    best_result = None
    best_side = float('inf')
    
    # Try multiple ring2 angle configurations
    ring2_configs = [
        [30, 150, 210, 330],   # 30 + 60k for k=0,2,3,5
        [0, 120, 180, 300],    # 0 + 60k for k=0,2,3,5
        [60, 180, 240, 0],     # 60 + 60k for k=0,2,3,5
        [90, 210, 270, 30],    # 90 + 60k for k=0,2,3,5
        [150, 270, 330, 90],   # 150 + 60k for k=0,2,3,5
        [30, 90, 150, 210],    # consecutive 60-degree spacing
    ]
    
    # Initial guesses for each configuration
    for ring2_angles in ring2_configs:
        if time.time() > deadline - 30:
            break
        
        initial_guesses = [
            [2.5, 4.5, 0.0],
            [2.5, 4.5, 30.0],
            [2.5, 4.5, 60.0],
            [2.0, 4.0, 0.0],
            [3.0, 5.0, 0.0],
        ]
        
        for guess in initial_guesses:
            if time.time() > deadline - 30:
                break
            
            result = _optimize_configuration(guess[0], guess[1], guess[2], ring2_angles, deadline)
            if result is not None:
                inner, outer, side = result
                if side < best_side:
                    best_side = side
                    best_result = (inner, outer, side)
    
    # Also try a simple grid-based search as fallback
    if best_result is None:
        # Try fixed angle=0 and search r1, r2 on a grid
        for gangle in [0.0, 30.0, 60.0]:
            for r1 in np.arange(2.0, 3.5, 0.1):
                if time.time() > deadline - 10:
                    break
                for r2 in np.arange(3.5, 6.0, 0.1):
                    for ring2_angles in ring2_configs[:3]:
                        inner = _build_inner(r1, r2, gangle, ring2_angles)
                        outer = np.array([0.0, 0.0, gangle])
                        side = _compute_min_enclosing_hex(inner, gangle)
                        if _valid(inner, outer, side) and side < best_side:
                            best_side = side
                            best_result = (inner, outer, side)
    
    if best_result is not None:
        return best_result
    
    # Fallback to the seed arrangement
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


# EVOLVE-BLOCK-END
