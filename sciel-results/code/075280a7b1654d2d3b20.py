# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations


def _hex_vertices(cx, cy, s, a_deg):
    """Return the 6 vertices of a regular hexagon with center (cx,cy), side s, rotation a_deg."""
    a = np.radians(a_deg)
    angles = a + np.deg2rad(60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def _hex_edge_normals(verts):
    """Compute outward unit normals for each edge of a hexagon given its 6 vertices."""
    norms = []
    for k in range(6):
        p1 = verts[k]
        p2 = verts[(k + 1) % 6]
        e = p2 - p1
        n = np.array([e[1], -e[0]])
        norm = np.linalg.norm(n)
        if norm > 1e-15:
            n = n / norm
        else:
            n = np.array([1.0, 0.0])
        norms.append(n)
    return np.array(norms)


def _min_gap_pair(v1, v2):
    """
    Compute the minimum gap between two hexagons over all 12 edge normals.
    Returns the minimum gap (positive = separated, negative = overlapping).
    """
    n1 = _hex_edge_normals(v1)
    n2 = _hex_edge_normals(v2)
    all_normals = np.vstack([n1, n2])
    
    min_gap = float('inf')
    for n in all_normals:
        proj1 = v1 @ n
        proj2 = v2 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        gap = max(max1 - min2, max2 - min1)
        if gap < min_gap:
            min_gap = gap
    return min_gap


def _min_enclosing_hex_side(inner_hex_data, outer_angle_deg=0.0):
    """
    Compute minimum side of regular hexagon (centered at origin, given angle) containing all inner hexes.
    Returns side length (float).
    """
    outer_angle = np.radians(outer_angle_deg)
    normals = []
    for k in range(6):
        ang = outer_angle + np.radians(30 + 60 * k)
        normals.append(np.array([np.cos(ang), np.sin(ang)]))
    normals = np.array(normals)
    
    max_proj = 0.0
    for i in range(len(inner_hex_data)):
        x, y, a_deg = inner_hex_data[i]
        verts = _hex_vertices(x, y, 1.0, a_deg)
        projs = verts @ normals.T
        m = np.max(projs)
        if m > max_proj:
            max_proj = m
    
    return max_proj * 2.0 / np.sqrt(3.0)


def _check_validity(inner_hex_data, outer_hex_data, side):
    """Check all validity conditions. Returns True if valid."""
    if not (np.all(np.isfinite(inner_hex_data)) and np.all(np.isfinite(outer_hex_data))):
        return False
    if not np.isfinite(side):
        return False
    if side < np.sqrt(11) - 1e-9:
        return False
    
    ox, oy, oa = outer_hex_data
    outer_verts = _hex_vertices(ox, oy, side, oa)
    
    # Check every inner vertex is inside outer hexagon
    for i in range(len(inner_hex_data)):
        x, y, a_deg = inner_hex_data[i]
        verts = _hex_vertices(x, y, 1.0, a_deg)
        for p in verts:
            for j in range(6):
                p1 = outer_verts[j]
                p2 = outer_verts[(j + 1) % 6]
                e = p2 - p1
                v = p - p1
                cross = e[0] * v[1] - e[1] * v[0]
                if cross < -1e-6:
                    return False
    
    # Check every pair of inner hexagons is separated
    n_inner = len(inner_hex_data)
    verts_list = []
    for i in range(n_inner):
        x, y, a_deg = inner_hex_data[i]
        verts_list.append(_hex_vertices(x, y, 1.0, a_deg))
    
    for i in range(n_inner):
        for j in range(i + 1, n_inner):
            min_gap = _min_gap_pair(verts_list[i], verts_list[j])
            if min_gap <= 1e-6:
                return False
    
    return True


def _objective(params, outer_angle_deg=0.0, beta=50.0):
    """
    Smooth objective: log-sum-exp of enclosing-side constraint + penalty for overlaps.
    """
    inner = params.reshape(11, 3)
    
    # Compute enclosing side
    side = _min_enclosing_hex_side(inner, outer_angle_deg)
    
    # Compute overlap penalties (smooth)
    total_overlap = 0.0
    n = len(inner)
    verts_list = []
    for i in range(n):
        x, y, a_deg = inner[i]
        verts_list.append(_hex_vertices(x, y, 1.0, a_deg))
    
    for i in range(n):
        for j in range(i + 1, n):
            min_gap = _min_gap_pair(verts_list[i], verts_list[j])
            # Penalize if gap < 1e-6 (overlap or touching)
            if min_gap < 1e-6:
                total_overlap += (1e-6 - min_gap)
    
    return side + 100.0 * total_overlap


def _objective_exact(params, outer_angle_deg=0.0):
    """Exact objective for final polish."""
    return _objective(params, outer_angle_deg, beta=1000.0)


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    
    Strategy: 
    1. Start from multiple diverse initial configurations
    2. Use smooth log-sum-exp surrogate for coarse optimization
    3. Polish with exact objective
    4. Optimize outer hexagon angle
    """
    start_time = time.time()
    deadline = start_time + 600.0  # Safety margin
    
    def time_left():
        return deadline - time.time()
    
    # Initial configurations
    configs = []
    
    # Config 1: Honeycomb-like arrangement
    d = np.sqrt(3)
    inner_seed1 = np.array([
        [0, 0, 0],
        [d, 0, 0],
        [-d, 0, 0],
        [d/2, d*np.sqrt(3)/2, 0],
        [-d/2, d*np.sqrt(3)/2, 0],
        [d/2, -d*np.sqrt(3)/2, 0],
        [-d/2, -d*np.sqrt(3)/2, 0],
        [1.5*d, d*np.sqrt(3)/2, 0],
        [-1.5*d, d*np.sqrt(3)/2, 0],
        [1.5*d, -d*np.sqrt(3)/2, 0],
        [-1.5*d, -d*np.sqrt(3)/2, 0],
    ])
    configs.append(inner_seed1)
    
    # Config 2: Tighter arrangement
    inner_seed2 = np.array([
        [0, 0, 0],
        [2.5, 0, 0],
        [-2.5, 0, 0],
        [1.25, 2.165, 0],
        [-1.25, 2.165, 0],
        [1.25, -2.165, 0],
        [-1.25, -2.165, 0],
        [3.75, 2.165, 0],
        [-3.75, 2.165, 0],
        [3.75, -2.165, 0],
        [-3.75, -2.165, 0],
    ])
    configs.append(inner_seed2)
    
    # Config 3: Rotated honeycomb
    angle_rot = np.radians(30)
    inner_seed3 = inner_seed1.copy()
    for i in range(11):
        x, y = inner_seed3[i, 0], inner_seed3[i, 1]
        inner_seed3[i, 0] = x * np.cos(angle_rot) - y * np.sin(angle_rot)
        inner_seed3[i, 1] = x * np.sin(angle_rot) + y * np.cos(angle_rot)
    configs.append(inner_seed3)
    
    best_inner = None
    best_side = float('inf')
    best_outer = np.array([0.0, 0.0, 0.0])
    
    # Phase 1: Coarse optimization with smooth objective
    for cfg_idx, cfg in enumerate(configs):
        if time_left() < 60:
            break
        
        x0 = cfg.flatten()
        
        try:
            # Coarse optimization with smooth objective
            res = minimize(
                lambda p: _objective(p, 0.0, beta=50.0),
                x0,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
            )
            
            candidate = res.x.reshape(11, 3)
            candidate[:, 2] = candidate[:, 2] % 360.0
            
            # Check validity
            side = _min_enclosing_hex_side(candidate, 0.0)
            if _check_validity(candidate, np.array([0.0, 0.0, 0.0]), side):
                if side < best_side:
                    best_side = side
                    best_inner = candidate.copy()
        except Exception:
            pass
    
    # Phase 2: Polish with exact objective
    if best_inner is not None and time_left() > 30:
        x0 = best_inner.flatten()
        try:
            res = minimize(
                lambda p: _objective_exact(p, 0.0),
                x0,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
            )
            
            candidate = res.x.reshape(11, 3)
            candidate[:, 2] = candidate[:, 2] % 360.0
            
            side = _min_enclosing_hex_side(candidate, 0.0)
            if _check_validity(candidate, np.array([0.0, 0.0, 0.0]), side):
                if side < best_side:
                    best_side = side
                    best_inner = candidate.copy()
        except Exception:
            pass
    
    # Phase 3: Optimize outer hexagon angle
    if best_inner is not None and time_left() > 15:
        def obj_outer_angle(outer_ang):
            return _min_enclosing_hex_side(best_inner, outer_ang)
        
        # Grid search
        best_outer_ang = 0.0
        best_side_outer = best_side
        
        for outer_ang in np.arange(0, 60, 2.0):
            if time_left() < 15:
                break
            side = obj_outer_angle(outer_ang)
            if side < best_side_outer:
                best_side_outer = side
                best_outer_ang = outer_ang
        
        # Refine
        if time_left() > 10:
            try:
                res = minimize(obj_outer_angle, best_outer_ang, method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
                if res.fun < best_side_outer:
                    best_outer_ang = res.x[0]
                    best_side_outer = res.fun
            except Exception:
                pass
        
        best_side = best_side_outer
        best_outer = np.array([0.0, 0.0, best_outer_ang])
    
    # Final validity check
    if best_inner is None or not _check_validity(best_inner, best_outer, best_side):
        # Fallback to seed
        best_inner = inner_seed2.copy()
        best_outer = np.array([0.0, 0.0, 0.0])
        best_side = _min_enclosing_hex_side(best_inner, 0.0)
    
    return best_inner, best_outer, best_side


# EVOLVE-BLOCK-END
