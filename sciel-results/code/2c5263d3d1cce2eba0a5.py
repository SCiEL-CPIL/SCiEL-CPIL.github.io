# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations


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
    normals = []
    for k in range(6):
        ang = a_rad + np.deg2rad(30 + 60 * k)
        normals.append([np.cos(ang), np.sin(ang)])
    normals = np.array(normals)
    
    verts = _hex_vertices_batch(inner)
    projs = np.abs(np.einsum('ijx,kx->ijk', verts, normals))
    max_proj = np.max(projs)
    
    return max_proj / (np.sqrt(3) / 2)


def _axial_to_pixel(q, r):
    """Convert axial hex coordinates to pixel coordinates (spacing=2)."""
    x = 2.0 * q + r
    y = np.sqrt(3) * r
    return (x, y)


def _generate_grid(max_radius=3):
    """Generate hex grid positions with spacing 2."""
    positions = []
    for q in range(-max_radius, max_radius + 1):
        for r in range(-max_radius, max_radius + 1):
            if abs(q + r) <= max_radius:
                positions.append(_axial_to_pixel(q, r))
    return positions


def _optimize_common_angle(positions, deadline):
    """Optimize the common rotation angle for a set of positions."""
    def obj(params):
        if time.time() > deadline:
            return 1e6
        angle = params[0]
        inner = np.array([[x, y, angle] for x, y in positions])
        if not _hexagons_separated(inner):
            return 1e6
        return _compute_min_side(inner, 0.0)
    
    best_side = float('inf')
    best_angle = 0.0
    
    for start_angle in [0, 15, 30, 45, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]:
        if time.time() > deadline:
            break
        try:
            res = minimize(
                obj,
                [start_angle],
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-6}
            )
            if res.fun < best_side and res.fun < 10.0:
                best_side = res.fun
                best_angle = res.x[0]
        except Exception:
            continue
    
    return best_side, best_angle


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon.
    Returns (inner_hex_data, outer_hex_data, outer_hex_side_length).
    """
    t0 = time.time()
    deadline = t0 + 580.0
    
    best_side = float('inf')
    best_inner = None
    best_outer = None
    
    # Generate hex grid with spacing 2
    grid = _generate_grid(max_radius=3)
    
    # Generate candidate 11-subsets using heuristic: prefer compact clusters
    # Use the fact that we need 11 positions from the grid
    # Sort positions by distance from origin, take combinations of nearby ones
    
    # Phase 1: Try symmetric patterns first
    center = (0.0, 0.0)
    ring1 = [_axial_to_pixel(q, r) for q, r in [(2,0),(1,1),(-1,1),(-2,0),(-1,-1),(1,-1)]]
    ring2 = [_axial_to_pixel(q, r) for q, r in [(4,0),(3,1),(2,2),(0,2),(-2,2),(-3,1),(-4,0),(-3,-1),(-2,-2),(0,-2),(2,-2),(3,-1)]]
    
    candidate_sets = []
    
    # Pattern A: center + 6 ring1 + 4 ring2 (various selections)
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                for l in range(k+1, 12):
                    candidate_sets.append([center] + ring1 + [ring2[i], ring2[j], ring2[k], ring2[l]])
    
    # Pattern B: center + 5 ring1 + 5 ring2
    for omit in range(6):
        r1_sub = [ring1[i] for i in range(6) if i != omit]
        for r2_start in range(12-5+1):
            r2_sub = ring2[r2_start:r2_start+5]
            candidate_sets.append([center] + r1_sub + r2_sub)
    
    # Pattern C: 2 from ring1 + 9 from ring2
    for i in range(6):
        for j in range(i+1, 6):
            for r2_start in range(12-9+1):
                r2_sub = ring2[r2_start:r2_start+9]
                candidate_sets.append([ring1[i], ring1[j]] + r2_sub)
    
    # Remove duplicates
    unique_sets = []
    seen = set()
    for s in candidate_sets:
        key = tuple(sorted((round(x,4), round(y,4)) for x,y in s))
        if key not in seen:
            seen.add(key)
            unique_sets.append(s)
    
    candidate_sets = unique_sets
    
    # Limit to reasonable number
    if len(candidate_sets) > 200:
        # Prioritize sets with smaller bounding box
        def bbox_size(s):
            xs = [p[0] for p in s]
            ys = [p[1] for p in s]
            return (max(xs)-min(xs)) * (max(ys)-min(ys))
        candidate_sets.sort(key=bbox_size)
        candidate_sets = candidate_sets[:200]
    
    # Phase 1: Try each candidate
    for positions in candidate_sets:
        if time.time() > deadline - 180:
            break
        
        side, angle = _optimize_common_angle(positions, deadline - 180)
        if side < best_side and side < 10.0:
            inner = np.array([[x, y, angle] for x, y in positions])
            outer = np.array([0.0, 0.0, 0.0])
            if _valid(inner, outer, side):
                best_side = side
                best_inner = inner
                best_outer = outer
    
    # Phase 2: Free-form polish
    if best_inner is not None and time.time() < deadline - 60:
        def free_obj(p):
            if time.time() > deadline:
                return 1e6
            inner_p = p.reshape(11, 3)
            if not np.all(np.isfinite(inner_p)):
                return 1e6
            if not _hexagons_separated(inner_p):
                return 1e6
            return _compute_min_side(inner_p, 0.0)
        
        try:
            res = minimize(
                free_obj,
                best_inner.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True}
            )
            if res.fun < best_side and res.fun < 10.0:
                inner_opt = res.x.reshape(11, 3)
                outer = np.array([0.0, 0.0, 0.0])
                if _valid(inner_opt, outer, res.fun):
                    best_side = res.fun
                    best_inner = inner_opt
                    best_outer = outer
        except Exception:
            pass
    
    # Phase 3: Multi-start
    if best_inner is not None and time.time() < deadline - 30:
        np.random.seed(42)
        for attempt in range(15):
            if time.time() > deadline - 20:
                break
            perturbed = best_inner + np.random.randn(11, 3) * 0.15
            perturbed[:, 2] = np.degrees(np.arctan2(
                np.sin(np.deg2rad(perturbed[:, 2])), 
                np.cos(np.deg2rad(perturbed[:, 2]))
            ))
            
            def pert_obj(p):
                if time.time() > deadline:
                    return 1e6
                inner_p = p.reshape(11, 3)
                if not np.all(np.isfinite(inner_p)):
                    return 1e6
                if not _hexagons_separated(inner_p):
                    return 1e6
                return _compute_min_side(inner_p, 0.0)
            
            try:
                res = minimize(
                    pert_obj,
                    perturbed.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True}
                )
                if res.fun < best_side and res.fun < 10.0:
                    inner_opt = res.x.reshape(11, 3)
                    outer = np.array([0.0, 0.0, 0.0])
                    if _valid(inner_opt, outer, res.fun):
                        best_side = res.fun
                        best_inner = inner_opt
                        best_outer = outer
            except Exception:
                continue
    
    # Final validation
    if best_inner is not None:
        side = _compute_min_side(best_inner, 0.0)
        outer = np.array([0.0, 0.0, 0.0])
        if _valid(best_inner, outer, side):
            return best_inner, outer, side
        for ds in [0.01, 0.05, 0.1, 0.5, 1.0]:
            if _valid(best_inner, outer, side + ds):
                return best_inner, outer, side + ds
    
    # Fallback
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


# EVOLVE-BLOCK-END
