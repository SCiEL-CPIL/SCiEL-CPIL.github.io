# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, d, angle_deg):
    """Compute 6 vertices of a regular hexagon."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _is_inside_outer(verts_inner, outer_verts, tol=1e-6):
    """Check if all vertices of inner hex are inside outer hex (CCW polygon)."""
    n = len(outer_verts)
    for i in range(n):
        p1 = outer_verts[i]
        p2 = outer_verts[(i + 1) % n]
        edge = p2 - p1
        for v in verts_inner:
            cross = edge[0] * (v[1] - p1[1]) - edge[1] * (v[0] - p1[0])
            if cross < -tol:
                return False
    return True

def _separation_check(verts1, verts2, tol=1e-6):
    """Separating axis test: return True if separated (non-overlapping)."""
    n = len(verts1)
    for i in range(n):
        p1 = verts1[i]
        p2 = verts1[(i + 1) % n]
        edge = p2 - p1
        normal = np.array([edge[1], -edge[0]])
        norm = np.linalg.norm(normal)
        if norm > 1e-15:
            normal /= norm
            proj1 = verts1 @ normal
            proj2 = verts2 @ normal
            if np.max(proj1) < np.min(proj2) - tol or np.max(proj2) < np.min(proj1) - tol:
                return True
    for i in range(n):
        p1 = verts2[i]
        p2 = verts2[(i + 1) % n]
        edge = p2 - p1
        normal = np.array([edge[1], -edge[0]])
        norm = np.linalg.norm(normal)
        if norm > 1e-15:
            normal /= norm
            proj1 = verts1 @ normal
            proj2 = verts2 @ normal
            if np.max(proj1) < np.min(proj2) - tol or np.max(proj2) < np.min(proj1) - tol:
                return True
    return False

def _validate(inner_data, outer_data, s):
    """Full validation check. Returns True if valid."""
    if len(inner_data) != 12:
        return False
    if not np.all(np.isfinite(inner_data)) or not np.all(np.isfinite(outer_data)):
        return False
    if s < np.sqrt(12) - 1e-9:
        return False
    
    outer_verts = _hex_vertices(outer_data[0], outer_data[1], s, outer_data[2])
    
    for i in range(6):
        for j in range(i+1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-10:
                return False
    
    inner_verts_list = []
    for i in range(12):
        x, y, ang = inner_data[i]
        verts = _hex_vertices(x, y, 1.0, ang)
        inner_verts_list.append(verts)
        for k in range(6):
            v1 = verts[k]
            v2 = verts[(k+1) % 6]
            if abs(np.linalg.norm(v2 - v1) - 1.0) > 1e-8:
                return False
    
    for i in range(12):
        for j in range(i+1, 12):
            if not _separation_check(inner_verts_list[i], inner_verts_list[j]):
                return False
    
    for i in range(12):
        if not _is_inside_outer(inner_verts_list[i], outer_verts):
            return False
    
    return True

def _compute_min_outer(inner_data, outer_center, outer_angle):
    """Find minimum side length s such that all inner hexagons fit inside outer hex."""
    lo = 1.0
    hi = 50.0
    
    for _ in range(20):
        outer_verts = _hex_vertices(outer_center[0], outer_center[1], hi, outer_angle)
        ok = True
        for i in range(len(inner_data)):
            x, y, ang = inner_data[i]
            verts = _hex_vertices(x, y, 1.0, ang)
            if not _is_inside_outer(verts, outer_verts):
                ok = False
                break
        if ok:
            break
        hi *= 2
    
    if not ok:
        return hi
    
    for _ in range(60):
        mid = (lo + hi) / 2
        outer_verts = _hex_vertices(outer_center[0], outer_center[1], mid, outer_angle)
        ok = True
        for i in range(len(inner_data)):
            x, y, ang = inner_data[i]
            verts = _hex_vertices(x, y, 1.0, ang)
            if not _is_inside_outer(verts, outer_verts):
                ok = False
                break
        if ok:
            hi = mid
        else:
            lo = mid
    
    return hi

def _lattice_positions(spacing, rotation_deg):
    """
    Generate 12 positions on a triangular lattice in a hexagonal cluster.
    The 12 points are: center + 6 neighbors + 5 from next ring.
    """
    rot = np.radians(rotation_deg)
    e1 = spacing * np.array([np.cos(rot), np.sin(rot)])
    e2 = spacing * np.array([np.cos(rot + np.radians(60)), np.sin(rot + np.radians(60))])
    
    indices = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1),
        (2, 0), (-2, 0), (0, 2), (0, -2), (2, -2),
    ]
    
    positions = np.zeros((12, 2))
    for idx, (i, j) in enumerate(indices):
        positions[idx] = i * e1 + j * e2
    
    return positions

def _objective_lattice(params):
    """
    params: (spacing, rotation_deg)
    Returns s (to minimize), or large value if invalid.
    """
    spacing, rotation_deg = params
    
    if spacing < 1.8 or spacing > 5.0:
        return 1e10
    
    positions = _lattice_positions(spacing, rotation_deg)
    
    inner_angle = 30.0
    inner_data = np.column_stack([positions, np.full(12, inner_angle)])
    
    outer_center = np.mean(positions, axis=0)
    
    best_s = np.inf
    for outer_angle in [0, 30, 60, 90, 120, 150]:
        s = _compute_min_outer(inner_data, outer_center, outer_angle)
        if s < best_s:
            best_s = s
    
    if not np.isfinite(best_s):
        return 1e10
    
    outer_data = np.array([outer_center[0], outer_center[1], 30.0])
    if not _validate(inner_data, outer_data, best_s):
        return 1e10
    
    return best_s

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon.
    Returns:
        inner_hex_data: (12, 3) array of (x, y, angle_deg)
        outer_hex_data: (3,) array of (x, y, angle_deg)
        outer_hex_side_length: float
    """
    start_time = time.time()
    deadline = start_time + 580
    
    best_s = np.inf
    best_result = None
    
    try:
        starts = [
            (2.0, 0), (2.0, 30), (2.0, 60),
            (2.1, 0), (2.1, 30), (2.1, 60),
            (2.2, 0), (2.2, 30), (2.2, 60),
            (2.5, 0), (2.5, 30), (2.5, 60),
            (3.0, 0), (3.0, 30), (3.0, 60),
        ]
        
        bounds = [(1.8, 5.0), (0, 180)]
        
        for start in starts:
            if time.time() > deadline - 60:
                break
            
            try:
                res = minimize(
                    _objective_lattice,
                    start,
                    method='Nelder-Mead',
                    bounds=bounds,
                    options={'maxiter': 1000, 'xatol': 1e-7, 'fatol': 1e-9}
                )
                
                if res.fun < best_s:
                    best_s = res.fun
                    spacing, rotation_deg = res.x
                    
                    positions = _lattice_positions(spacing, rotation_deg)
                    inner_angle = 30.0
                    inner_data = np.column_stack([positions, np.full(12, inner_angle)])
                    outer_center = np.mean(positions, axis=0)
                    
                    best_outer_angle = 30.0
                    for oa in [0, 30, 60, 90, 120, 150]:
                        s_test = _compute_min_outer(inner_data, outer_center, oa)
                        if s_test < best_s:
                            best_s = s_test
                            best_outer_angle = oa
                    
                    outer_data = np.array([outer_center[0], outer_center[1], best_outer_angle])
                    
                    if _validate(inner_data, outer_data, best_s):
                        best_result = (inner_data, outer_data, best_s)
            except Exception:
                pass
    except Exception:
        pass
    
    if best_result is None:
        spacing = 2.0
        rotation_deg = 30.0
        positions = _lattice_positions(spacing, rotation_deg)
        inner_angle = 30.0
        inner_data = np.column_stack([positions, np.full(12, inner_angle)])
        outer_center = np.mean(positions, axis=0)
        
        best_s = np.inf
        best_outer_angle = 30.0
        for oa in [0, 30, 60, 90, 120, 150]:
            s_test = _compute_min_outer(inner_data, outer_center, oa)
            if s_test < best_s:
                best_s = s_test
                best_outer_angle = oa
        
        outer_data = np.array([outer_center[0], outer_center[1], best_outer_angle])
        
        if _validate(inner_data, outer_data, best_s):
            best_result = (inner_data, outer_data, best_s)
        else:
            spacing = 2.5
            positions = _lattice_positions(spacing, rotation_deg)
            inner_data = np.column_stack([positions, np.full(12, inner_angle)])
            outer_center = np.mean(positions, axis=0)
            
            best_s = np.inf
            best_outer_angle = 30.0
            for oa in [0, 30, 60, 90, 120, 150]:
                s_test = _compute_min_outer(inner_data, outer_center, oa)
                if s_test < best_s:
                    best_s = s_test
                    best_outer_angle = oa
            
            outer_data = np.array([outer_center[0], outer_center[1], best_outer_angle])
            best_result = (inner_data, outer_data, best_s)
    
    return best_result[0], best_result[1], best_result[2]

# EVOLVE-BLOCK-END
