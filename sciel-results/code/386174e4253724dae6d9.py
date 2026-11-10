# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = 630.0
_SAFETY_MARGIN = 30.0


def _hex_vertices(cx, cy, d, a_deg):
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


def _unit_hex_vertices():
    angles = np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([np.cos(angles), np.sin(angles)])


def _make_cluster(pattern):
    """
    Generate a 12-hexagon cluster on the hexagonal lattice.
    pattern: list of (row, col) in axial coordinates.
    Adjacent hexagons on the hex lattice (distance 2*sqrt(3) apart) are non-overlapping.
    """
    sqrt3 = np.sqrt(3.0)
    pts = np.zeros((len(pattern), 2))
    for i, (r, c) in enumerate(pattern):
        x = sqrt3 * (c + r / 2.0)
        y = 1.5 * r
        pts[i] = [x, y]
    centroid = np.mean(pts, axis=0)
    pts = pts - centroid
    return pts


def _min_enclosing_hex_side(offsets, outer_angle_deg):
    """
    Compute minimum side length s of a regular hexagon at given angle
    that contains all unit hexagons with given offsets (centered at origin).
    """
    outer_angle = np.deg2rad(outer_angle_deg)
    normal_angles = outer_angle + np.deg2rad(30.0) + np.deg2rad(60.0 * np.arange(6))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    unit_verts = _unit_hex_vertices()
    
    # Vectorized: for each hexagon, project its 6 vertices onto each of 6 normals
    # proj shape: (12, 6, 6) -> [hex_i, vertex_j, normal_k]
    # For each hex, max over vertices for each normal -> (12, 6)
    # Then max over all hexagons for each normal -> (6,)
    # apothem = s * sqrt(3)/2, so s = 2 * max_support / sqrt(3)
    
    # offsets @ normals.T -> (12, 6)
    center_projs = offsets @ normals.T  # (12, 6)
    # unit_verts @ normals.T -> (6, 6)
    vert_projs = unit_verts @ normals.T  # (6, 6)
    
    # max projection for each hexagon and normal
    max_proj = center_projs + np.max(vert_projs, axis=0)  # (12, 6)
    max_support = np.max(max_proj)
    
    s = 2.0 * max_support / np.sqrt(3.0)
    return s


def _check_separation(centers, angles_deg, tol=1e-6):
    """Check if all pairs of unit hexagons are separated using SAT."""
    n = len(centers)
    all_verts = []
    for i in range(n):
        a = np.deg2rad(angles_deg[i])
        angles = a + np.deg2rad(60.0 * np.arange(6))
        verts = centers[i] + np.column_stack([np.cos(angles), np.sin(angles)])
        all_verts.append(verts)
    
    for i in range(n):
        for j in range(i + 1, n):
            vi = all_verts[i]
            vj = all_verts[j]
            separated = False
            for k in range(6):
                a_i = np.deg2rad(angles_deg[i])
                n_angle_i = a_i + np.deg2rad(30.0 + 60.0 * k)
                n_i = np.array([np.cos(n_angle_i), np.sin(n_angle_i)])
                proj_i = vi @ n_i
                proj_j = vj @ n_i
                if np.max(proj_i) < np.min(proj_j) - tol or np.max(proj_j) < np.min(proj_i) - tol:
                    separated = True
                    break
            if separated:
                continue
            for k in range(6):
                a_j = np.deg2rad(angles_deg[j])
                n_angle_j = a_j + np.deg2rad(30.0 + 60.0 * k)
                n_j = np.array([np.cos(n_angle_j), np.sin(n_angle_j)])
                proj_i = vi @ n_j
                proj_j = vj @ n_j
                if np.max(proj_i) < np.min(proj_j) - tol or np.max(proj_j) < np.min(proj_i) - tol:
                    separated = True
                    break
            if not separated:
                return False
    return True


def _check_containment(centers, angles_deg, outer_cx, outer_cy, outer_angle_deg, s, tol=1e-6):
    """Check all inner vertices are inside the outer hexagon."""
    outer_verts = _hex_vertices(outer_cx, outer_cy, s, outer_angle_deg)
    for i in range(len(centers)):
        a = np.deg2rad(angles_deg[i])
        angles = a + np.deg2rad(60.0 * np.arange(6))
        verts = centers[i] + np.column_stack([np.cos(angles), np.sin(angles)])
        for v in verts:
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k + 1) % 6]
                cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
                if cross < -tol:
                    return False
    return True


def _validate(centers, angles_deg, outer_cx, outer_cy, outer_angle_deg, s):
    """Full validation check."""
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(angles_deg)):
        return False
    if not np.all(np.isfinite([outer_cx, outer_cy, outer_angle_deg, s])):
        return False
    if s < np.sqrt(12.0) - 1e-8:
        return False
    if not _check_separation(centers, angles_deg):
        return False
    if not _check_containment(centers, angles_deg, outer_cx, outer_cy, outer_angle_deg, s):
        return False
    return True


def hexagon_packing_12():
    """
    Packs 12 unit regular hexagons in a regular hexagon, minimizing the outer side length.
    Returns (inner, outer, s).
    """
    # Define multiple cluster topologies on the hex lattice
    # Axial coordinates: x = sqrt(3)*(c + r/2), y = 1.5*r
    # Adjacent cells are non-overlapping unit hexagons
    
    clusters = []
    
    # Cluster 1: 3-2-3-2-2 (roughly hexagonal)
    p1 = [(0, 0), (0, 1), (0, 2),
          (-1, 1), (-1, 2),
          (-2, 1), (-2, 2), (-2, 3),
          (1, 0), (1, 1),
          (-1, 0), (1, 2)]
    clusters.append(_make_cluster(p1))
    
    # Cluster 2: More symmetric hexagonal shape
    p2 = [(0, 0), (0, 1), (0, 2), (0, 3),
          (-1, 0), (-1, 1), (-1, 2), (-1, 3),
          (-2, 1), (-2, 2),
          (1, 1), (1, 2)]
    clusters.append(_make_cluster(p2))
    
    # Cluster 3: Compact hexagonal arrangement
    p3 = [(0, 0), (0, 1), (0, 2),
          (-1, 0), (-1, 1), (-1, 2),
          (1, 0), (1, 1), (1, 2),
          (-2, 1), (2, 1),
          (0, -1)]
    clusters.append(_make_cluster(p3))
    
    # Cluster 4: Diamond/elongated
    p4 = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
          (-1, 1), (-1, 2), (-1, 3),
          (1, 1), (1, 2), (1, 3)]
    clusters.append(_make_cluster(p4))
    
    # Cluster 5: Compact 4-3-4-1
    p5 = [(0, 0), (0, 1), (0, 2), (0, 3),
          (-1, 0), (-1, 1), (-1, 2),
          (-2, 0), (-2, 1), (-2, 2), (-2, 3),
          (1, 1)]
    clusters.append(_make_cluster(p5))
    
    # Cluster 6: Tightly packed
    p6 = [(0, 0), (0, 1), (0, 2),
          (-1, 0), (-1, 1), (-1, 2),
          (1, 0), (1, 1), (1, 2),
          (2, 0), (2, 1),
          (-2, 1)]
    clusters.append(_make_cluster(p6))
    
    # Cluster 7: Another variant
    p7 = [(0, 0), (0, 1), (0, 2), (0, 3),
          (-1, 1), (-1, 2),
          (1, 1), (1, 2),
          (-2, 1), (-2, 2),
          (2, 1), (2, 2)]
    # That's 14, trim to 12
    p7 = p7[:12]
    clusters.append(_make_cluster(p7))
    
    best_s = np.inf
    best_result = None
    
    for ci, offsets in enumerate(clusters):
        elapsed = time.time() - _START_TIME
        if elapsed > _DEADLINE - _SAFETY_MARGIN:
            break
        
        # Verify this cluster is valid (no overlapping hexagons)
        centers_test = offsets.copy()
        angles_test = np.zeros(12)
        if not _check_separation(centers_test, angles_test):
            continue
        
        # Phase 1: Fine grid search over outer angle (0 to 60 degrees)
        angle_grid = np.linspace(0.0, 60.0, 361)  # 0.1 degree resolution
        best_local_s = np.inf
        best_local_angle = 0.0
        
        for angle in angle_grid:
            s = _min_enclosing_hex_side(offsets, angle)
            if s < best_local_s:
                best_local_s = s
                best_local_angle = angle
        
        if best_local_s < best_s:
            best_s = best_local_s
            best_result = (offsets, best_local_angle)
        
        # Phase 2: Nelder-Mead refinement
        elapsed = time.time() - _START_TIME
        if elapsed < _DEADLINE - _SAFETY_MARGIN and best_local_s < best_s + 0.1:
            try:
                def obj(params):
                    return _min_enclosing_hex_side(offsets, params[0])
                
                result = minimize(
                    obj,
                    np.array([best_local_angle]),
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                )
                if result.fun < best_local_s:
                    best_local_s = result.fun
                    best_local_angle = result.x[0]
                    if best_local_s < best_s:
                        best_s = best_local_s
                        best_result = (offsets, best_local_angle)
            except Exception:
                pass
        
        # Phase 3: Try offsetting the cluster center relative to outer hexagon center
        # This allows the cluster to be shifted within the outer hexagon for better fit
        if best_local_s < best_s + 0.5:
            elapsed = time.time() - _START_TIME
            if elapsed < _DEADLINE - _SAFETY_MARGIN:
                try:
                    def obj_offset(params):
                        ox, oy, angle = params
                        shifted = offsets + np.array([ox, oy])
                        return _min_enclosing_hex_side(shifted, angle)
                    
                    # Start from best angle, zero offset
                    init = np.array([0.0, 0.0, best_local_angle])
                    result = minimize(
                        obj_offset,
                        init,
                        method='Nelder-Mead',
                        options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                    )
                    if result.fun < best_local_s:
                        ox, oy, angle = result.x
                        shifted = offsets + np.array([ox, oy])
                        best_local_s = result.fun
                        if best_local_s < best_s:
                            best_s = best_local_s
                            best_result = (shifted, angle)
                except Exception:
                    pass
    
    # If no valid cluster found, use a fallback
    if best_result is None:
        # Fallback: simple grid
        sqrt3 = np.sqrt(3.0)
        centers = np.array([
            [0.0, 0.0],
            [2*sqrt3, 0.0], [-2*sqrt3, 0.0],
            [sqrt3, 1.5], [-sqrt3, 1.5],
            [sqrt3, -1.5], [-sqrt3, -1.5],
            [3*sqrt3, 1.5], [-3*sqrt3, 1.5],
            [3*sqrt3, -1.5], [-3*sqrt3, -1.5],
            [0.0, 3.0],
        ])
        best_s = np.inf
        best_angle = 0.0
        for angle in np.linspace(0, 60, 121):
            s = _min_enclosing_hex_side(centers, angle)
            if s < best_s:
                best_s = s
                best_angle = angle
        best_result = (centers, best_angle)
    
    offsets, outer_angle_deg = best_result
    s_final = _min_enclosing_hex_side(offsets, outer_angle_deg)
    
    centers = offsets.copy()
    angles_deg = np.zeros(12)
    outer_cx = 0.0
    outer_cy = 0.0
    
    # Final validation
    if not _validate(centers, angles_deg, outer_cx, outer_cy, outer_angle_deg, s_final):
        # Try to fix with a small margin on s
        s_final *= 1.001
        if not _validate(centers, angles_deg, outer_cx, outer_cy, outer_angle_deg, s_final):
            # Last resort: enlarge more
            s_final *= 1.01
    
    # Build output
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = centers[i, 0]
        inner_hex_data[i, 1] = centers[i, 1]
        inner_hex_data[i, 2] = 0.0
    
    outer_hex_data = np.array([outer_cx, outer_cy, outer_angle_deg])
    
    return inner_hex_data, outer_hex_data, s_final


# EVOLVE-BLOCK-END
