# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _hex_vertices(cx, cy, s, angle_deg):
    """Compute 6 vertices of a regular hexagon."""
    angles = np.radians(angle_deg + 60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def _edge_normals(vertices):
    """Compute 6 edge outward normals (unit) for a hexagon given its 6 CCW vertices."""
    n = 6
    normals = np.zeros((n, 2))
    for i in range(n):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % n]
        edge = p2 - p1
        # Outward normal (right-hand side of CCW edge)
        nx, ny = edge[1], -edge[0]
        norm = np.linalg.norm([nx, ny])
        if norm > 1e-12:
            normals[i] = [nx / norm, ny / norm]
    return normals


def _separation_check(v1, n1, v2, n2):
    """
    Check if two hexagons (given vertices and normals) are separated.
    Separation means: on at least one of the 12 edge normals, the projected intervals are apart by > 1e-6.
    """
    all_normals = np.vstack([n1, n2])
    for i in range(12):
        n = all_normals[i]
        proj1 = v1 @ n
        proj2 = v2 @ n
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        if max1 < min2 - 1e-6 or max2 < min1 - 1e-6:
            return True
    return False


def _all_pairs_separated(inner_hexes):
    """Check if all pairs of inner hexagons are separated."""
    n = len(inner_hexes)
    vertices_list = []
    normals_list = []
    for i in range(n):
        x, y, a = inner_hexes[i]
        v = _hex_vertices(x, y, 1.0, a)
        nm = _edge_normals(v)
        vertices_list.append(v)
        normals_list.append(nm)
    
    for i in range(n):
        for j in range(i + 1, n):
            if not _separation_check(vertices_list[i], normals_list[i], vertices_list[j], normals_list[j]):
                return False
    return True


def _outer_contains_inner(outer_center, outer_side, outer_angle, inner_hexes, tol=1e-6):
    """Check if all inner hexagon vertices are inside the outer hexagon."""
    outer_v = _hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    
    for i in range(len(inner_hexes)):
        x, y, a = inner_hexes[i]
        v = _hex_vertices(x, y, 1.0, a)
        for k in range(6):
            for p in v:
                for e in range(6):
                    p1 = outer_v[e]
                    p2 = outer_v[(e + 1) % 6]
                    cross = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
                    if cross < -tol:
                        return False
    return True


def _compute_side_length(inner_hexes, outer_center, outer_angle):
    """
    Compute the minimum side length of outer hexagon (centered at outer_center, angle outer_angle)
    that contains all inner hexagons.
    """
    outer_v_template = _hex_vertices(0, 0, 1.0, outer_angle)
    
    max_s = 0.0
    for e in range(6):
        p1 = outer_v_template[e]
        p2 = outer_v_template[(e + 1) % 6]
        edge = p2 - p1
        nx, ny = edge[1], -edge[0]
        norm = np.linalg.norm([nx, ny])
        if norm < 1e-12:
            continue
        n = np.array([nx / norm, ny / norm])
        
        for i in range(len(inner_hexes)):
            x, y, a = inner_hexes[i]
            v = _hex_vertices(x, y, 1.0, a)
            for p in v:
                dp = p - outer_center
                proj = dp @ n
                if proj > 0:
                    required_s = proj * 2.0 / np.sqrt(3.0)
                    max_s = max(max_s, required_s)
    
    return max_s


def _objective(params, outer_center, outer_angle):
    """
    Objective: minimize the side length of the outer hexagon.
    params: 11*3 = 33 values: for each inner hex, (x, y, angle_deg)
    """
    inner_hexes = params.reshape(11, 3)
    
    if not _all_pairs_separated(inner_hexes):
        return 1e6
    
    side = _compute_side_length(inner_hexes, outer_center, outer_angle)
    
    if not _outer_contains_inner(outer_center, side, outer_angle, inner_hexes):
        return 1e6
    
    return side


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    """
    start_time = time.time()
    deadline = start_time + 600.0
    
    # Initial guess: central hexagon + 6 around + 4 more
    init = np.array([
        [0, 0, 0],
        [2.0, 0, 0],
        [1.0, np.sqrt(3), 0],
        [-1.0, np.sqrt(3), 0],
        [-2.0, 0, 0],
        [-1.0, -np.sqrt(3), 0],
        [1.0, -np.sqrt(3), 0],
        [3.0, np.sqrt(3), 0],
        [0, 2*np.sqrt(3), 0],
        [-3.0, np.sqrt(3), 0],
        [3.0, -np.sqrt(3), 0],
    ])
    
    outer_center = np.array([0.0, 0.0])
    outer_angle = 0.0
    
    best_params = init.copy()
    best_side = _compute_side_length(best_params, outer_center, outer_angle)
    
    # Try optimization with Nelder-Mead
    try:
        result = minimize(
            _objective,
            init.flatten(),
            args=(outer_center, outer_angle),
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        
        if result.success or result.fun < best_side:
            candidate = result.x.reshape(11, 3)
            if _all_pairs_separated(candidate):
                side = _compute_side_length(candidate, outer_center, outer_angle)
                if _outer_contains_inner(outer_center, side, outer_angle, candidate):
                    best_params = candidate
                    best_side = side
    except Exception:
        pass
    
    # Try a few different outer angles
    for angle in [0, 30, 60, 90, 120, 150]:
        if time.time() > deadline - 60:
            break
        outer_angle = float(angle)
        try:
            result = minimize(
                _objective,
                best_params.flatten(),
                args=(outer_center, outer_angle),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            
            if result.fun < best_side:
                candidate = result.x.reshape(11, 3)
                if _all_pairs_separated(candidate):
                    side = _compute_side_length(candidate, outer_center, outer_angle)
                    if _outer_contains_inner(outer_center, side, outer_angle, candidate):
                        best_params = candidate
                        best_side = side
        except Exception:
            continue
    
    # Use the best outer angle found
    best_angle = outer_angle
    final_side = _compute_side_length(best_params, outer_center, best_angle)
    
    # Final verification
    if not _all_pairs_separated(best_params):
        best_params = init
        final_side = _compute_side_length(best_params, outer_center, 0.0)
        best_angle = 0.0
    
    inner_hex_data = best_params
    outer_hex_data = np.array([0.0, 0.0, float(best_angle)])
    outer_hex_side_length = float(final_side)
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
