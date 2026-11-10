# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
from scipy.optimize import minimize

_DEADLINE = 620.0  # safety margin before 630s kill


def _hex_vertices(cx, cy, side, angle_deg):
    """Compute 6 vertices of a regular hexagon with center (cx,cy), side length, angle in degrees."""
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3.0
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])


def _outer_vertices(s, ox, oy, angle_deg):
    """Compute 6 vertices of the outer hexagon."""
    return _hex_vertices(ox, oy, s, angle_deg)


def _edge_normal(v1, v2):
    """Unit outward normal of edge from v1 to v2 (assuming CCW polygon)."""
    d = v2 - v1
    # For CCW polygon, outward normal is (dy, -dx) normalized
    n = np.array([d[1], -d[0]])
    norm = np.linalg.norm(n)
    if norm < 1e-15:
        return np.array([1.0, 0.0])
    return n / norm


def _separation_distance(v1, v2, normal):
    """Minimum distance from point v1 to line through v2 along normal direction."""
    return np.dot(v1 - v2, normal)


def _check_overlap(hex1_verts, hex2_verts):
    """
    Separating axis test: returns the minimum violation (negative means overlap).
    We compute for each edge normal of both hexagons the gap.
    Returns max(0, -min_gap) as a penalty (0 means no overlap).
    """
    min_gap = np.inf
    # Edge normals of hex1
    for i in range(6):
        v1 = hex1_verts[i]
        v2 = hex1_verts[(i + 1) % 6]
        n = _edge_normal(v1, v2)
        # Project all vertices of hex2 onto n
        proj2 = np.dot(hex2_verts, n)
        proj1 = np.dot(hex1_verts, n)
        # For separation: max(proj1) < min(proj2) or vice versa
        # We need the gap along this normal
        # The hexagons are separated if the projection intervals don't overlap
        # gap = min(proj2) - max(proj1) if we expect hex1 on the "negative" side
        # But we need to check both directions
        gap1 = np.min(proj2) - np.max(proj1)
        gap2 = np.min(proj1) - np.max(proj2)
        # The actual separation along this axis is max(gap1, gap2) if positive
        # But for the SAT, if ANY axis gives a positive gap, they're separated
        # For penalty, we want the minimum gap over all axes (most negative = most overlapping)
        # Actually: if they overlap, ALL axes give negative or zero gap
        # The minimum gap (most negative) indicates deepest overlap
        min_gap = min(min_gap, max(gap1, gap2))
    
    # Edge normals of hex2
    for i in range(6):
        v1 = hex2_verts[i]
        v2 = hex2_verts[(i + 1) % 6]
        n = _edge_normal(v1, v2)
        proj1 = np.dot(hex1_verts, n)
        proj2 = np.dot(hex2_verts, n)
        gap1 = np.min(proj2) - np.max(proj1)
        gap2 = np.min(proj1) - np.max(proj2)
        min_gap = min(min_gap, max(gap1, gap2))
    
    # If min_gap > 0, they are separated (no overlap)
    # If min_gap < 0, they overlap by -min_gap
    return max(0.0, -min_gap)


def _containment_violation(inner_verts, outer_verts):
    """
    Check if all inner vertices are inside the outer hexagon.
    For each outer edge (p1, p2) in CCW order, cross(p2-p1, v-p1) >= 0 for inside.
    Returns max(0, -min_cross) as penalty.
    """
    min_cross = np.inf
    for i in range(6):
        p1 = outer_verts[i]
        p2 = outer_verts[(i + 1) % 6]
        edge = p2 - p1
        for v in inner_verts:
            cross = edge[0] * (v[1] - p1[1]) - edge[1] * (v[0] - p1[0])
            min_cross = min(min_cross, cross)
    return max(0.0, -min_cross)


def _objective(params):
    """
    params: [x0, y0, a0, x1, y1, a1, ..., x11, y11, a11, s, ox, oy, oa]
    Total: 12*3 + 4 = 40 parameters
    Returns: s + penalty * (overlap + containment)
    """
    inner_params = params[:36].reshape(12, 3)
    s = params[36]
    ox = params[37]
    oy = params[38]
    oa = params[39]
    
    if s < 1e-6:
        return 1e6
    
    # Compute all inner hex vertices
    inner_verts_list = []
    for i in range(12):
        x, y, a = inner_params[i]
        verts = _hex_vertices(x, y, 1.0, a)
        inner_verts_list.append(verts)
    
    # Outer hex vertices
    outer_verts = _outer_vertices(s, ox, oy, oa)
    
    # Containment penalty
    containment_pen = 0.0
    for iv in inner_verts_list:
        containment_pen += _containment_violation(iv, outer_verts)
    
    # Overlap penalty
    overlap_pen = 0.0
    for i in range(12):
        for j in range(i + 1, 12):
            overlap_pen += _check_overlap(inner_verts_list[i], inner_verts_list[j])
    
    penalty = 100.0 * (containment_pen + overlap_pen)
    return s + penalty


def _make_initial_config(seed_idx):
    """Create different initial configurations for multi-start."""
    rng = np.random.RandomState(seed_idx)
    
    # Start with a reasonable hexagonal grid arrangement
    # Hex grid with spacing 2.5 in x, 2.165 in y
    positions = []
    # 4 rows: 3, 4, 3, 2 or similar
    # Let's use a compact arrangement
    # Row 0: 4 hexes
    for i in range(4):
        positions.append([i * 2.5, 0.0])
    # Row 1: 4 hexes offset
    for i in range(4):
        positions.append([i * 2.5 + 1.25, 2.165])
    # Row 2: 4 hexes
    for i in range(4):
        positions.append([i * 2.5, 4.33])
    
    positions = np.array(positions[:12])
    
    # Add some random perturbation
    positions += rng.normal(0, 0.3, size=positions.shape)
    
    angles = np.zeros(12)
    # Small random angles
    angles += rng.normal(0, 10, size=12)
    
    # Outer hex: centered near centroid of inner hexes
    cx = np.mean(positions[:, 0])
    cy = np.mean(positions[:, 1])
    # Estimate needed side length
    max_r = np.max(np.sqrt(positions[:, 0]**2 + positions[:, 1]**2)) + 2.0
    s = max_r + 1.0
    oa = 0.0
    
    return positions, angles, cx, cy, s, oa


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length using gradient-based optimization.
    """
    start_time = time.time()
    
    best_params = None
    best_score = np.inf
    best_result = None
    
    n_starts = 8  # number of multi-start attempts
    
    for seed_idx in range(n_starts):
        if time.time() - start_time > _DEADLINE - 60:
            break
        
        positions, angles, cx, cy, s_init, oa_init = _make_initial_config(seed_idx)
        
        # Build initial parameter vector
        params = np.zeros(40)
        for i in range(12):
            params[3*i] = positions[i, 0]
            params[3*i+1] = positions[i, 1]
            params[3*i+2] = angles[i]
        params[36] = s_init
        params[37] = cx
        params[38] = cy
        params[39] = oa_init
        
        # Bounds
        bounds = [(-20, 20), (-20, 20), (-180, 180)] * 12
        bounds += [(3.0, 20.0), (-10, 10), (-10, 10), (-180, 180)]
        
        # Optimize
        try:
            result = minimize(
                _objective,
                params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
            )
        except Exception:
            continue
        
        if result.fun < best_score:
            best_score = result.fun
            best_params = result.x
        
        # Check deadline
        if time.time() - start_time > _DEADLINE - 30:
            break
    
    # If no optimization succeeded, fall back to seed
    if best_params is None:
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
                [0, -4, 0],
            ]
        )
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8.0
        return inner_hex_data, outer_hex_data, outer_hex_side_length
    
    # Extract best result
    inner_params = best_params[:36].reshape(12, 3)
    s = best_params[36]
    ox = best_params[37]
    oy = best_params[38]
    oa = best_params[39]
    
    # Polish: try a few more iterations with tighter tolerance
    if time.time() - start_time < _DEADLINE - 10:
        try:
            result2 = minimize(
                _objective,
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
            )
            if result2.fun < best_score:
                best_score = result2.fun
                best_params = result2.x
                inner_params = best_params[:36].reshape(12, 3)
                s = best_params[36]
                ox = best_params[37]
                oy = best_params[38]
                oa = best_params[39]
        except Exception:
            pass
    
    inner_hex_data = inner_params.copy()
    outer_hex_data = np.array([ox, oy, oa])
    outer_hex_side_length = float(s)
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
