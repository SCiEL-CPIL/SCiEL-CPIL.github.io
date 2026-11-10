# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

def _hex_vertices(cx, cy, d, angle_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side d, angle in degrees."""
    a = math.radians(angle_deg)
    verts = np.empty((6, 2))
    for i in range(6):
        theta = a + math.radians(60 * i)
        verts[i, 0] = cx + d * math.cos(theta)
        verts[i, 1] = cy + d * math.sin(theta)
    return verts

def _check_separation(hex1, hex2):
    """Check if two hexagons (each (6,2) arrays) are separated by SAT."""
    for verts in (hex1, hex2):
        for i in range(6):
            p1 = verts[i]
            p2 = verts[(i + 1) % 6]
            edge = p2 - p1
            n = np.array([-edge[1], edge[0]])
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-15:
                continue
            n = n / n_norm
            proj1 = hex1 @ n
            proj2 = hex2 @ n
            if proj1.max() <= proj2.min() + 1e-6 or proj2.max() <= proj1.min() + 1e-6:
                return True
    return False

def _check_inside(hex_verts, outer_verts):
    """Check if all vertices of hex are inside outer hexagon."""
    for v in hex_verts:
        for i in range(6):
            p1 = outer_verts[i]
            p2 = outer_verts[(i + 1) % 6]
            cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
            if cross < -1e-6:
                return False
    return True

def _is_valid(inner_data, outer_data, s):
    """Full validity check."""
    if not all(np.isfinite(inner_data).all()):
        return False
    if not all(np.isfinite(outer_data)):
        return False
    if s < math.sqrt(12) - 1e-8:
        return False
    
    outer_verts = _hex_vertices(outer_data[0], outer_data[1], s, outer_data[2])
    for i in range(6):
        for j in range(i + 1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-10:
                return False
    
    inner_verts_list = []
    for i in range(12):
        v = _hex_vertices(inner_data[i, 0], inner_data[i, 1], 1.0, inner_data[i, 2])
        inner_verts_list.append(v)
        if not _check_inside(v, outer_verts):
            return False
    
    for i in range(12):
        for j in range(i + 1, 12):
            if not _check_separation(inner_verts_list[i], inner_verts_list[j]):
                return False
    return True

def _get_all_inner_vertices(inner_data):
    """Get all 72 vertices from all 12 inner hexagons."""
    all_verts = []
    for i in range(12):
        v = _hex_vertices(inner_data[i, 0], inner_data[i, 1], 1.0, inner_data[i, 2])
        all_verts.append(v)
    return all_verts

def _min_s_for_all_vertices(all_inner_verts, outer_center, angle_deg):
    """Find minimum s such that all inner vertices fit in outer hexagon."""
    a = math.radians(angle_deg)
    normals = []
    for i in range(6):
        theta = a + math.radians(30 + 60 * i)
        normals.append(np.array([math.cos(theta), math.sin(theta)]))
    
    max_proj = 0
    for verts in all_inner_verts:
        for v in verts:
            rel = v - outer_center
            for n in normals:
                proj = abs(rel @ n)
                if proj > max_proj:
                    max_proj = proj
    return max_proj / (math.sqrt(3) / 2)

def _objective(params, outer_angle_deg):
    """
    Objective function for optimization: minimize s.
    params: (24,) array of 12 (x, y) pairs.
    Returns s (the outer hexagon side length needed).
    """
    inner = np.zeros((12, 3))
    for i in range(12):
        inner[i, 0] = params[2*i]
        inner[i, 1] = params[2*i+1]
        inner[i, 2] = 0.0  # fixed angle for inner hexagons
    
    all_verts = _get_all_inner_vertices(inner)
    s = _min_s_for_all_vertices(all_verts, np.array([0, 0]), outer_angle_deg)
    
    # Add penalty for overlaps
    penalty = 0.0
    inner_verts_list = all_verts
    for i in range(12):
        for j in range(i + 1, 12):
            if not _check_separation(inner_verts_list[i], inner_verts_list[j]):
                # Overlapping - add penalty
                c1 = np.array([inner[i, 0], inner[i, 1]])
                c2 = np.array([inner[j, 0], inner[j, 1]])
                dist = np.linalg.norm(c1 - c2)
                min_dist = 2.0 * math.sqrt(3)  # minimum center distance for non-overlap
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2
    
    return s + penalty

def _hexagonal_lattice_positions():
    """
    Generate a compact hexagonal lattice arrangement of 12 hexagons.
    Using a honeycomb-like pattern for tight packing.
    """
    # For hexagons with side 1, the distance between adjacent centers is 2*sqrt(3)/2 = sqrt(3)
    # for edge-to-edge contact in a honeycomb pattern.
    # Actually, for regular hexagons of side 1, the distance between centers of adjacent
    # hexagons sharing an edge is 2 * apothem = 2 * (sqrt(3)/2) = sqrt(3).
    
    # Let's use a more compact arrangement:
    # Row 0: 3 hexagons
    # Row 1: 4 hexagons (offset)
    # Row 2: 3 hexagons
    # Row 3: 2 hexagons (offset)
    # Total: 12
    
    positions = []
    # Horizontal spacing between centers in same row: 3.0 (for flat-top hexagons)
    # Vertical spacing between rows: 3*sqrt(3)/2
    dx = 3.0
    dy = 3 * math.sqrt(3) / 2
    dx_offset = 1.5
    
    # Row 0: 3 hexagons at y=0
    for col in range(3):
        positions.append([col * dx - dx, 0, 0.0])  # center them
    
    # Row 1: 4 hexagons at y=dy, offset by dx/2
    for col in range(4):
        positions.append([col * dx - 1.5 * dx + dx/2, dy, 0.0])
    
    # Row 2: 3 hexagons at y=2*dy
    for col in range(3):
        positions.append([col * dx - dx, 2 * dy, 0.0])
    
    # Row 3: 2 hexagons at y=3*dy, offset
    for col in range(2):
        positions.append([col * dx + 0.5 * dx - dx + dx/2, 3 * dy, 0.0])
    
    return np.array(positions)

def _compact_arrangement_1():
    """
    Compact arrangement: honeycomb-like with 12 hexagons.
    Using pointy-top orientation for tighter packing.
    """
    # For pointy-top hexagons of side 1:
    # Width = sqrt(3), Height = 2
    # Horizontal spacing: sqrt(3)
    # Vertical spacing: 1.5
    # Offset for alternating rows: sqrt(3)/2
    
    positions = []
    hw = math.sqrt(3)  # half-width
    vs = 1.5  # vertical spacing
    
    # Arrangement: 4-3-3-2 or similar compact pattern
    # Let's try a 3-4-3-2 pattern
    rows = [
        (3, 0),    # 3 hexagons, no offset
        (4, 1),    # 4 hexagons, offset
        (3, 0),    # 3 hexagons, no offset
        (2, 1),    # 2 hexagons, offset
    ]
    
    y = 0
    for count, offset in rows:
        for col in range(count):
            x = col * hw
            if offset == 1:
                x += hw / 2
            positions.append([x, y, 0.0])
        y += vs
    
    return np.array(positions)

def _compact_arrangement_2():
    """
    Another compact arrangement using a different pattern.
    """
    positions = []
    # Try a more hexagonal cluster
    # Center + 6 around + 5 more
    
    # Center
    positions.append([0, 0, 0.0])
    
    # 6 around the center (hexagonal neighbors)
    for i in range(6):
        angle = math.radians(60 * i)
        r = 2 * math.sqrt(3)  # distance to adjacent center
        positions.append([r * math.cos(angle), r * math.sin(angle), 0.0])
    
    # 5 more in the second ring
    for i in range(5):
        angle = math.radians(60 * i + 30)
        r = 4 * math.sqrt(3)  # distance to second ring
        positions.append([r * math.cos(angle), r * math.sin(angle), 0.0])
    
    return np.array(positions)

def _optimize_from_start(start_params, outer_angle_deg, deadline):
    """
    Optimize from a starting position using scipy.optimize.
    """
    start_params = np.array(start_params, dtype=float)
    
    # Use L-BFGS-B for optimization
    result = minimize(
        _objective,
        start_params,
        args=(outer_angle_deg,),
        method='L-BFGS-B',
        options={'maxiter': 500, 'ftol': 1e-10}
    )
    
    if time.time() > deadline:
        return start_params
    
    return result.x

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees)
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree)
        outer_hex_side_length: float
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s kill

    best_s = float('inf')
    best_inner = None
    best_outer = None
    
    # Try multiple starting arrangements
    arrangements = [
        _hexagonal_lattice_positions(),
        _compact_arrangement_1(),
        _compact_arrangement_2(),
    ]
    
    for arr in arrangements:
        if time.time() > deadline:
            break
        
        # Center the arrangement
        com = np.mean(arr[:, :2], axis=0)
        arr_centered = arr.copy()
        arr_centered[:, :2] -= com
        
        # Try different outer hexagon angles
        for angle_deg in np.linspace(0, 60, 13):
            if time.time() > deadline:
                break
            
            # Compute initial s
            all_verts = _get_all_inner_vertices(arr_centered)
            s_cand = _min_s_for_all_vertices(all_verts, np.array([0, 0]), angle_deg)
            
            if s_cand < best_s:
                best_s = s_cand
                best_inner = arr_centered.copy()
                best_outer = np.array([0, 0, angle_deg])
    
    if best_inner is None:
        # Fallback: simple grid
        inner_fallback = np.array([
            [0, 0, 0],
            [-3, 0, 0],
            [3, 0, 0],
            [-1.5, 2.598, 0],
            [1.5, 2.598, 0],
            [-4.5, 2.598, 0],
            [4.5, 2.598, 0],
            [-1.5, -2.598, 0],
            [1.5, -2.598, 0],
            [-4.5, -2.598, 0],
            [4.5, -2.598, 0],
            [0, -5.196, 0]
        ])
        com_fallback = np.mean(inner_fallback[:, :2], axis=0)
        inner_fallback[:, :2] -= com_fallback
        best_inner = inner_fallback
        best_outer = np.array([0, 0, 0])
        best_s = 10.0
    
    # Now optimize from the best starting point
    current_inner = best_inner.copy()
    current_outer = best_outer.copy()
    current_s = best_s
    
    # Extract parameters for optimization
    current_params = np.zeros(24)
    for i in range(12):
        current_params[2*i] = current_inner[i, 0]
        current_params[2*i+1] = current_inner[i, 1]
    
    # Try optimization from multiple outer angles
    for angle_deg in np.linspace(0, 60, 7):
        if time.time() > deadline:
            break
        
        # Optimize
        optimized_params = _optimize_from_start(current_params, angle_deg, deadline)
        
        # Convert back to inner data
        trial_inner = np.zeros((12, 3))
        for i in range(12):
            trial_inner[i, 0] = optimized_params[2*i]
            trial_inner[i, 1] = optimized_params[2*i+1]
            trial_inner[i, 2] = 0.0
        
        # Compute s for this arrangement
        all_verts = _get_all_inner_vertices(trial_inner)
        trial_s = _min_s_for_all_vertices(all_verts, np.array([0, 0]), angle_deg)
        
        # Check validity
        trial_outer = np.array([0, 0, angle_deg])
        if _is_valid(trial_inner, trial_outer, trial_s):
            if trial_s < current_s:
                current_inner = trial_inner
                current_outer = trial_outer
                current_s = trial_s
        
        # Update parameters for next iteration
        for i in range(12):
            current_params[2*i] = current_inner[i, 0]
            current_params[2*i+1] = current_inner[i, 1]
    
    # Final validation
    if not _is_valid(current_inner, current_outer, current_s):
        # Use fallback
        inner_fallback = np.array([
            [0, 0, 0],
            [-3, 0, 0],
            [3, 0, 0],
            [-1.5, 2.598, 0],
            [1.5, 2.598, 0],
            [-4.5, 2.598, 0],
            [4.5, 2.598, 0],
            [-1.5, -2.598, 0],
            [1.5, -2.598, 0],
            [-4.5, -2.598, 0],
            [4.5, -2.598, 0],
            [0, -5.196, 0]
        ])
        com_fallback = np.mean(inner_fallback[:, :2], axis=0)
        inner_fallback[:, :2] -= com_fallback
        current_inner = inner_fallback
        current_outer = np.array([0, 0, 0])
        current_s = 10.0
    
    return current_inner, current_outer, current_s

# EVOLVE-BLOCK-END
