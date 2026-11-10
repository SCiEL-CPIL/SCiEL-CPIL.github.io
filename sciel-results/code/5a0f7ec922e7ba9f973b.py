# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import itertools


def _hex_vertices(cx, cy, s, a_deg):
    """Return vertices of a hexagon with center (cx,cy), side s, rotation a_deg."""
    a = np.radians(a_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def _hex_edge_normals(cx, cy, s, a_deg):
    """Return 12 edge normals (6 edges, 2 directions each) of a unit hexagon."""
    a = np.radians(a_deg)
    # Edge directions: between consecutive vertices
    angles = a + np.radians(60 * np.arange(6))
    verts = np.column_stack([s * np.cos(angles), s * np.sin(angles)])
    # Edge normals: perpendicular to each edge, pointing outward
    normals = []
    for i in range(6):
        v1 = verts[i]
        v2 = verts[(i + 1) % 6]
        edge = v2 - v1
        # Outward normal (rotate edge by -90 degrees for CCW polygon)
        n = np.array([edge[1], -edge[0]])
        n = n / np.linalg.norm(n)
        normals.append(n)
        normals.append(-n)
    return np.array(normals)


def _separation_check(hex1, hex2):
    """Check if two unit hexagons are separated (no intersection).
    Returns True if separated by more than 1e-6 on at least one of 12 edge normals."""
    cx1, cy1, a1 = hex1
    cx2, cy2, a2 = hex2
    
    # Get vertices relative to centers
    a1_rad = np.radians(a1)
    a2_rad = np.radians(a2)
    
    angles1 = a1_rad + np.radians(60 * np.arange(6))
    angles2 = a2_rad + np.radians(60 * np.arange(6))
    
    verts1 = np.column_stack([np.cos(angles1), np.sin(angles1)])
    verts2 = np.column_stack([np.cos(angles2), np.sin(angles2)])
    
    d = np.array([cx2 - cx1, cy2 - cy1])
    
    # Check all 12 edge normals of hex1
    for i in range(6):
        v1 = verts1[i]
        v2 = verts1[(i + 1) % 6]
        edge = v2 - v1
        n = np.array([edge[1], -edge[0]])
        n = n / np.linalg.norm(n)
        
        # Project hex1 onto n
        proj1 = verts1 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        
        # Project hex2 onto n (shifted by d)
        proj2 = (verts2 + d) @ n
        min2, max2 = np.min(proj2), np.max(proj2)
        
        # Check if intervals are apart by more than 1e-6
        gap1 = max1 < min2 - 1e-6
        gap2 = max2 < min1 - 1e-6
        if gap1 or gap2:
            continue  # This normal shows separation
        else:
            break  # Not separated by this normal
    
    # If we broke, check hex2's normals too
    else:
        return True  # All hex1 normals show separation
    
    # Check all 12 edge normals of hex2
    for i in range(6):
        v1 = verts2[i]
        v2 = verts2[(i + 1) % 6]
        edge = v2 - v1
        n = np.array([edge[1], -edge[0]])
        n = n / np.linalg.norm(n)
        
        # Project hex1 onto n (shifted by -d)
        proj1 = (verts1 - d) @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        
        # Project hex2 onto n
        proj2 = verts2 @ n
        min2, max2 = np.min(proj2), np.max(proj2)
        
        gap1 = max1 < min2 - 1e-6
        gap2 = max2 < min1 - 1e-6
        if gap1 or gap2:
            continue
        else:
            return False
    
    return True


def _all_separated(hexagons):
    """Check if all pairs of hexagons are separated."""
    n = len(hexagons)
    for i in range(n):
        for j in range(i + 1, n):
            if not _separation_check(hexagons[i], hexagons[j]):
                return False
    return True


def _outer_side_needed(hexagons):
    """Compute the minimum outer hexagon side length needed to contain all inner hexagons.
    The outer hexagon is centered at origin with side length S and angle 0."""
    # For a regular hexagon centered at origin with side S and angle 0,
    # the vertices are at distance S from center.
    # A point (x,y) is inside if for each of the 6 edge normals, the projection is within bounds.
    # Edge normals of outer hex (angle 0): directions 0, 60, 120, 180, 240, 300 degrees
    # The constraint is: |x*cos(theta) + y*sin(theta)| <= S * sin(60) = S * sqrt(3)/2
    # where theta ranges over the 6 normal directions.
    
    max_needed = 0
    for hex in hexagons:
        cx, cy, a = hex
        verts = _hex_vertices(cx, cy, 1.0, a)
        for v in verts:
            for theta in np.radians([0, 60, 120, 180, 240, 300]):
                proj = np.abs(v[0] * np.cos(theta) + v[1] * np.sin(theta))
                needed = proj / (np.sqrt(3) / 2)
                max_needed = max(max_needed, needed)
    return max_needed


def _pack_with_optimizer(initial_positions, initial_angles, max_time=580):
    """Optimize positions and angles to minimize outer hex side length."""
    n = 11
    
    def objective(params):
        positions = params[:2 * n].reshape(n, 2)
        angles = params[2 * n:]
        
        hexagons = []
        for i in range(n):
            hexagons.append((positions[i, 0], positions[i, 1], np.degrees(angles[i])))
        
        # Check separation
        if not _all_separated(hexagons):
            return 1e6
        
        return _outer_side_needed(hexagons)
    
    def constraint_check(params):
        positions = params[:2 * n].reshape(n, 2)
        angles = params[2 * n:]
        hexagons = [(positions[i, 0], positions[i, 1], np.degrees(angles[i])) for i in range(n)]
        return _all_separated(hexagons)
    
    x0 = np.concatenate([initial_positions.flatten(), initial_angles])
    
    try:
        result = minimize(objective, x0, method='Nelder-Mead',
                         options={'maxiter': 5000, 'fatol': 1e-8, 'xatol': 1e-8,
                                  'adaptive': True})
        
        if result.success or result.fun < 1e6:
            positions = result.x[:2 * n].reshape(n, 2)
            angles = result.x[2 * n:]
            
            hexagons = [(positions[i, 0], positions[i, 1], np.degrees(angles[i])) for i in range(n)]
            
            if _all_separated(hexagons):
                side = _outer_side_needed(hexagons)
                inner_data = np.array([[hexagons[i][0], hexagons[i][1], hexagons[i][2]] for i in range(n)])
                outer_data = np.array([0.0, 0.0, 0.0])
                return inner_data, outer_data, side
    except Exception:
        pass
    
    return None


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon.
    Returns
        inner_hex_data: np.ndarray of shape (11,3)
        outer_hex_data: np.ndarray of shape (3,)
        outer_hex_side_length: float
    """
    start_time = time.time()
    deadline = start_time + 580  # 20 second safety margin
    
    n = 11
    best_result = None
    best_side = float('inf')
    
    # Initial hexagonal grid arrangement
    base_positions = np.array([
        [0, 0],
        [-2.5, 0],
        [2.5, 0],
        [-1.25, 2.165],
        [1.25, 2.165],
        [-1.25, -2.165],
        [1.25, -2.165],
        [-3.75, 2.165],
        [3.75, 2.165],
        [-3.75, -2.165],
        [3.75, -2.165],
    ])
    base_angles = np.zeros(n)
    
    # Try the base arrangement first
    hexagons = [(base_positions[i, 0], base_positions[i, 1], np.degrees(base_angles[i])) for i in range(n)]
    if _all_separated(hexagons):
        side = _outer_side_needed(hexagons)
        inner_data = base_positions.copy()
        inner_data = np.column_stack([inner_data, np.degrees(base_angles)])
        outer_data = np.array([0.0, 0.0, 0.0])
        best_result = (inner_data, outer_data, side)
        best_side = side
    
    # Try optimization with perturbed initial conditions
    np.random.seed(42)
    for trial in range(5):
        if time.time() > deadline:
            break
        
        # Perturb positions and angles
        perturb_pos = np.random.randn(n, 2) * 0.5
        perturb_ang = np.random.randn(n) * 0.3
        init_pos = base_positions + perturb_pos
        init_ang = base_angles + perturb_ang
        
        result = _pack_with_optimizer(init_pos, init_ang, max_time=100)
        if result is not None:
            inner_data, outer_data, side = result
            if side < best_side:
                best_side = side
                best_result = (inner_data, outer_data, side)
    
    if best_result is None:
        # Fallback: return base arrangement
        inner_data = np.column_stack([base_positions, np.degrees(base_angles)])
        outer_data = np.array([0.0, 0.0, 0.0])
        side = _outer_side_needed([(base_positions[i, 0], base_positions[i, 1], 0) for i in range(n)])
        best_result = (inner_data, outer_data, side)
    
    return best_result


# EVOLVE-BLOCK-END
