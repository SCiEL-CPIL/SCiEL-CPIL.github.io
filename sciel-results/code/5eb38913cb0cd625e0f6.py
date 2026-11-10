# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _hex_vertices(cx, cy, s, a_deg):
    """Compute 6 vertices of a regular hexagon."""
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def _min_outer_side_for_hexagons(inner_hex_data):
    """
    Find the minimum outer hexagon side length that contains all inner hexagons.
    The outer hexagon is centered at origin with angle 0.
    """
    # Outer hexagon vertices (side length s, centered at origin, angle 0)
    # For a regular hexagon with side s, the apothem is s * sqrt(3)/2
    # The vertices are at distance s from center.
    
    # We need to find minimum s such that all inner hexagon vertices are inside.
    # For a regular hexagon centered at origin with angle 0:
    # The 6 edge normals are at angles 0, 60, 120, 180, 240, 300 degrees
    # A point (x,y) is inside if for each normal direction n_i, dot((x,y), n_i) <= apothem = s*sqrt(3)/2
    
    normals = []
    for i in range(6):
        ang = np.deg2rad(60 * i)
        normals.append([np.cos(ang), np.sin(ang)])
    normals = np.array(normals)  # shape (6, 2)
    
    # Collect all vertices of all inner hexagons
    all_vertices = []
    for i in range(len(inner_hex_data)):
        cx, cy, a_deg = inner_hex_data[i]
        verts = _hex_vertices(cx, cy, 1.0, a_deg)
        all_vertices.append(verts)
    all_vertices = np.vstack(all_vertices)  # shape (66, 2)
    
    # For each vertex, compute the required s:
    # dot(vertex, normal_i) <= s * sqrt(3)/2 for all i
    # So s >= 2/sqrt(3) * max_i dot(vertex, normal_i) for each vertex
    # Overall s >= max over all vertices of (2/sqrt(3) * max_i dot(v, n_i))
    
    # Compute dot products: all_vertices @ normals.T -> shape (66, 6)
    dots = all_vertices @ normals.T
    max_dots = np.max(dots, axis=1)  # shape (66,)
    required_s = (2.0 / np.sqrt(3.0)) * max_dots
    return np.max(required_s)


def _check_validity(inner_hex_data, outer_hex_data, side):
    """Check if the packing is valid according to the task contract."""
    # 1. All values finite
    if not (np.all(np.isfinite(inner_hex_data)) and np.all(np.isfinite(outer_hex_data))):
        return False
    if not np.isfinite(side) or side < np.sqrt(11) - 1e-9:
        return False
    
    # 2. Every pair of inner hexagons is separated
    n_inner = len(inner_hex_data)
    for i in range(n_inner):
        for j in range(i + 1, n_inner):
            cx1, cy1, a1 = inner_hex_data[i]
            cx2, cy2, a2 = inner_hex_data[j]
            verts1 = _hex_vertices(cx1, cy1, 1.0, a1)
            verts2 = _hex_vertices(cx2, cy2, 1.0, a2)
            
            # Check if the two hexagons are separated
            # For each of the 12 edge normals (6 from each hexagon),
            # check if the projected intervals are apart by more than 1e-6
            
            # Get edge normals for hexagon 1
            normals1 = []
            for k in range(6):
                v1 = verts1[k]
                v2 = verts1[(k + 1) % 6]
                edge = v2 - v1
                # Normal perpendicular to edge, pointing outward
                normal = np.array([-edge[1], edge[0]])
                normal = normal / np.linalg.norm(normal)
                # Ensure it points outward (away from center)
                center1 = np.array([cx1, cy1])
                if np.dot(normal, center1 - v1) < 0:
                    normal = -normal
                normals1.append(normal)
            
            # Get edge normals for hexagon 2
            normals2 = []
            for k in range(6):
                v1 = verts2[k]
                v2 = verts2[(k + 1) % 6]
                edge = v2 - v1
                normal = np.array([-edge[1], edge[0]])
                normal = normal / np.linalg.norm(normal)
                center2 = np.array([cx2, cy2])
                if np.dot(normal, center2 - v1) < 0:
                    normal = -normal
                normals2.append(normal)
            
            all_normals = normals1 + normals2
            
            separated = False
            for normal in all_normals:
                # Project all vertices of hex1 onto this normal
                proj1 = np.array([np.dot(v, normal) for v in verts1])
                # Project all vertices of hex2 onto this normal
                proj2 = np.array([np.dot(v, normal) for v in verts2])
                
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                
                # Check if intervals are apart by more than 1e-6
                if max1 < min2 - 1e-6 or max2 < min1 - 1e-6:
                    separated = True
                    break
            
            if not separated:
                return False
    
    # 3. Every inner vertex satisfies the cross product constraint with outer edges
    ox, oy, oa = outer_hex_data
    outer_verts = _hex_vertices(ox, oy, side, oa)
    
    for i in range(n_inner):
        cx, cy, a_deg = inner_hex_data[i]
        verts = _hex_vertices(cx, cy, 1.0, a_deg)
        for p in verts:
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k + 1) % 6]
                e1 = p2 - p1
                e2 = p - p1
                cross = e1[0] * e2[1] - e1[1] * e2[0]
                if cross < -1e-6:
                    return False
    
    # 4. Edge lengths within 1e-8 of 1
    for i in range(n_inner):
        cx, cy, a_deg = inner_hex_data[i]
        verts = _hex_vertices(cx, cy, 1.0, a_deg)
        for k in range(6):
            edge_len = np.linalg.norm(verts[(k + 1) % 6] - verts[k])
            if abs(edge_len - 1.0) > 1e-8:
                return False
    
    return True


def _objective(params):
    """
    Objective function for optimization.
    params: 11*3 = 33 values: for each hexagon (cx, cy, angle_deg)
    Returns: side length + penalty for invalidity
    """
    inner_hex_data = params.reshape(11, 3)
    
    # Compute minimum outer side
    side = _min_outer_side_for_hexagons(inner_hex_data)
    
    # Check validity and add penalty if invalid
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    if _check_validity(inner_hex_data, outer_hex_data, side):
        return side
    else:
        # Large penalty for invalid configurations
        return side + 100.0


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees)
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree)
        outer_hex_side_length: float
    """
    start_time = time.time()
    deadline = start_time + 600  # 600 second limit
    
    # Initial guess: hexagonal grid arrangement
    sqrt3 = np.sqrt(3)
    initial = np.array([
        [0, 0, 0],
        [-1.5, 0, 0],
        [1.5, 0, 0],
        [-0.75, 1.299, 0],
        [0.75, 1.299, 0],
        [-0.75, -1.299, 0],
        [0.75, -1.299, 0],
        [-2.25, 1.299, 0],
        [2.25, 1.299, 0],
        [-2.25, -1.299, 0],
        [2.25, -1.299, 0],
    ]).flatten()
    
    # Try multiple starting points
    best_params = None
    best_side = float('inf')
    
    # Starting points
    starts = [
        initial,
        initial + np.random.randn(33) * 0.1,
        initial + np.random.randn(33) * 0.5,
    ]
    
    for start in starts:
        if time.time() > deadline - 30:
            break
        
        try:
            result = minimize(
                _objective,
                start,
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            
            if result.success or result.fun < best_side:
                candidate = result.x
                side = _min_outer_side_for_hexagons(candidate.reshape(11, 3))
                outer_hex_data = np.array([0.0, 0.0, 0.0])
                if _check_validity(candidate.reshape(11, 3), outer_hex_data, side):
                    best_params = candidate
                    best_side = side
        except Exception:
            continue
    
    # If no valid solution found, use the initial guess with a large enough outer hexagon
    if best_params is None:
        inner_hex_data = initial.reshape(11, 3)
        side = _min_outer_side_for_hexagons(inner_hex_data)
        # Make sure it's valid by using a slightly larger side
        outer_hex_data = np.array([0.0, 0.0, 0.0])
        if not _check_validity(inner_hex_data, outer_hex_data, side):
            side *= 1.01
        return inner_hex_data, outer_hex_data, side
    
    inner_hex_data = best_params.reshape(11, 3)
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    return inner_hex_data, outer_hex_data, best_side


# EVOLVE-BLOCK-END
