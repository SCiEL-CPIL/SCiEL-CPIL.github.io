# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hexagon_vertices(cx, cy, d, angle_deg):
    """Return the 6 vertices of a regular hexagon with center (cx,cy), side d, angle angle_deg."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _edge_normals(vertices):
    """Return outward unit normals for each edge of a convex polygon given in CCW order."""
    n = len(vertices)
    normals = np.zeros((n, 2))
    for i in range(n):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % n]
        edge = p2 - p1
        # For CCW polygon, outward normal is (edge_y, -edge_x) normalized
        normal = np.array([edge[1], -edge[0]])
        norm = np.linalg.norm(normal)
        if norm > 1e-12:
            normals[i] = normal / norm
        else:
            normals[i] = np.array([1.0, 0.0])
    return normals

def _separating_axis_overlap(v1, n1, v2, n2):
    """Check if two convex polygons (with precomputed vertices and normals) are separated.
    Returns True if separated (no overlap), False if overlapping."""
    # Check axes from polygon 1
    for i in range(6):
        proj1_min = np.min(v1 @ n1[i])
        proj1_max = np.max(v1 @ n1[i])
        proj2_min = np.min(v2 @ n1[i])
        proj2_max = np.max(v2 @ n1[i])
        if proj1_max < proj2_min - 1e-6 or proj2_max < proj1_min - 1e-6:
            return True
    # Check axes from polygon 2
    for i in range(6):
        proj1_min = np.min(v1 @ n2[i])
        proj1_max = np.max(v1 @ n2[i])
        proj2_min = np.min(v2 @ n2[i])
        proj2_max = np.max(v2 @ n2[i])
        if proj1_max < proj2_min - 1e-6 or proj2_max < proj1_min - 1e-6:
            return True
    return False

def _point_in_hexagon(v, vertices):
    """Check if point v is inside convex polygon given vertices in CCW order.
    Returns True if inside or on boundary."""
    for i in range(6):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % 6]
        cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
        if cross < -1e-6:
            return False
    return True

def _compute_objective(params):
    """
    params: array of length 12*3 + 3 = 39
    First 36: 12 hexagons, each (x, y, angle_deg)
    Last 3: outer hexagon (x, y, angle_deg)
    Returns: 1/s (to be maximized) if valid, else a large penalty.
    """
    inner = params[:36].reshape(12, 3)
    outer = params[36:39]
    
    ox, oy, oangle = outer
    # We need to determine s. Actually, s is not in params directly.
    # Let's re-parameterize: the outer hexagon is defined by center (ox, oy), angle, and we optimize s separately?
    # No, the task says return s. The objective is 1/s.
    # So s should be a parameter. Let's make params length 12*3 + 4 = 40.
    # But the current function signature returns (inner, outer, s).
    # Let's stick with 12*3 + 4 parameters: 12 inner (x,y,ang), outer (x,y,ang), s.
    
    # Wait, I defined params as 39 above. Let's fix that.
    # Let's redefine _compute_objective to take 40 params.
    
    # This function is a helper, let's just return a dummy for now and fix in the main optimizer.
    return 0.0

def _objective(params):
    """
    params: array of length 40
    0-35: 12 inner hexagons (x, y, angle_deg)
    36-38: outer hexagon (x, y, angle_deg)
    39: outer hexagon side length s
    Returns: negative 1/s (for minimization) + penalty
    """
    inner = params[:36].reshape(12, 3)
    outer_xy = params[36:38]
    outer_ang = params[38]
    s = params[39]
    
    if s <= np.sqrt(12):
        return 1e10
    
    # Compute outer vertices
    outer_verts = _hexagon_vertices(outer_xy[0], outer_xy[1], s, outer_ang)
    
    # Check outer vertices distinct
    for i in range(6):
        for j in range(i+1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-6:
                return 1e10
    
    # Compute inner vertices and normals
    inner_verts_list = []
    inner_normals_list = []
    for i in range(12):
        cx, cy, ang = inner[i]
        verts = _hexagon_vertices(cx, cy, 1.0, ang)
        inner_verts_list.append(verts)
        inner_normals_list.append(_edge_normals(verts))
    
    # Check separation between inner hexagons
    for i in range(12):
        for j in range(i+1, 12):
            if not _separating_axis_overlap(inner_verts_list[i], inner_normals_list[i], 
                                            inner_verts_list[j], inner_normals_list[j]):
                # Overlapping
                return 1e10
    
    # Check inner vertices inside outer hexagon
    for i in range(12):
        for v in inner_verts_list[i]:
            if not _point_in_hexagon(v, outer_verts):
                return 1e10
    
    # Valid packing
    return -1.0 / s

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 600.0  # 600 seconds limit, leave margin
    
    # Initial guess: honeycomb arrangement
    # Center at (0,0), 6 around it, 6 further out
    # Unit hexagon has width 2 (flat-to-flat), height sqrt(3) (point-to-point)
    # Distance between centers of touching hexagons is 2 (if flat sides touch) or 2*sqrt(3)/2?
    # For flat-top hexagons, horizontal distance is 3, vertical is sqrt(3)?
    # Let's use a simple grid:
    # Row 0: (-3, 0), (0, 0), (3, 0)
    # Row 1: (-1.5, sqrt(3)), (1.5, sqrt(3))
    # Row 2: (-4.5, sqrt(3)), (-1.5, 2*sqrt(3)), (1.5, 2*sqrt(3)), (4.5, sqrt(3))
    # This is getting complicated. Let's just use a random start or a known good start.
    
    # Known good start: 12 hexagons in a hexagonal cluster
    # Center (0,0)
    # 6 around at distance 2, angles 0, 60, 120, 180, 240, 300
    # 6 more around that? No, that's 19.
    # 12 hexagons: maybe 3 in a row, 3 in next, 3, 3?
    
    # Let's try a simple random initialization and optimize
    np.random.seed(42)
    
    best_params = None
    best_val = -np.inf
    
    n_restarts = 10
    for r in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        # Random initial positions in a circle
        angles = np.random.uniform(0, 2*np.pi, 12)
        radii = np.random.uniform(0, 5, 12)
        inner_init = np.column_stack([radii * np.cos(angles), radii * np.sin(angles), np.zeros(12)])
        
        outer_init = np.array([0, 0, 0, 10.0])
        
        params_init = np.concatenate([inner_init.flatten(), outer_init])
        
        # Nelder-Mead
        res = minimize(_objective, params_init, method='Nelder-Mead', 
                       options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-6})
        
        if res.fun < best_val:
            best_val = res.fun
            best_params = res.x
        
        if time.time() > deadline - 60:
            break
    
    if best_params is None:
        # Fallback to the original solution
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
                [0, -4, 0]
            ]
        )
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8
        return inner_hex_data, outer_hex_data, outer_hex_side_length
    
    inner = best_params[:36].reshape(12, 3)
    outer = best_params[36:39]
    s = best_params[39]
    
    return inner, outer, s

# EVOLVE-BLOCK-END
