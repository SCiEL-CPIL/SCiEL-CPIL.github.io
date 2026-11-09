import numpy as np
import time
from scipy.optimize import linprog

N = 11

def hex_vertices(cx, cy, s, angle_deg):
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3
        verts.append([cx + s * np.cos(theta), cy + s * np.sin(theta)])
    return np.array(verts)

def solve_min_hexagon(vertices, angle_deg):
    """
    Solve for the minimum side length S and center (cx, cy) of a regular hexagon
    with given angle that encloses all points in 'vertices'.
    Uses Linear Programming.
    """
    a = np.radians(angle_deg)
    # Normals to the 6 sides of the hexagon
    # The sides are perpendicular to directions a + pi/6 + i*pi/3
    normals = np.array([[np.cos(a + np.pi/6 + i * np.pi / 3), 
                         np.sin(a + np.pi/6 + i * np.pi / 3)] for i in range(6)])
    
    # Variables: x = [cx, cy, S]
    # Minimize S -> c = [0, 0, 1]
    c = np.array([0.0, 0.0, 1.0])
    
    # Constraints:
    # For each vertex p and each normal n_i:
    # n_i . (p - c) <= S * sqrt(3) / 2
    # n_i . p - n_i . c <= S * sqrt(3) / 2
    # -n_i . c - S * sqrt(3) / 2 + n_i . p <= 0  (Rearranged for A_ub @ x <= b_ub)
    # Wait, standard form is A_ub @ x <= b_ub.
    # Let x = [cx, cy, S]
    # n_i_x * (p_x - cx) + n_i_y * (p_y - cy) <= S * sqrt(3) / 2
    # -n_i_x * cx - n_i_y * cy - S * sqrt(3) / 2 <= -n_i . p
    # This is one constraint.
    # The other direction:
    # -n_i . (p - c) <= S * sqrt(3) / 2
    # n_i . (p - c) >= -S * sqrt(3) / 2
    # n_i . c - n_i . p <= S * sqrt(3) / 2
    # n_i_x * cx + n_i_y * cy - S * sqrt(3) / 2 <= n_i . p
    
    # So for each p and each i, we have 2 constraints.
    A_ub_rows = []
    b_ub_rows = []
    
    sqrt3_over_2 = np.sqrt(3) / 2.0
    
    for p in vertices:
        for i in range(6):
            nx, ny = normals[i]
            # Constraint 1: n_i . (p - c) <= S * R
            # -nx*cx - ny*cy - R*S <= -n_i . p
            A_ub_rows.append([-nx, -ny, -sqrt3_over_2])
            b_ub_rows.append(-(nx * p[0] + ny * p[1]))
            
            # Constraint 2: -n_i . (p - c) <= S * R  =>  n_i . (p - c) >= -S * R
            # nx*cx + ny*cy - R*S <= n_i . p
            A_ub_rows.append([nx, ny, -sqrt3_over_2])
            b_ub_rows.append(nx * p[0] + ny * p[1])
            
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_rows)
    
    # Bounds: S >= 0, cx, cy can be anything (but practically bounded by points)
    # To help linprog, we can bound cx, cy to the bounding box of points
    min_x = vertices[:, 0].min()
    max_x = vertices[:, 0].max()
    min_y = vertices[:, 1].min()
    max_y = vertices[:, 1].max()
    
    bounds = [
        (min_x - 5, max_x + 5),
        (min_y - 5, max_y + 5),
        (0, None)
    ]
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res.success:
        cx, cy, S = res.x
        return S, cx, cy
    else:
        return None, None, None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Define the 11 inner hexagons in a hexagonal cluster
    # Axial coordinates (q, r)
    # Center (0,0)
    # Ring 1: (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
    # Ring 2 (partial to make 11): (2,0), (2,-1), (1,-2), (0,-2)
    # This forms a shape like a larger hexagon missing 3 corners?
    # Let's verify non-overlap.
    # Distance between adjacent centers is sqrt(3).
    # This is a valid packing.
    
    hex_positions_axial = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
        (2, 0), (2, -1), (1, -2), (0, -2),
    ]
    
    centers = []
    for q, r in hex_positions_axial:
        x = 1.5 * q
        y = np.sqrt(3) * (r + 0.5 * q)
        centers.append([x, y])
    
    centers = np.array(centers)
    
    # Center the configuration at origin for symmetry
    centers -= centers.mean(axis=0)
    
    # Generate all vertices of the inner hexagons (all aligned at 0 deg)
    all_vertices = []
    for cx, cy in centers:
        v = hex_vertices(cx, cy, 1.0, 0.0)
        all_vertices.append(v)
    all_vertices = np.vstack(all_vertices)
    
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Try a few outer angles
    # 0 and 30 are the primary candidates for hexagonal symmetry
    angles_to_try = [0.0, 30.0]
    
    for angle in angles_to_try:
        if time.time() > deadline:
            break
            
        side, cx, cy = solve_min_hexagon(all_vertices, angle)
        
        if side is not None and side < best_side:
            best_side = side
            best_center = (cx, cy)
            best_angle = angle
            
    # If linprog failed or returned inf, fallback to a safe large box
    if best_side == float('inf') or best_side is None:
        # Calculate bounding box
        min_x = all_vertices[:, 0].min()
        max_x = all_vertices[:, 0].max()
        min_y = all_vertices[:, 1].min()
        max_y = all_vertices[:, 1].max()
        width = max_x - min_x
        height = max_y - min_y
        # Approximate side
        best_side = max(width, height) / np.sqrt(3) * 1.5 # Safety factor
        best_center = ((min_x + max_x)/2, (min_y + max_y)/2)
        best_angle = 0.0

    # Add a tiny safety margin to side length to ensure containment due to floating point errors
    best_side *= 1.001
    
    inner = [[float(cx), float(cy), 0.0] for cx, cy in centers]
    
    return inner, (float(best_center[0]), float(best_center[1])), float(best_side), float(best_angle)
