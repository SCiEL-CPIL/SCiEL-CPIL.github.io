import numpy as np
import time

N = 11

def get_hex_vertices(center, angle_deg):
    """
    Returns vertices of a unit regular hexagon centered at 'center'
    with rotation 'angle_deg'.
    """
    cx, cy = center
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + angle_deg)
    xs = cx + np.cos(angles)
    ys = cy + np.sin(angles)
    return np.column_stack((xs, ys))

def min_enclosing_hexagon_side(vertices):
    """
    Finds the minimum side length S of a regular hexagon (centered at origin)
    that encloses the given vertices.
    """
    best_S = float('inf')
    best_alpha = 0.0
    # Search for optimal rotation of the outer hexagon
    for alpha_deg in np.linspace(0, 30, 300):
        phi = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + alpha_deg)
        normals = np.column_stack((np.cos(phi), np.sin(phi)))
        
        dots = vertices @ normals.T
        max_dots = np.max(dots, axis=0)
        h = np.max(max_dots)
        S = 2 * h / np.sqrt(3)
        
        if S < best_S:
            best_S = S
            best_alpha = alpha_deg
            
    return best_S, best_alpha

def construct():
    start_time = time.time()
    
    # Generate hex grid centers (axial to cartesian)
    centers = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            d = (abs(q) + abs(r) + abs(q+r)) / 2
            if d <= 2:
                x = np.sqrt(3) * (q + r/2)
                y = 1.5 * r
                centers.append((x, y, d))
                
    centers.sort(key=lambda c: c[2])
    
    base_centers = [c[:2] for c in centers[:7]]
    ring2_centers = [c[:2] for c in centers[7:19]]
    
    candidate_sets = []
    
    # Symmetric subsets of ring 2
    idxs_sets = [
        [0, 3, 6, 9],
        [0, 2, 5, 7],
        [0, 1, 6, 7],
        [0, 4, 6, 10],
        [1, 3, 5, 7],
        [1, 5, 7, 11],
    ]
    
    for idxs in idxs_sets:
        extra = [ring2_centers[i] for i in idxs]
        candidate_sets.append(base_centers + extra)
        
    best_result = None
    best_S = float('inf')
    
    for centers_set in candidate_sets:
        for inner_rot in [0.0, 30.0]:
            all_vertices = []
            for c in centers_set:
                verts = get_hex_vertices(c, inner_rot)
                all_vertices.append(verts)
            all_vertices = np.vstack(all_vertices)
            
            S, alpha = min_enclosing_hexagon_side(all_vertices)
            
            if S < best_S:
                best_S = S
                best_result = {
                    "inner": [[c[0], c[1], inner_rot] for c in centers_set],
                    "center": (0.0, 0.0),
                    "side": S,
                    "angle": alpha
                }
                
    if best_result is None:
        return [[0.0, 0.0, 0.0]], (0.0, 0.0), 10.0, 0.0

    return best_result["inner"], best_result["center"], best_result["side"], best_result["angle"]
