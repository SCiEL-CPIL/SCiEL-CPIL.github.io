import numpy as np
import time

N = 12
DEADLINE = 140.0

def hexagon_vertices_unit(cx, cy, angle_deg):
    """Return 6 vertices of a unit regular hexagon (side=1) centered at (cx,cy)."""
    angle = np.radians(angle_deg)
    angles = angle + np.radians(np.arange(6) * 60)
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])

def min_side_for_hexagons(inner_positions, outer_center, outer_angle_deg):
    """Find minimum side length of outer hexagon that contains all inner hexagons' vertices."""
    cx, cy = outer_center
    all_verts = []
    for (ix, iy, iangle) in inner_positions:
        verts = hexagon_vertices_unit(ix, iy, iangle)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    angle = np.radians(outer_angle_deg)
    normals = []
    for k in range(6):
        edge_angle = np.radians(30 + k * 60) + angle
        normals.append(np.array([np.cos(edge_angle), np.sin(edge_angle)]))
    normals = np.array(normals)
    
    rel = all_verts - np.array([cx, cy])
    dots = rel @ normals.T
    max_dot = np.max(dots)
    s_min = max_dot / (np.sqrt(3) / 2)
    return max(s_min, 0.0)

def construct():
    start_time = time.time()
    
    # Multiple arrangements of 12 unit hexagons in flat-top hex grid
    arrangements = []
    
    # Arrangement 1: 4x3 block
    arr1 = []
    for col in range(4):
        for row in range(3):
            x = 1.5 * col
            y = np.sqrt(3) * row
            if col % 2 == 1:
                y += np.sqrt(3) / 2
            arr1.append([x, y, 0.0])
    arrangements.append(arr1)
    
    # Arrangement 2: 3x4 block
    arr2 = []
    for col in range(3):
        for row in range(4):
            x = 1.5 * col
            y = np.sqrt(3) * row
            if col % 2 == 1:
                y += np.sqrt(3) / 2
            arr2.append([x, y, 0.0])
    arrangements.append(arr2)
    
    # Arrangement 3: staggered rows
    arr3 = []
    for i in range(4):
        arr3.append([i * 1.5, 0.0, 0.0])
    for i in range(4):
        arr3.append([0.75 + i * 1.5, -np.sqrt(3) / 2, 0.0])
    for i in range(4):
        arr3.append([1.5 + i * 1.5, -np.sqrt(3), 0.0])
    arrangements.append(arr3)
    
    # Arrangement 4: more compact hexagonal cluster
    # Center + 6 neighbors + 5 more
    arr4 = []
    arr4.append([0.0, 0.0, 0.0])  # center
    # 6 neighbors
    neighbors = [
        (1.5, 0.0), (0.75, np.sqrt(3)/2), (-0.75, np.sqrt(3)/2),
        (-1.5, 0.0), (-0.75, -np.sqrt(3)/2), (0.75, -np.sqrt(3)/2)
    ]
    for (nx, ny) in neighbors:
        arr4.append([nx, ny, 0.0])
    # 5 more: extend in some directions
    arr4.append([3.0, 0.0, 0.0])
    arr4.append([1.5, np.sqrt(3), 0.0])
    arr4.append([0.0, np.sqrt(3), 0.0])
    arr4.append([-1.5, np.sqrt(3), 0.0])
    arr4.append([-3.0, 0.0, 0.0])
    arrangements.append(arr4)
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    for arr in arrangements:
        cx = np.mean([p[0] for p in arr])
        cy = np.mean([p[1] for p in arr])
        
        for outer_angle in [0, 30, 60, 90, 120, 150]:
            # Center the arrangement at origin
            arr_c = [[p[0] - cx, p[1] - cy, p[2]] for p in arr]
            
            # Try small offsets for outer center
            for dx in [-0.3, -0.15, 0.0, 0.15, 0.3]:
                for dy in [-0.3, -0.15, 0.0, 0.15, 0.3]:
                    arr_shifted = [[p[0] + dx, p[1] + dy, p[2]] for p in arr_c]
                    side = min_side_for_hexagons(arr_shifted, (0.0, 0.0), outer_angle)
                    if side < best_side:
                        best_side = side
                        best_inner = arr_shifted
                        best_center = (0.0, 0.0)
                        best_angle = outer_angle
        
        if time.time() - start_time > DEADLINE:
            break
    
    if best_inner is None:
        inner = [[i * 2.0, 0.0, 0.0] for i in range(N)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_inner, best_center, best_side, best_angle
