import numpy as np
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    sqrt3 = np.sqrt(3)
    
    def hex_vertices(cx, cy, side, angle_deg):
        """Return vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
        angles = np.array([0, 60, 120, 180, 240, 300]) + angle_deg
        angles_rad = np.radians(angles)
        r = side  # circumradius = side length for regular hexagon
        x = cx + r * np.cos(angles_rad)
        y = cy + r * np.sin(angles_rad)
        return np.column_stack([x, y])
    
    def min_enclosing_hexagon(vertices):
        """Find minimum side length of a regular hexagon containing all vertices.
        vertices: Nx2 array. Returns (center_x, center_y, side_length, angle_deg)."""
        best_side = float('inf')
        best_center = (0.0, 0.0)
        best_angle = 0.0
        
        centroid = np.mean(vertices, axis=0)
        
        # Try various rotations of the outer hexagon
        for angle_deg in np.linspace(0, 60, 121):
            if time.time() > deadline:
                break
            angle_rad = np.radians(angle_deg)
            normals = np.array([[np.cos(angle_rad + i * np.pi / 3), 
                                 np.sin(angle_rad + i * np.pi / 3)] for i in range(6)])
            
            required_side = 0.0
            for i in range(6):
                n = normals[i]
                projections = vertices @ n
                max_proj = np.max(projections)
                min_proj = np.min(projections)
                max_dist = max_proj - centroid @ n
                min_dist = min_proj - centroid @ n
                required_inradius = max(abs(max_dist), abs(min_dist))
                required_side_i = required_inradius * 2 / sqrt3
                if required_side_i > required_side:
                    required_side = required_side_i
            
            if required_side < best_side:
                best_side = required_side
                best_center = tuple(centroid)
                best_angle = angle_deg
        
        return best_center, best_side, best_angle
    
    def evaluate_arrangement(positions, angles_deg):
        """Given positions and angles of N hexagons, compute bounding hexagon."""
        all_vertices = []
        for (x, y), ang in zip(positions, angles_deg):
            verts = hex_vertices(x, y, 1.0, ang)
            all_vertices.append(verts)
        vertices = np.vstack(all_vertices)
        center, side, angle = min_enclosing_hexagon(vertices)
        return center, side, angle
    
    # Define several candidate arrangements
    
    # Arrangement 1: Linear row
    positions1 = [(4.0 * k, 0.0) for k in range(-N // 2, N - N // 2)]
    angles1 = [0.0] * N
    
    # Arrangement 2: 2 rows of 6, honeycomb
    positions2 = []
    angles2 = []
    for row in range(2):
        for col in range(6):
            x = col * sqrt3 - (5 * sqrt3) / 2
            y = row * 1.5 - 0.75
            if row % 2 == 1:
                x += sqrt3 / 2
            positions2.append((x, y))
            angles2.append(0.0)
    
    # Arrangement 3: 3 rows of 4, honeycomb
    positions3 = []
    angles3 = []
    for row in range(3):
        for col in range(4):
            x = col * sqrt3 - (3 * sqrt3) / 2
            y = row * 1.5 - 1.5
            if row % 2 == 1:
                x += sqrt3 / 2
            positions3.append((x, y))
            angles3.append(0.0)
    
    # Arrangement 4: Compact hexagonal cluster (center + ring)
    positions4 = []
    angles4 = []
    positions4.append((0.0, 0.0))
    angles4.append(0.0)
    for i in range(6):
        ang = i * 60
        x = sqrt3 * np.cos(np.radians(ang))
        y = sqrt3 * np.sin(np.radians(ang))
        positions4.append((x, y))
        angles4.append(0.0)
    # Add 5 more in a second ring
    for i in range(5):
        ang = i * 60 + 30
        x = 2 * sqrt3 * np.cos(np.radians(ang))
        y = 2 * sqrt3 * np.sin(np.radians(ang))
        positions4.append((x, y))
        angles4.append(0.0)
    
    # Arrangement 5: 4x3 grid
    positions5 = []
    angles5 = []
    for row in range(3):
        for col in range(4):
            x = col * sqrt3 - (3 * sqrt3) / 2
            y = row * 1.5 - 1.5
            if row % 2 == 1:
                x += sqrt3 / 2
            positions5.append((x, y))
            angles5.append(0.0)
    
    arrangements = [
        (positions1, angles1),
        (positions2, angles2),
        (positions3, angles3),
        (positions4, angles4),
        (positions5, angles5),
    ]
    
    best_result = None
    best_side = float('inf')
    
    for positions, angles in arrangements:
        if time.time() > deadline:
            break
        center, side, angle = evaluate_arrangement(positions, angles)
        if side < best_side:
            best_side = side
            best_result = (positions, angles, center, side, angle)
    
    if best_result is None:
        return positions1, (0.0, 0.0), 40.0, 0.0
    
    positions, angles, center, side, angle = best_result
    inner = [[x, y, a] for (x, y), a in zip(positions, angles)]
    return inner, center, side, angle
# EVOLVE-BLOCK-END

