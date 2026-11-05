import numpy as np
import time

N = 12

def _hexagon_vertices(center, side, angle_deg):
    """
    Returns vertices of a regular hexagon.
    center: (x, y)
    side: length
    angle_deg: rotation in degrees
    """
    cx, cy = center
    angle = np.deg2rad(angle_deg)
    # Vertices of unit hexagon (flat-top, side 1) relative to center
    # Angles: 0, 60, 120, 180, 240, 300
    angles = np.array([0, 60, 120, 180, 240, 300]) + angle_deg
    angles_rad = np.deg2rad(angles)
    # Distance from center to vertex is 'side'
    # Wait, for regular hexagon, side length = radius.
    # So vertices are at distance 'side' from center.
    xs = cx + side * np.cos(angles_rad)
    ys = cy + side * np.sin(angles_rad)
    return np.column_stack((xs, ys))

def _poly_overlap(p1, p2):
    """
    Check if two convex polygons overlap using SAT.
    p1, p2: Nx2 arrays of vertices.
    Returns True if overlap.
    """
    # Get edges of p1
    for i in range(len(p1)):
        v1 = p1[i]
        v2 = p1[(i + 1) % len(p1)]
        # Edge vector
        edge = v2 - v1
        # Normal
        normal = np.array([-edge[1], edge[0]])
        
        # Project p1 and p2 onto normal
        proj1 = np.dot(p1, normal)
        proj2 = np.dot(p2, normal)
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2 or max2 < min1:
            return False
            
    # Get edges of p2 (if p1 and p2 have same orientation, 6 axes might be enough, 
    # but they can rotate independently, so we must check p2's axes too)
    for i in range(len(p2)):
        v1 = p2[i]
        v2 = p2[(i + 1) % len(p2)]
        edge = v2 - v1
        normal = np.array([-edge[1], edge[0]])
        
        proj1 = np.dot(p1, normal)
        proj2 = np.dot(p2, normal)
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2 or max2 < min1:
            return False
            
    return True

def _calculate_required_S(positions, angles):
    """
    Calculate minimum S for outer hexagon (flat-top, centered at 0,0) 
    to contain all inner hexagons (side 1).
    """
    all_vertices = []
    for i in range(len(positions)):
        cx, cy = positions[i]
        verts = _hexagon_vertices((cx, cy), 1.0, angles[i])
        all_vertices.append(verts)
    
    all_vertices = np.vstack(all_vertices)
    
    # Normals for flat-top hexagon sides
    # Side normals point outwards
    # 0 deg: (1, 0) -> x <= S * sqrt(3)/2 ? No.
    # Distance from center to side is apothem = S * sqrt(3) / 2.
    # So u . v <= S * sqrt(3) / 2.
    # S >= 2/sqrt(3) * max(u . v)
    
    normals = np.array([
        [1, 0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1, 0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    
    # Project all vertices onto all normals
    # all_vertices: (M, 2)
    # normals: (6, 2)
    # We want max over all projections
    
    # Vectorized:
    # projections = np.dot(all_vertices, normals.T) # (M, 6)
    # max_proj = np.max(projections)
    # S = max_proj * 2 / np.sqrt(3)
    
    projections = np.dot(all_vertices, normals.T)
    max_proj = np.max(projections)
    S = max_proj * 2.0 / np.sqrt(3)
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Initialize with 2x6 grid
    # Pointy-top grid logic for initialization?
    # Let's use flat-top hexagons for inner as well, aligned.
    # Flat-top hex width (x) = 2, height (y) = sqrt(3).
    # Spacing x: sqrt(3) ? No.
    # If flat-top, neighbors in x are at distance sqrt(3)?
    # Vertex at (1,0). Next vertex (1/2, sqrt(3)/2).
    # Center to center distance for touching flat-top hexes:
    # Horizontal: 2 * (sqrt(3)/2) = sqrt(3).
    # Vertical: 2 * (1/2) = 1? No.
    # Let's just use a known good spacing.
    # For flat-top hexes:
    # dx = sqrt(3), dy = 1.5
    # Offset for odd rows: dx/2
    
    positions = []
    angles = []
    for row in range(2):
        for col in range(6):
            x = col * np.sqrt(3)
            y = row * 1.5
            if row % 2 == 1:
                x += np.sqrt(3) / 2
            positions.append([x, y])
            angles.append(0.0)
            
    # Center the configuration
    pos_arr = np.array(positions)
    center_offset = pos_arr.mean(axis=0)
    pos_arr -= center_offset
    positions = [list(p) for p in pos_arr]
    
    best_S = _calculate_required_S(positions, angles)
    best_positions = [list(p) for p in positions]
    best_angles = list(angles)
    
    # Local Search
    current_positions = [list(p) for p in positions]
    current_angles = list(angles)
    current_S = best_S
    
    step_size = 0.1
    num_iterations = 20000
    
    for i in range(num_iterations):
        if time.time() > deadline:
            break
            
        # Randomly select a hexagon to move
        idx = np.random.randint(N)
        
        # Perturb position
        dx = np.random.uniform(-step_size, step_size)
        dy = np.random.uniform(-step_size, step_size)
        
        # Perturb angle (small)
        d_angle = np.random.uniform(-5, 5)
        
        new_pos = [current_positions[idx][0] + dx, current_positions[idx][1] + dy]
        new_angle = current_angles[idx] + d_angle
        
        # Store old
        old_pos = current_positions[idx]
        old_angle = current_angles[idx]
        
        # Apply
        current_positions[idx] = new_pos
        current_angles[idx] = new_angle
        
        # Check collisions
        valid = True
        for j in range(N):
            if j == idx:
                continue
            p1 = _hexagon_vertices(current_positions[idx], 1.0, current_angles[idx])
            p2 = _hexagon_vertices(current_positions[j], 1.0, current_angles[j])
            if _poly_overlap(p1, p2):
                valid = False
                break
        
        if valid:
            new_S = _calculate_required_S(current_positions, current_angles)
            if new_S < current_S:
                current_S = new_S
                if new_S < best_S:
                    best_S = new_S
                    best_positions = [list(p) for p in current_positions]
                    best_angles = list(current_angles)
            else:
                # Revert if worse
                current_positions[idx] = old_pos
                current_angles[idx] = old_angle
        else:
            # Revert if collision
            current_positions[idx] = old_pos
            current_angles[idx] = old_angle
            
    # Return best found
    # Format: inner rows (x, y, angle_degrees)
    inner = []
    for i in range(N):
        inner.append([best_positions[i][0], best_positions[i][1], best_angles[i]])
        
    # Outer hexagon center (0,0), side best_S, angle 0
    return inner, (0.0, 0.0), best_S, 0.0

# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    import time

    N = 12

    def _hexagon_vertices(center, side, angle_deg):
        cx, cy = center
        angles = np.array([0, 60, 120, 180, 240, 300]) + angle_deg
        angles_rad = np.deg2rad(angles)
        xs = cx + side * np.cos(angles_rad)
        ys = cy + side * np.sin(angles_rad)
        return np.column_stack((xs, ys))

    def _poly_overlap(p1, p2):
        for i in range(len(p1)):
            v1 = p1[i]
            v2 = p1[(i + 1) % len(p1)]
            edge = v2 - v1
            normal = np.array([-edge[1], edge[0]])
            
            proj1 = np.dot(p1, normal)
            proj2 = np.dot(p2, normal)
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            if max1 < min2 or max2 < min1:
                return False
            
        for i in range(len(p2)):
            v1 = p2[i]
            v2 = p2[(i + 1) % len(p2)]
            edge = v2 - v1
            normal = np.array([-edge[1], edge[0]])
            
            proj1 = np.dot(p1, normal)
            proj2 = np.dot(p2, normal)
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            if max1 < min2 or max2 < min1:
                return False
                
        return True

    def _calculate_required_S(positions, angles):
        all_vertices = []
        for i in range(len(positions)):
            cx, cy = positions[i]
            verts = _hexagon_vertices((cx, cy), 1.0, angles[i])
            all_vertices.append(verts)
        
        all_vertices = np.vstack(all_vertices)
        
        normals = np.array([
            [1, 0],
            [0.5, np.sqrt(3)/2],
            [-0.5, np.sqrt(3)/2],
            [-1, 0],
            [-0.5, -np.sqrt(3)/2],
            [0.5, -np.sqrt(3)/2]
        ])
        
        projections = np.dot(all_vertices, normals.T)
        max_proj = np.max(projections)
        S = max_proj * 2.0 / np.sqrt(3)
        return S

    start_time = time.time()
    deadline = start_time + 110.0

    # Initialize with 2x6 grid
    positions = []
    angles = []
    for row in range(2):
        for col in range(6):
            x = col * np.sqrt(3)
            y = row * 1.5
            if row % 2 == 1:
                x += np.sqrt(3) / 2
            positions.append([x, y])
            angles.append(0.0)
            
    pos_arr = np.array(positions)
    center_offset = pos_arr.mean(axis=0)
    pos_arr -= center_offset
    positions = [list(p) for p in pos_arr]
    
    best_S = _calculate_required_S(positions, angles)
    best_positions = [list(p) for p in positions]
    best_angles = list(angles)
    
    current_positions = [list(p) for p in positions]
    current_angles = list(angles)
    current_S = best_S
    
    step_size = 0.1
    num_iterations = 20000
    
    for i in range(num_iterations):
        if time.time() > deadline:
            break
            
        idx = np.random.randint(N)
        
        dx = np.random.uniform(-step_size, step_size)
        dy = np.random.uniform(-step_size, step_size)
        d_angle = np.random.uniform(-5, 5)
        
        new_pos = [current_positions[idx][0] + dx, current_positions[idx][1] + dy]
        new_angle = current_angles[idx] + d_angle
        
        old_pos = current_positions[idx]
        old_angle = current_angles[idx]
        
        current_positions[idx] = new_pos
        current_angles[idx] = new_angle
        
        valid = True
        for j in range(N):
            if j == idx:
                continue
            p1 = _hexagon_vertices(current_positions[idx], 1.0, current_angles[idx])
            p2 = _hexagon_vertices(current_positions[j], 1.0, current_angles[j])
            if _poly_overlap(p1, p2):
                valid = False
                break
        
        if valid:
            new_S = _calculate_required_S(current_positions, current_angles)
            if new_S < current_S:
                current_S = new_S
                if new_S < best_S:
                    best_S = new_S
                    best_positions = [list(p) for p in current_positions]
                    best_angles = list(current_angles)
            else:
                current_positions[idx] = old_pos
                current_angles[idx] = old_angle
        else:
            current_positions[idx] = old_pos
            current_angles[idx] = old_angle
            
    inner = []
    for i in range(N):
        inner.append([best_positions[i][0], best_positions[i][1], best_angles[i]])
        
    return inner, (0.0, 0.0), best_S, 0.0
# EVOLVE-BLOCK-END
