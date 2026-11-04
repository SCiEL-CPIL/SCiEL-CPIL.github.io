import numpy as np
import time

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    # Initial positions
    # Center
    centers = np.array([[0.0, 0.0]])
    angles = np.array([0.0])
    
    # Ring 1
    for i in range(6):
        theta = np.pi / 3 * i
        x = np.sqrt(3) * np.cos(theta)
        y = np.sqrt(3) * np.sin(theta)
        centers = np.vstack([centers, [[x, y]]])
        angles = np.append(angles, 0.0)
        
    # Ring 2 (4 hexes)
    # Positions at distance 3
    # (0, 3), (1.5*sqrt(3), 1.5), (1.5*sqrt(3), -1.5), (0, -3)
    # Let's pick these 4
    r2_pos = [
        [0.0, 3.0],
        [1.5 * np.sqrt(3), 1.5],
        [1.5 * np.sqrt(3), -1.5],
        [0.0, -3.0]
    ]
    for p in r2_pos:
        centers = np.vstack([centers, [p]])
        angles = np.append(angles, 0.0)
        
    # Total 11 hexes
    # Verify count
    assert len(centers) == N
    
    best_side = 40.0
    best_centers = centers.copy()
    best_angles = angles.copy()
    
    # Optimization loop
    # We will perturb positions and angles and check if they fit in a smaller hexagon
    # and don't overlap.
    
    def get_hex_vertices(center, angle):
        c = np.array(center)
        a = np.deg2rad(angle)
        # Unit hexagon vertices relative to center
        # Flat top: (1,0), (0.5, sqrt(3)/2), ...
        v_rel = np.array([
            [1.0, 0.0],
            [0.5, np.sqrt(3)/2],
            [-0.5, np.sqrt(3)/2],
            [-1.0, 0.0],
            [-0.5, -np.sqrt(3)/2],
            [0.5, -np.sqrt(3)/2]
        ])
        # Rotate
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        v_rot = v_rel @ R.T
        return c + v_rot
        
    def check_overlap(c1, a1, c2, a2):
        # Simple check: if distance < 1.0, they might overlap.
        # Min distance between centers of two disjoint unit hexagons is sqrt(3) ~ 1.732 if aligned.
        # If rotated, can be closer?
        # Actually, if they touch at a vertex, distance is 2.
        # If they touch at a side, distance is sqrt(3).
        # So if dist < sqrt(3) - epsilon, they overlap.
        d = np.linalg.norm(c1 - c2)
        if d < np.sqrt(3) - 1e-6:
            return True
        # More precise check:
        # Get vertices
        v1 = get_hex_vertices(c1, a1)
        v2 = get_hex_vertices(c2, a2)
        # Check if any vertex of 1 is inside 2 or vice versa
        # Or if edges intersect.
        # For speed, we can just use the distance check for now, 
        # but it might allow overlaps if they are rotated and "interlock".
        # However, unit hexagons are convex.
        # If distance between centers > 2, no overlap.
        # If distance < sqrt(3), likely overlap.
        # Let's use a conservative bound: 1.6
        return d < 1.6

    def get_min_side(centers, angles, outer_rotation=0.0):
        # Calculate min side S for outer hexagon centered at 0,0, rotated by outer_rotation
        # Outer hexagon vertices:
        # S * (cos(outer_rotation + k*pi/3), sin(outer_rotation + k*pi/3))
        # Sides are lines.
        # For each side, the normal is n_k.
        # The side is at distance S * cos(pi/6) = S * sqrt(3)/2 from center?
        # No, apothem is S * sqrt(3)/2.
        # So for each of the 6 normals, we need max dot(vertex, normal) <= S * sqrt(3)/2.
        # So S >= max(dot) / (sqrt(3)/2).
        
        # Normals for flat-top hexagon (vertices at 0, 60, ...):
        # Sides are perpendicular to angles 30, 90, 150, ...
        # If we rotate outer hexagon by phi, normals are at phi + 30, phi + 90, ...
        normals = []
        for k in range(6):
            theta = np.deg2rad(outer_rotation + 30 + 60 * k)
            normals.append(np.array([np.cos(theta), np.sin(theta)]))
            
        max_proj = 0.0
        for i in range(len(centers)):
            v = get_hex_vertices(centers[i], angles[i])
            for n in normals:
                projs = np.dot(v, n)
                max_proj = max(max_proj, np.max(projs))
                
        S = max_proj / (np.sqrt(3) / 2)
        return S

    # Local search
    step_size = 0.1
    current_side = get_min_side(centers, angles)
    best_side = current_side
    best_centers = centers.copy()
    best_angles = angles.copy()
    
    while time.time() < deadline:
        # Perturb
        new_centers = centers.copy()
        new_angles = angles.copy()
        
        # Move a random hexagon
        idx = np.random.randint(0, N)
        new_centers[idx] += np.random.normal(0, step_size, 2)
        new_angles[idx] += np.random.normal(0, step_size)
        
        # Check overlap
        overlap = False
        for i in range(N):
            for j in range(i+1, N):
                if check_overlap(new_centers[i], new_angles[i], new_centers[j], new_angles[j]):
                    overlap = True
                    break
            if overlap: break
            
        if overlap:
            continue
            
        # Check side
        new_side = get_min_side(new_centers, new_angles)
        
        if new_side < best_side:
            best_side = new_side
            best_centers = new_centers.copy()
            best_angles = new_angles.copy()
            # Accept
            centers = new_centers
            angles = new_angles
            # Reduce step size
            step_size *= 0.95
        else:
            # Maybe accept with some probability?
            # Or just reject.
            # Let's try to reduce step size slightly if no improvement
            step_size *= 0.99

    return [[c[0], c[1], a] for c, a in zip(best_centers, best_angles)], (0.0, 0.0), best_side, 0.0

