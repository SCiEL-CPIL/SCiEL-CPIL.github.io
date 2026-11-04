import numpy as np
import time
import random

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Hexagon geometry
    SQRT3 = np.sqrt(3)
    
    def hex_vertices(cx, cy, angle_deg):
        """Generate vertices of a unit hexagon centered at (cx, cy) with rotation angle_deg."""
        angle_rad = np.deg2rad(angle_deg)
        # Vertices of a unit hexagon (side 1) centered at origin, flat-top
        # Radius (center to vertex) is 1.
        # Angles: 0, 60, 120, 180, 240, 300
        base_angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]))
        verts = np.column_stack([np.cos(base_angles), np.sin(base_angles)])
        # Rotate
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        R = np.array([[c, -s], [s, c]])
        verts = verts @ R.T
        # Translate
        verts += np.array([cx, cy])
        return verts

    def calc_side_length(positions, angles):
        """Calculate the minimal side length of the outer hexagon to contain all inner hexagons."""
        max_s = 0.0
        for i in range(N):
            cx, cy = positions[i]
            ang = angles[i]
            verts = hex_vertices(cx, cy, ang)
            for vx, vy in verts:
                # Condition for point (vx, vy) to be inside regular hexagon of side S centered at 0
                # S >= |vx| + 1/sqrt(3) * |vy|
                s_req = abs(vx) + (1.0 / SQRT3) * abs(vy)
                if s_req > max_s:
                    max_s = s_req
        return max_s

    def check_overlap(positions, angles):
        """Check if any two hexagons overlap. Returns True if valid (no overlap)."""
        # Minimum distance between centers of two non-overlapping unit hexagons is sqrt(3)
        # This is true if they are aligned. If rotated, it's still at least sqrt(3) for regular hexagons?
        # Actually, the minimum distance between centers of two disjoint regular hexagons of side 1 is sqrt(3).
        # Proof: The inradius is sqrt(3)/2. Distance >= inradius1 + inradius2 = sqrt(3).
        for i in range(N):
            for j in range(i + 1, N):
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < SQRT3 - 1e-9:
                    return False
        return True

    # Initial configuration: 3x3 grid + 2
    # Axial to Cartesian: x = sqrt(3)*(q + r/2), y = 1.5*r
    def axial_to_cart(q, r):
        return np.array([SQRT3 * (q + r / 2.0), 1.5 * r])

    # 3x3 block centered at 0
    coords = []
    for q in [-1, 0, 1]:
        for r in [-1, 0, 1]:
            coords.append(axial_to_cart(q, r))
    # Add 2 more
    coords.append(axial_to_cart(2, 0))
    coords.append(axial_to_cart(-2, 0))
    
    positions = np.array(coords)
    angles = np.zeros(N)
    
    # Center the cluster
    center = np.mean(positions, axis=0)
    positions -= center
    
    best_s = calc_side_length(positions, angles)
    best_positions = positions.copy()
    best_angles = angles.copy()
    
    # Local Search
    step_size = 0.5
    no_improve_count = 0
    
    while time.time() < deadline:
        if no_improve_count > 200:
            step_size *= 0.9
            no_improve_count = 0
            if step_size < 0.01:
                break
        
        # Randomly perturb a few hexagons
        new_positions = best_positions.copy()
        new_angles = best_angles.copy()
        
        num_perturb = random.randint(1, 3)
        indices = random.sample(range(N), num_perturb)
        
        for idx in indices:
            # Translate
            dx = random.uniform(-step_size, step_size)
            dy = random.uniform(-step_size, step_size)
            new_positions[idx][0] += dx
            new_positions[idx][1] += dy
            
            # Rotate (small chance)
            if random.random() < 0.1:
                new_angles[idx] += random.uniform(-10, 10)
        
        # Check overlap
        if check_overlap(new_positions, new_angles):
            s_new = calc_side_length(new_positions, new_angles)
            if s_new < best_s:
                best_s = s_new
                best_positions = new_positions
                best_angles = new_angles
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1

    # Convert to list format
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i][0]), float(best_positions[i][1]), float(best_angles[i])])
    
    # Outer hexagon center is (0,0)
    # Side length is best_s
    # Rotation is 0.0
    return inner, (0.0, 0.0), float(best_s), 0.0
# EVOLVE-BLOCK-END
