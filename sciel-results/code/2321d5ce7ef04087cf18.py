import numpy as np
import time
import random

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def polygons_disjoint_sat(p1, p2, eps=1e-6):
        """
        Returns True if two convex polygons are disjoint (allowing touching with eps tolerance).
        Uses Separating Axis Theorem (SAT).
        Optimized for hexagons (6 vertices).
        """
        # Pre-compute normals for hexagons? No, angles vary.
        # For hexagons, the normals are at 0, 60, 120, 180, 240, 300 degrees relative to the hex orientation.
        # Actually, SAT requires checking edges of both polygons.
        
        # Optimization: Instead of full SAT, we can use the property of hexagons.
        # But let's stick to a robust SAT with early exit.
        
        # To speed up, we can check bounding boxes first?
        # min/max x and y of p1 vs p2.
        if p1[:,0].max() < p2[:,0].min() - eps or p2[:,0].max() < p1[:,0].min() - eps:
            return True
        if p1[:,1].max() < p2[:,1].min() - eps or p2[:,1].max() < p1[:,1].min() - eps:
            return True

        # Check axes from p1
        for i in range(6):
            j = (i + 1) % 6
            edge = p1[j] - p1[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-10: continue
            normal /= norm
            
            proj1 = p1.dot(normal)
            proj2 = p2.dot(normal)
            
            if proj1.max() < proj2.min() - eps or proj2.max() < proj1.min() - eps:
                return True
        
        # Check axes from p2
        for i in range(6):
            j = (i + 1) % 6
            edge = p2[j] - p2[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-10: continue
            normal /= norm
            
            proj1 = p1.dot(normal)
            proj2 = p2.dot(normal)
            
            if proj1.max() < proj2.min() - eps or proj2.max() < proj1.min() - eps:
                return True
                
        return False

    def check_validity(positions, angles):
        """
        positions: (N, 2) array
        angles: (N,) array
        Returns True if all are pairwise disjoint.
        """
        vertices_list = []
        for i in range(N):
            x, y = positions[i]
            a = angles[i]
            verts = hexagon_vertices(x, y, 1.0, a)
            vertices_list.append(verts)
        
        for i in range(N):
            for j in range(i + 1, N):
                if not polygons_disjoint_sat(vertices_list[i], vertices_list[j]):
                    return False
        return True

    def calculate_score(positions, angles):
        """
        Calculates the minimal side length S of an outer regular hexagon (centered at 0,0, angle 0)
        that contains all inner hexagons.
        """
        s_max = 0.0
        sqrt3 = np.sqrt(3.0)
        
        for i in range(N):
            x, y = positions[i]
            a = angles[i]
            verts = hexagon_vertices(x, y, 1.0, a)
            for vx, vy in verts:
                # Condition for point (vx, vy) to be in hexagon of side S (angle 0):
                # |vx| <= S
                # |vy| * 2/sqrt(3) <= S  => |vy| <= S * sqrt(3)/2
                # |vx| + |vy|/sqrt(3) <= S
                
                s_req = max(abs(vx), abs(vy) * 2.0 / sqrt3, abs(vx) + abs(vy) / sqrt3)
                if s_req > s_max:
                    s_max = s_req
        return s_max

    # --- Initialization ---
    # Start with a compact hexagonal lattice cluster of 12 hexagons.
    # This is a valid configuration.
    # Axial coordinates for a cluster of 12:
    # Ring 0: (0,0)
    # Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Ring 2 partial: (2,0), (1,1), (0,2), (-1,2), (-2,1)
    
    coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
    ]
    
    init_positions = np.zeros((N, 2))
    init_angles = np.zeros(N)
    
    for i, (q, r) in enumerate(coords):
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        init_positions[i] = [x, y]
        
    # Verify init
    if not check_validity(init_positions, init_angles):
        # Fallback: spread out
        for i in range(N):
            init_positions[i] = [i * 2.0, 0.0]
            init_angles[i] = 0.0

    current_pos = init_positions.copy()
    current_ang = init_angles.copy()
    current_score = calculate_score(current_pos, current_ang)
    
    best_pos = current_pos.copy()
    best_ang = current_ang.copy()
    best_score = current_score
    
    # --- Simulated Annealing ---
    # Parameters
    T = 0.5
    T_min = 0.001
    cooling_rate = 0.9995
    
    # We will perform steps until deadline
    # Each step: pick a random hexagon, perturb its position and/or angle slightly.
    
    while time.time() < deadline:
        if T < T_min:
            break
            
        # Pick random index
        idx = random.randint(0, N - 1)
        
        # Save old state
        old_x, old_y = current_pos[idx]
        old_a = current_ang[idx]
        
        # Perturbation
        # Move distance scales with temperature
        step_size = T * 0.5
        delta_x = random.gauss(0, step_size)
        delta_y = random.gauss(0, step_size)
        delta_a = random.gauss(0, step_size)
        
        new_x = old_x + delta_x
        new_y = old_y + delta_y
        new_a = old_a + delta_a
        
        # Apply
        current_pos[idx] = [new_x, new_y]
        current_ang[idx] = new_a
        
        # Check validity
        if check_validity(current_pos, current_ang):
            new_score = calculate_score(current_pos, current_ang)
            
            # Accept if better, or with probability exp(-(new_score - current_score)/T)
            if new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_pos = current_pos.copy()
                    best_ang = current_ang.copy()
            else:
                # Worse
                if random.random() < np.exp(-(new_score - current_score) / T):
                    current_score = new_score
                else:
                    # Reject
                    current_pos[idx] = [old_x, old_y]
                    current_ang[idx] = old_a
        else:
            # Invalid, reject
            current_pos[idx] = [old_x, old_y]
            current_ang[idx] = old_a
            
        # Cool down
        T *= cooling_rate
        
        # Periodic restart to escape local minima?
        # If we are stuck (no improvement for a long time), we might want to reheat or perturb more.
        # For simplicity, we just let it cool.
        
    # Final refinement: try to optimize outer hexagon rotation?
    # The problem allows outer rotation.
    # calculate_score assumes outer angle 0.
    # We can try a few outer angles and see if it reduces the required side length.
    # If we rotate the outer hexagon by phi, the condition for point (x,y) changes.
    # Actually, it's easier to rotate the inner points by -phi and calculate score for angle 0.
    # Let's try a few outer angles.
    
    best_final_score = best_score
    best_outer_angle = 0.0
    
    # Try rotating the whole configuration relative to the outer hex
    # This is equivalent to rotating the outer hexagon.
    for phi in np.linspace(0, np.pi/6, 7): # 0 to 30 degrees
        # Rotate inner points by -phi
        c = np.cos(-phi)
        s = np.sin(-phi)
        rot_mat = np.array([[c, -s], [s, c]])
        
        rotated_pos = best_pos @ rot_mat.T
        
        # Calculate score for this rotated config (outer angle 0)
        s_val = calculate_score(rotated_pos, best_ang)
        
        if s_val < best_final_score:
            best_final_score = s_val
            best_outer_angle = phi
            
    # Construct return
    inner = []
    for i in range(N):
        x, y = best_pos[i]
        a = best_ang[i]
        inner.append([float(x), float(y), float(a)])
        
    # Outer hexagon center (0,0), side length, rotation
    # Note: The score is the side length.
    return inner, (0.0, 0.0), float(best_final_score), float(best_outer_angle)

# EVOLVE-BLOCK-END
