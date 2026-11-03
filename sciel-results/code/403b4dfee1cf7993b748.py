import math
import random
import time
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Initial layout: Hexagonal cluster
    # Center
    centers = np.array([[0.0, 0.0]])
    angles = np.array([0.0])
    
    # 6 neighbors
    d = math.sqrt(3)
    for i in range(6):
        angle = i * math.pi / 3
        x = d * math.cos(angle)
        y = d * math.sin(angle)
        centers = np.vstack([centers, [[x, y]]])
        angles = np.append(angles, 0.0)
        
    # 4 more from the next ring
    # Axial (q,r) -> x = sqrt(3)*(q + r/2), y = 1.5*r
    # Let's pick 4 positions that are symmetric and fit well.
    # (2,0) -> (2*sqrt(3), 0)
    # (-2,0) -> (-2*sqrt(3), 0)
    # (1,1) -> (1.5*sqrt(3), 1.5)
    # (-1,-1) -> (-1.5*sqrt(3), -1.5)
    extra_points = [
        (2*math.sqrt(3), 0.0),
        (-2*math.sqrt(3), 0.0),
        (1.5*math.sqrt(3), 1.5),
        (-1.5*math.sqrt(3), -1.5)
    ]
    for p in extra_points:
        centers = np.vstack([centers, [p]])
        angles = np.append(angles, 0.0)
        
    # Now we have 11 hexagons.
    
    # Function to calculate required side length S for a given configuration
    # centers: (N, 2) array
    # angles: (N,) array
    # container_angle: scalar
    def calc_S(centers, angles, container_angle):
        # Vertices of unit hexagon relative to center
        # Angles: 0, 60, 120, 180, 240, 300 degrees
        v_angles = np.array([0, math.pi/3, 2*math.pi/3, math.pi, 4*math.pi/3, 5*math.pi/3])
        v_cos = np.cos(v_angles)
        v_sin = np.sin(v_angles)
        
        # Total vertices: N * 6
        # We can compute all vertex positions in one go using broadcasting
        # centers: (N, 1, 2)
        # angles: (N, 1)
        # v_cos: (1, 6)
        
        # Expand centers to (N, 1, 2)
        c_x = centers[:, 0:1]
        c_y = centers[:, 1:2]
        
        # Expand angles to (N, 1)
        a = angles[:, np.newaxis]
        
        # Vertex offsets
        # vx_offset = cos(a + v_angle)
        # vy_offset = sin(a + v_angle)
        # Using angle addition formulas:
        # cos(a + v) = cos(a)cos(v) - sin(a)sin(v)
        # sin(a + v) = sin(a)cos(v) + cos(a)sin(v)
        
        cos_a = np.cos(a)
        sin_a = np.sin(a)
        
        vx_offsets = cos_a * v_cos - sin_a * v_sin
        vy_offsets = sin_a * v_cos + cos_a * v_sin
        
        # Global vertex positions
        vx = c_x + vx_offsets # (N, 6)
        vy = c_y + vy_offsets # (N, 6)
        
        # Rotate by -container_angle
        cos_c = math.cos(container_angle)
        sin_c = math.sin(container_angle)
        
        rx = vx * cos_c + vy * sin_c
        ry = -vx * sin_c + vy * cos_c
        
        # Required S for point (rx, ry) in flat-top hexagon
        abs_rx = np.abs(rx)
        abs_ry = np.abs(ry)
        
        s1 = abs_rx
        s2 = 2.0 * abs_ry / math.sqrt(3)
        s3 = abs_rx + abs_ry / math.sqrt(3)
        
        s_req = np.maximum(s1, np.maximum(s2, s3))
        
        return np.max(s_req)

    # Initial evaluation
    current_centers = centers.copy()
    current_angles = angles.copy()
    current_container_angle = 0.0
    
    current_score = calc_S(current_centers, current_angles, current_container_angle)
    best_score = current_score
    best_state = (current_centers.copy(), current_angles.copy(), current_container_angle)
    
    # Local search
    random.seed(42)
    step_size_pos = 0.5
    step_size_rot = 0.2
    step_size_cont = 0.1
    
    # Simulated Annealing / Random Restart style
    # We will do a simple hill climbing with occasional random jumps
    while time.time() < deadline:
        # Try moving a random inner hexagon
        i = random.randint(0, N-1)
        old_cx, old_cy = current_centers[i]
        dx = (random.random() - 0.5) * step_size_pos
        dy = (random.random() - 0.5) * step_size_pos
        current_centers[i, 0] += dx
        current_centers[i, 1] += dy
        
        new_score = calc_S(current_centers, current_angles, current_container_angle)
        if new_score < best_score:
            best_score = new_score
            best_state = (current_centers.copy(), current_angles.copy(), current_container_angle)
        else:
            current_centers[i, 0] = old_cx
            current_centers[i, 1] = old_cy
            
        # Try rotating a random inner hexagon
        i = random.randint(0, N-1)
        old_a = current_angles[i]
        da = (random.random() - 0.5) * step_size_rot
        current_angles[i] += da
        
        new_score = calc_S(current_centers, current_angles, current_container_angle)
        if new_score < best_score:
            best_score = new_score
            best_state = (current_centers.copy(), current_angles.copy(), current_container_angle)
        else:
            current_angles[i] = old_a
            
        # Try rotating the container
        old_ca = current_container_angle
        dca = (random.random() - 0.5) * step_size_cont
        current_container_angle += dca
        
        new_score = calc_S(current_centers, current_angles, current_container_angle)
        if new_score < best_score:
            best_score = new_score
            best_state = (current_centers.copy(), current_angles.copy(), current_container_angle)
        else:
            current_container_angle = old_ca
            
        # Reduce step size occasionally
        if random.random() < 0.02:
            step_size_pos *= 0.95
            step_size_rot *= 0.95
            step_size_cont *= 0.95

    final_centers, final_angles, final_container_angle = best_state
    
    # Format output
    inner = []
    for i in range(N):
        cx, cy = final_centers[i]
        a_deg = math.degrees(final_angles[i])
        inner.append([float(cx), float(cy), float(a_deg)])
        
    return inner, (0.0, 0.0), float(best_score), float(math.degrees(final_container_angle))

# EVOLVE-BLOCK-END
