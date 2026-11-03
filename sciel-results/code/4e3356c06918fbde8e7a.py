import math
import time
import random
import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145.0  # 150s limit, keep 5s margin
    
    # Helper functions
    def get_hex_vertices(cx, cy, angle_deg):
        """Get vertices of a unit regular hexagon."""
        angle = math.radians(angle_deg)
        verts = []
        for i in range(6):
            theta = angle + i * math.pi / 3
            x = cx + math.cos(theta)
            y = cy + math.sin(theta)
            verts.append((x, y))
        return verts

    R_in_unit = math.sqrt(3)/2

    def point_in_hex_simple(px, py, cx, cy, angle_deg):
        angle_rad = math.radians(angle_deg)
        for k in range(6):
            n_theta = angle_rad + math.pi/6 + k * math.pi/3
            nx = math.cos(n_theta)
            ny = math.sin(n_theta)
            dx = px - cx
            dy = py - cy
            if dx * nx + dy * ny > R_in_unit + 1e-9:
                return False
        return True

    def segments_intersect(p1, p2, p3, p4):
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

    def check_overlap(hex1, hex2):
        """Check if two unit hexagons overlap. hex is (cx, cy, angle)."""
        cx1, cy1, a1 = hex1
        cx2, cy2, a2 = hex2
        dx = cx2 - cx1
        dy = cy2 - cy1
        d2 = dx*dx + dy*dy
        # If distance is large, no overlap
        if d2 > 4.0: # (2*1)^2
            return False
        # If distance is small, likely overlap
        if d2 < 3.0 - 1e-9:
            return True
        
        # If sqrt(3) <= d <= 2, we need to check vertices/edges.
        v1 = get_hex_vertices(cx1, cy1, a1)
        v2 = get_hex_vertices(cx2, cy2, a2)
        
        for vx, vy in v1:
            if point_in_hex_simple(vx, vy, cx2, cy2, a2):
                return True
        for vx, vy in v2:
            if point_in_hex_simple(vx, vy, cx1, cy1, a1):
                return True
        
        for i in range(6):
            p1 = v1[i]
            p2 = v1[(i+1)%6]
            for j in range(6):
                p3 = v2[j]
                p4 = v2[(j+1)%6]
                if segments_intersect(p1, p2, p3, p4):
                    return True
        return False

    def calculate_min_side(hexagons, outer_center, outer_angle):
        """Calculate the minimum side length of outer hexagon to contain all inner hexagons."""
        cx, cy = outer_center
        angle = math.radians(outer_angle)
        max_dist = 0
        # Normals for outer hexagon
        normals = []
        for k in range(6):
            n_theta = angle + math.pi/6 + k * math.pi/3
            normals.append((math.cos(n_theta), math.sin(n_theta)))
        
        for hex in hexagons:
            hx, hy, ha = hex
            verts = get_hex_vertices(hx, hy, ha)
            for vx, vy in verts:
                dx = vx - cx
                dy = vy - cy
                for nx, ny in normals:
                    dist = abs(dx * nx + dy * ny)
                    if dist > max_dist:
                        max_dist = dist
        
        # max_dist is the required inradius R_in
        # Side length S = 2 * R_in / sqrt(3)
        return 2 * max_dist / math.sqrt(3)

    def is_valid(hexagons):
        for i in range(N):
            for j in range(i+1, N):
                if check_overlap(hexagons[i], hexagons[j]):
                    return False
        return True

    # Initial Layouts
    def init_hex_cluster():
        hexes = []
        hexes.append((0.0, 0.0, 0.0))
        for i in range(6):
            angle = i * 60.0
            rad = math.radians(angle)
            x = math.sqrt(3) * math.cos(rad)
            y = math.sqrt(3) * math.sin(rad)
            hexes.append((x, y, 0.0))
        for i in range(5):
            angle = i * 60.0
            rad = math.radians(angle)
            x = 2 * math.sqrt(3) * math.cos(rad)
            y = 2 * math.sqrt(3) * math.sin(rad)
            hexes.append((x, y, 0.0))
        return hexes

    def init_grid():
        hexes = []
        for r in range(3):
            for c in range(4):
                x = c * math.sqrt(3) + (r % 2) * math.sqrt(3)/2
                y = r * 1.5
                hexes.append((x, y, 0.0))
        cx = sum(h[0] for h in hexes) / N
        cy = sum(h[1] for h in hexes) / N
        hexes = [(h[0]-cx, h[1]-cy, h[2]) for h in hexes]
        return hexes

    best_hexes = init_hex_cluster()
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
    best_side = calculate_min_side(best_hexes, best_outer_center, best_outer_angle)
    
    # Local Search
    current_hexes = [list(h) for h in best_hexes]
    current_outer_center = [0.0, 0.0]
    current_outer_angle = 0.0
    current_side = best_side
    
    step_size = 0.1
    
    while time.time() < deadline:
        new_hexes = [list(h) for h in current_hexes]
        new_center = list(current_outer_center)
        new_angle = current_outer_angle
        
        for _ in range(3):
            idx = random.randint(0, N-1)
            new_hexes[idx][0] += random.uniform(-step_size, step_size)
            new_hexes[idx][1] += random.uniform(-step_size, step_size)
            new_hexes[idx][2] += random.uniform(-10, 10)
            
            new_center[0] += random.uniform(-step_size, step_size)
            new_center[1] += random.uniform(-step_size, step_size)
            new_angle += random.uniform(-5, 5)
        
        new_hexes_tuples = [tuple(h) for h in new_hexes]
        
        if not is_valid(new_hexes_tuples):
            continue
            
        new_side = calculate_min_side(new_hexes_tuples, tuple(new_center), new_angle)
        
        if new_side < current_side:
            current_hexes = new_hexes
            current_outer_center = new_center
            current_outer_angle = new_angle
            current_side = new_side
            
            if current_side < best_side:
                best_hexes = [tuple(h) for h in current_hexes]
                best_outer_center = tuple(current_outer_center)
                best_outer_angle = current_outer_angle
                best_side = current_side
                
            step_size *= 0.995
            
    inner = [[h[0], h[1], h[2]] for h in best_hexes]
    return inner, best_outer_center, best_side, best_outer_angle

# EVOLVE-BLOCK-END
