import numpy as np
import math
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    # Initial configuration
    def hex_to_cart(q, r):
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return x, y

    coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (2, -1), (1, -2), (0, -2)
    ]
    
    positions = []
    for q, r in coords:
        x, y = hex_to_cart(q, r)
        positions.append([x, y, 0.0])
        
    def update_vertices(pos_list, verts_list):
        for i in range(len(pos_list)):
            x, y, theta = pos_list[i]
            v = []
            for k in range(6):
                angle = theta + k * math.pi / 3.0
                vx = x + math.cos(angle)
                vy = y + math.sin(angle)
                v.append((vx, vy))
            verts_list[i] = v

    verts_list = [None] * N
    update_vertices(positions, verts_list)

    def polygons_overlap(v1, v2):
        for j in range(6):
            p1 = v1[j]
            p2 = v1[(j + 1) % 6]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            nx = -dy
            ny = dx
            min1, max1 = float('inf'), float('-inf')
            min2, max2 = float('inf'), float('-inf')
            for p in v1:
                proj = p[0] * nx + p[1] * ny
                if proj < min1: min1 = proj
                if proj > max1: max1 = proj
            for p in v2:
                proj = p[0] * nx + p[1] * ny
                if proj < min2: min2 = proj
                if proj > max2: max2 = proj
            if max1 < min2 or max2 < min1:
                return False
        # Check normals of v2 as well? 
        # For convex polygons, checking normals of both is sufficient.
        # But if they are identical shape, normals are same set (rotated).
        # Actually, for two convex polygons, checking all normals of both is required.
        # However, if we check all 6 of v1 and all 6 of v2, it's 12 axes.
        # The loop above only checks v1's normals.
        # We need to check v2's normals too.
        for j in range(6):
            p1 = v2[j]
            p2 = v2[(j + 1) % 6]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            nx = -dy
            ny = dx
            min1, max1 = float('inf'), float('-inf')
            min2, max2 = float('inf'), float('-inf')
            for p in v1:
                proj = p[0] * nx + p[1] * ny
                if proj < min1: min1 = proj
                if proj > max1: max1 = proj
            for p in v2:
                proj = p[0] * nx + p[1] * ny
                if proj < min2: min2 = proj
                if proj > max2: max2 = proj
            if max1 < min2 or max2 < min1:
                return False
        return True

    def calculate_required_side(verts_list):
        max_r = 0.0
        for v in verts_list:
            for vx, vy in v:
                d = 0.0
                # Check 3 axes is enough due to symmetry? 
                # Normals at 0, 60, 120.
                # |dot| handles the opposite side.
                for k in range(3):
                    angle = k * math.pi / 3.0
                    nx = math.cos(angle)
                    ny = math.sin(angle)
                    dot = abs(vx * nx + vy * ny)
                    if dot > d:
                        d = dot
                s_val = (2.0 / math.sqrt(3.0)) * d
                if s_val > max_r:
                    max_r = s_val
        return max_r

    current_side = calculate_required_side(verts_list)
    
    # Optimization
    while time.time() - start_time < time_limit:
        i = np.random.randint(0, N)
        old_pos = positions[i][:]
        old_verts = verts_list[i][:]
        
        new_pos = positions[i][:]
        new_pos[0] += np.random.normal(0, 0.5)
        new_pos[1] += np.random.normal(0, 0.5)
        new_pos[2] += np.random.normal(0, 0.1)
        
        # Update vertices for i
        x, y, theta = new_pos
        new_verts = []
        for k in range(6):
            angle = theta + k * math.pi / 3.0
            vx = x + math.cos(angle)
            vy = y + math.sin(angle)
            new_verts.append((vx, vy))
            
        # Check overlap
        overlaps = False
        for j in range(N):
            if i == j:
                continue
            if polygons_overlap(new_verts, verts_list[j]):
                overlaps = True
                break
        
        if not overlaps:
            # Update verts_list temporarily
            verts_list[i] = new_verts
            new_side = calculate_required_side(verts_list)
            
            if new_side < current_side:
                current_side = new_side
                positions[i] = new_pos
                # verts_list[i] is already updated
            else:
                # Revert
                verts_list[i] = old_verts
                positions[i] = old_pos
        else:
            # Revert
            verts_list[i] = old_verts
            positions[i] = old_pos

    return positions, (0.0, 0.0), current_side, 0.0
# EVOLVE-BLOCK-END
