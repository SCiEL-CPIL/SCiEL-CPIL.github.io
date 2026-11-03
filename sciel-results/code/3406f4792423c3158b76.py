import math
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Define centers of 11 hexagons in a honeycomb pattern
    sqrt3 = math.sqrt(3)
    centers = [
        (0.0, 0.0),
        (sqrt3, 0.0),
        (-sqrt3, 0.0),
        (sqrt3 / 2.0, 1.5),
        (-sqrt3 / 2.0, 1.5),
        (sqrt3 / 2.0, -1.5),
        (-sqrt3 / 2.0, -1.5),
        (2.0 * sqrt3, 0.0),
        (3.0 * sqrt3 / 2.0, 1.5),
        (3.0 * sqrt3 / 2.0, -1.5),
        (0.0, 3.0)
    ]
    
    # Generate vertices for each inner hexagon (side length 1, angle 0)
    all_vertices = []
    for cx, cy in centers:
        for i in range(6):
            angle = math.pi / 3 * i
            vx = cx + math.cos(angle)
            vy = cy + math.sin(angle)
            all_vertices.append((vx, vy))
            
    # Function to calculate minimum side length S for a given outer hexagon angle
    def min_side_length_for_angle(angle_deg):
        angle_rad = math.radians(angle_deg)
        # Rotate points by -angle_rad to align outer hexagon with axes
        # Then check constraints for axis-aligned hexagon
        max_s = 0.0
        for x, y in all_vertices:
            # Rotate point
            xr = x * math.cos(-angle_rad) - y * math.sin(-angle_rad)
            yr = x * math.sin(-angle_rad) + y * math.cos(-angle_rad)
            
            # Constraints for axis-aligned hexagon of side S:
            # 1. |xr| <= S
            # 2. |xr + sqrt(3)*yr| <= 2*S
            # 3. |-xr + sqrt(3)*yr| <= 2*S
            
            s1 = abs(xr)
            s2 = abs(xr + sqrt3 * yr) / 2.0
            s3 = abs(-xr + sqrt3 * yr) / 2.0
            
            current_s = max(s1, s2, s3)
            if current_s > max_s:
                max_s = current_s
        return max_s

    # Find optimal angle
    best_s = float('inf')
    best_angle = 0.0
    
    # Sweep angles
    for i in range(600):
        angle = i * 0.1
        s = min_side_length_for_angle(angle)
        if s < best_s:
            best_s = s
            best_angle = angle
            
    # Refine around best angle
    for i in range(100):
        angle = best_angle - 0.05 + i * 0.001
        s = min_side_length_for_angle(angle)
        if s < best_s:
            best_s = s
            best_angle = angle

    # Construct inner list
    inner = []
    for cx, cy in centers:
        inner.append([cx, cy, 0.0])
        
    # Outer hexagon
    outer_center = (0.0, 0.0)
    outer_side = best_s
    outer_angle = best_angle
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
