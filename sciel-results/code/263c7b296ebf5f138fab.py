import math
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Define the 11 hexagons
    # 1 center, 6 ring, 4 gaps
    inner = []
    
    # Center
    inner.append([0.0, 0.0, 0.0])
    
    # 6 Ring (distance sqrt(3))
    d1 = math.sqrt(3)
    for k in range(6):
        angle_rad = k * math.pi / 3
        x = d1 * math.cos(angle_rad)
        y = d1 * math.sin(angle_rad)
        inner.append([x, y, 0.0])
        
    # 4 Gaps (distance 3, angles 30, 90, 150, 270)
    d2 = 3.0
    gap_angles_deg = [30, 90, 150, 270]
    for deg in gap_angles_deg:
        angle_rad = math.radians(deg)
        x = d2 * math.cos(angle_rad)
        y = d2 * math.sin(angle_rad)
        inner.append([x, y, 0.0])
        
    # Calculate minimal outer side length
    # Outer hexagon center (0,0), angle 0
    # Normals at 0, 60, 120, 180, 240, 300
    
    max_proj = 0.0
    for k in range(6):
        theta = k * math.pi / 3
        nx = math.cos(theta)
        ny = math.sin(theta)
        
        for hex_row in inner:
            cx, cy, rot_deg = hex_row
            rot_rad = math.radians(rot_deg)
            
            # Projection of center
            proj_c = cx * nx + cy * ny
            
            # Max extent of hexagon in direction theta
            # Local angle relative to hexagon's orientation
            local_angle = theta - rot_rad
            # Reduce to 0..pi/3
            local_angle = local_angle % (math.pi / 3)
            # Distance to nearest vertex (0 or pi/3)
            # If local_angle < pi/6, nearest is 0. Else nearest is pi/3.
            # Actually, support function is cos(delta) where delta is dist to nearest multiple of pi/3
            delta = min(local_angle, math.pi / 3 - local_angle)
            extent = math.cos(delta)
            
            total_proj = proj_c + extent
            if total_proj > max_proj:
                max_proj = total_proj
                
    # max_proj is the required inradius R_in
    # Side length S = R_in * 2 / sqrt(3)
    side_length = max_proj * 2.0 / math.sqrt(3)
    
    return inner, (0.0, 0.0), side_length, 0.0
# EVOLVE-BLOCK-END
