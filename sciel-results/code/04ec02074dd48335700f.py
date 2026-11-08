import math
import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Use the compact configuration from the failed program but with a small
    # scaling factor to ensure strict disjointness (distance > sqrt(3)).
    # The failed program had touching hexagons.
    k = 1.001
    
    sqrt3 = math.sqrt(3)
    
    inner_centers = []
    
    # 1. Center
    inner_centers.append((0.0, 0.0))
    
    # 2-7. First shell
    # Original dist: sqrt(3)
    for i in range(6):
        angle = math.radians(30 + i * 60)
        x = k * sqrt3 * math.cos(angle)
        y = k * sqrt3 * math.sin(angle)
        inner_centers.append((x, y))
        
    # 8-12. Second shell (5 hexagons)
    # Original dist: 3.0
    # Angles: 0, 60, 120, 180, 240
    good_angles = [0, 60, 120, 180, 240]
    for deg in good_angles:
        angle = math.radians(deg)
        x = k * 3.0 * math.cos(angle)
        y = k * 3.0 * math.sin(angle)
        inner_centers.append((x, y))
        
    # Generate vertices for all inner hexagons
    all_vertices = []
    for cx, cy in inner_centers:
        for i in range(6):
            angle = math.radians(i * 60)
            vx = cx + math.cos(angle)
            vy = cy + math.sin(angle)
            all_vertices.append((vx, vy))
            
    all_vertices = np.array(all_vertices)
    
    # Find minimal enclosing regular hexagon
    # Center at (0,0)
    # Optimize rotation angle theta
    
    min_side = float('inf')
    best_theta = 0.0
    
    # Scan theta from 0 to 60 degrees
    for theta_deg in np.linspace(0, 60, 121):
        theta = math.radians(theta_deg)
        
        max_proj = 0.0
        for i in range(6):
            a = theta + math.radians(i * 60)
            nx = math.cos(a)
            ny = math.sin(a)
            
            projections = all_vertices[:, 0] * nx + all_vertices[:, 1] * ny
            current_max = np.max(np.abs(projections))
            if current_max > max_proj:
                max_proj = current_max
                
        required_side = (2.0 / sqrt3) * max_proj
        if required_side < min_side:
            min_side = required_side
            best_theta = theta_deg
            
    # Add epsilon for floating point safety
    side = min_side + 1e-6
    center = (0.0, 0.0)
    angle = best_theta
    
    # Construct inner list
    inner = []
    for cx, cy in inner_centers:
        inner.append([cx, cy, 0.0])
        
    return inner, center, side, angle
# EVOLVE-BLOCK-END
