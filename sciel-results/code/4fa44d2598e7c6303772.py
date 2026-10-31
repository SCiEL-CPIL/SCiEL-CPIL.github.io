import math
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Generate lattice points (q, r)
    # Estimate max coordinate
    # N ~ 3 * k^2 => k ~ sqrt(N/3)
    # Add a margin for non-magic numbers
    max_coord = int(math.sqrt(N)) + 5
    
    points = []
    for q in range(-max_coord, max_coord + 1):
        for r in range(-max_coord, max_coord + 1):
            # Axial to Cartesian
            # x = sqrt(3) * (q + r/2)
            # y = sqrt(3)/2 * r
            # Distance squared from origin (for sorting)
            # d2 = 3 * (q^2 + q*r + r^2)
            d2 = q*q + q*r + r*r
            points.append((d2, q, r))
            
    # Sort by distance from center
    points.sort(key=lambda x: x[0])
    
    # Take first N points
    selected = points[:N]
    
    # Calculate Cartesian coordinates
    coords = []
    for _, q, r in selected:
        x = math.sqrt(3) * (q + r / 2.0)
        y = (math.sqrt(3) / 2.0) * r
        coords.append((x, y))
        
    # Calculate required side length L
    # Directions
    # Vertex directions (0, 60, 120, 180, 240, 300)
    # Side normal directions (30, 90, 150, 210, 270, 330)
    
    max_vertex_proj = 0.0
    max_side_proj = 0.0
    
    # Precompute unit vectors
    v_vecs = []
    s_vecs = []
    for i in range(6):
        angle_v = i * math.pi / 3.0
        angle_s = (i + 0.5) * math.pi / 3.0
        v_vecs.append((math.cos(angle_v), math.sin(angle_v)))
        s_vecs.append((math.cos(angle_s), math.sin(angle_s)))
        
    for x, y in coords:
        for vx, vy in v_vecs:
            proj = x * vx + y * vy
            if proj > max_vertex_proj:
                max_vertex_proj = proj
        for sx, sy in s_vecs:
            proj = x * sx + y * sy
            if proj > max_side_proj:
                max_side_proj = proj
                
    # Constraints:
    # L >= max_vertex_proj + 1
    # L >= (2/sqrt(3)) * max_side_proj + 1
    
    L1 = max_vertex_proj + 1.0
    L2 = (2.0 / math.sqrt(3)) * max_side_proj + 1.0
    L = max(L1, L2)
    
    # Construct inner list: [x, y, angle]
    # Angle is 0 for all, as they are aligned
    inner = [[x, y, 0.0] for x, y in coords]
    
    # Outer center is (0,0), side L, angle 0
    return inner, (0.0, 0.0), L, 0.0
# EVOLVE-BLOCK-END
