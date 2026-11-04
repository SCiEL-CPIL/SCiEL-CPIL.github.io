import math

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Generate centers for a 3x4 grid of hexagons
    # i is the column index, j is the row index
    # 3 columns, 4 rows
    centers = []
    # To center the grid, we use offsets
    # i in {-1, 0, 1}
    # j in {-1.5, -0.5, 0.5, 1.5}
    i_vals = [-1, 0, 1]
    j_vals = [-1.5, -0.5, 0.5, 1.5]
    
    sqrt3 = math.sqrt(3)
    
    for i in i_vals:
        for j in j_vals:
            x = sqrt3 * (i + j / 2.0)
            y = 1.5 * j
            centers.append([x, y])
            
    # Center the cluster at (0,0)
    # Calculate mean
    mean_x = sum(c[0] for c in centers) / len(centers)
    mean_y = sum(c[1] for c in centers) / len(centers)
    
    centered_centers = []
    for c in centers:
        centered_centers.append([c[0] - mean_x, c[1] - mean_y])
        
    # Calculate the minimum side length S of the enclosing regular hexagon
    # The enclosing hexagon is centered at (0,0) with angle 0 (flat top)
    # Conditions for a point (x,y) to be inside a regular hexagon of side S (flat top):
    # 1. |x| <= S
    # 2. |y| <= S * sqrt(3) / 2  => S >= 2*|y|/sqrt(3)
    # 3. |x + y/sqrt(3)| <= S
    # 4. |-x + y/sqrt(3)| <= S
    
    max_s = 0.0
    for cx, cy in centered_centers:
        # Generate 6 vertices of the unit hexagon centered at (cx, cy)
        # Vertices are at angles 0, 60, 120, 180, 240, 300 degrees
        for k in range(6):
            angle = math.pi / 3.0 * k
            vx = cx + math.cos(angle)
            vy = cy + math.sin(angle)
            
            s1 = abs(vx)
            s2 = 2.0 * abs(vy) / sqrt3
            s3 = abs(vx + vy / sqrt3)
            s4 = abs(-vx + vy / sqrt3)
            
            s_req = max(s1, s2, s3, s4)
            if s_req > max_s:
                max_s = s_req
                
    # Add a small margin to ensure validity against floating point precision
    side_length = max_s + 1e-6
    
    # Format inner hexagons: (x, y, angle_degrees)
    # All inner hexagons have angle 0
    inner = []
    for cx, cy in centered_centers:
        inner.append([cx, cy, 0.0])
        
    return inner, (0.0, 0.0), side_length, 0.0
# EVOLVE-BLOCK-END
