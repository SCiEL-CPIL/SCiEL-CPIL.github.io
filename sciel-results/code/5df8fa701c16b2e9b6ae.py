import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Centers of the 11 hexagons
    # 1 center
    centers = [[0.0, 0.0]]
    
    # 6 neighbors
    for i in range(6):
        angle = np.deg2rad(60 * i)
        centers.append([np.sqrt(3) * np.cos(angle), np.sqrt(3) * np.sin(angle)])
        
    # 4 more at r=3, angles 30, 150, 210, 330
    for angle_deg in [30, 150, 210, 330]:
        angle = np.deg2rad(angle_deg)
        centers.append([3.0 * np.cos(angle), 3.0 * np.sin(angle)])
        
    # All inner hexagons have 0 rotation
    inner = [[c[0], c[1], 0.0] for c in centers]
    
    # Calculate minimal outer side S
    # Outer hexagon centered at (0,0), 0 rotation (vertex on x-axis)
    # Inequalities:
    # sqrt(3)x + y <= sqrt(3)S
    # -sqrt(3)x + y <= sqrt(3)S
    # y <= sqrt(3)/2 * S
    # -sqrt(3)x - y <= sqrt(3)S
    # sqrt(3)x - y <= sqrt(3)S
    # -y <= sqrt(3)/2 * S
    
    max_s = 0.0
    for c in centers:
        cx, cy = c
        # Vertices of this inner hexagon
        verts = [
            (cx + 1.0, cy),
            (cx + 0.5, cy + np.sqrt(3)/2),
            (cx - 0.5, cy + np.sqrt(3)/2),
            (cx - 1.0, cy),
            (cx - 0.5, cy - np.sqrt(3)/2),
            (cx + 0.5, cy - np.sqrt(3)/2)
        ]
        for vx, vy in verts:
            s1 = (np.sqrt(3)*vx + vy) / np.sqrt(3)
            s2 = (-np.sqrt(3)*vx + vy) / np.sqrt(3)
            s3 = vy / (np.sqrt(3)/2)
            s4 = (-np.sqrt(3)*vx - vy) / np.sqrt(3)
            s5 = (np.sqrt(3)*vx - vy) / np.sqrt(3)
            s6 = -vy / (np.sqrt(3)/2)
            
            current_s = max(s1, s2, s3, s4, s5, s6)
            if current_s > max_s:
                max_s = current_s
                
    # Add small epsilon to ensure strict containment if needed, or just use max_s
    side = max_s
    center = (0.0, 0.0)
    angle_outer = 0.0
    
    return inner, center, side, angle_outer
# EVOLVE-BLOCK-END
