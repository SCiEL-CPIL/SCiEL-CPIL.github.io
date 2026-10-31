import math
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    dx = math.sqrt(3)
    dy = 1.5
    
    centers = []
    for i in [-1, 0, 1]:
        centers.append((i * dx, 0.0))
    for i in [-1.5, -0.5, 0.5, 1.5]:
        centers.append((i * dx, dy))
    for i in [-1.5, -0.5, 0.5, 1.5]:
        centers.append((i * dx, -dy))
        
    inner = [[x, y, 0.0] for x, y in centers]
    
    best_S = float('inf')
    best_angle = 0.0
    
    # Sample outer angles finely
    for outer_angle in np.linspace(0, 60, 121):
        rad = math.radians(outer_angle)
        max_proj = 0
        for k in range(6):
            theta = rad + math.radians(k * 60)
            ct = math.cos(theta)
            st = math.sin(theta)
            
            proj = 0
            for cx, cy in centers:
                for va in [30, 90, 150, 210, 270, 330]:
                    vr = math.radians(va)
                    vx = cx + math.cos(vr)
                    vy = cy + math.sin(vr)
                    p = vx * ct + vy * st
                    if p > proj:
                        proj = p
            if proj > max_proj:
                max_proj = proj
                
        r = max_proj
        S = 2 * r / math.sqrt(3)
        if S < best_S:
            best_S = S
            best_angle = outer_angle
            
    return inner, (0.0, 0.0), best_S, best_angle
# EVOLVE-BLOCK-END
