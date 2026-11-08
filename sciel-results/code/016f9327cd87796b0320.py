import numpy as np
import math

N = 12

def construct():
    # Generate 12 hexagons in a 3x4 rhombus packing (flat-top)
    # Axial coords q in [0, 3], r in [-1, 1]
    # x = 1.5 * q
    # y = sqrt(3) * (r + q/2)
    
    centers = []
    sqrt3 = math.sqrt(3)
    for q in range(4):
        for r in range(-1, 2):
            x = 1.5 * q
            y = sqrt3 * (r + q / 2.0)
            centers.append((x, y))
            
    # Center the cluster
    xs = [c[0] for c in centers]
    ys = [c[1] for c in centers]
    cx = sum(xs) / N
    cy = sum(ys) / N
    
    inner = []
    for x, y in centers:
        inner.append([x - cx, y - cy, 0.0])
        
    # Optimize outer hexagon rotation
    # Outer hexagon is regular, side S.
    # We test rotations 0 to 30 degrees.
    
    best_S = float('inf')
    best_rot = 0.0
    
    # Normal angles for flat-top hexagon: 30, 90, 150, 210, 270, 330
    # Rotated by phi
    base_normals = [30, 90, 150, 210, 270, 330]
    
    for phi_deg in np.linspace(0, 30, 61):
        phi = math.radians(phi_deg)
        max_proj = 0.0
        for x, y, _ in inner:
            for base_ang in base_normals:
                ang = math.radians(base_ang) + phi
                # Project point onto normal
                proj = x * math.cos(ang) + y * math.sin(ang)
                if proj > max_proj:
                    max_proj = proj
        # Apothem A = max_proj
        # Side S = 2 * A / sqrt(3)
        S = 2.0 * max_proj / sqrt3
        if S < best_S:
            best_S = S
            best_rot = phi_deg
            
    return inner, (0.0, 0.0), best_S, best_rot
