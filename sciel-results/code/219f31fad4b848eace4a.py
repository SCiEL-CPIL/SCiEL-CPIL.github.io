import numpy as np
import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Generate hexagonal lattice points
    # Axial coordinates (q, r)
    # Euclidean coords:
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    # This corresponds to pointy-top hexagons?
    # Let's check distance.
    # (0,0) to (1,0) -> x=sqrt(3), y=0. Dist=sqrt(3). Correct.
    # (0,0) to (0,1) -> x=sqrt(3)/2, y=1.5. Dist=sqrt(3/4+2.25)=sqrt(3). Correct.
    
    points = []
    # Generate enough points
    max_dist = 5
    for q in range(-max_dist, max_dist + 1):
        for r in range(-max_dist, max_dist + 1):
            # Hex distance
            d = 0.5 * (abs(q) + abs(r) + abs(q + r))
            if d <= max_dist:
                x = math.sqrt(3) * (q + r / 2.0)
                y = 1.5 * r
                points.append((x, y))
    
    # Sort by distance to origin
    points.sort(key=lambda p: p[0]**2 + p[1]**2)
    
    # Take N points
    inner_points = points[:N]
    
    # Calculate required side length S
    # Container: Center (0,0), Angle 0
    # Normals to sides of container (angle 0 means vertices at 0, 60, ...)
    # Side normals are at 30, 90, 150, 210, 270, 330
    angles = [math.radians(30 + 60 * k) for k in range(6)]
    normals = [(math.cos(a), math.sin(a)) for a in angles]
    
    apothem_inner = math.sqrt(3) / 2.0
    apothem_factor = math.sqrt(3) / 2.0 # S * apothem_factor is the apothem of container
    
    max_proj = 0.0
    for x, y in inner_points:
        for nx, ny in normals:
            proj = x * nx + y * ny
            if proj > max_proj:
                max_proj = proj
                
    # max_proj + apothem_inner <= S * apothem_factor
    S = (max_proj + apothem_inner) / apothem_factor
    
    # Add a small epsilon to ensure strict inequality if needed, 
    # though evaluator usually handles boundary or we can just use S
    # To be safe, add 1e-9
    S += 1e-9
    
    inner = [[x, y, 0.0] for x, y in inner_points]
    center = (0.0, 0.0)
    angle = 0.0
    
    return inner, center, S, angle
# EVOLVE-BLOCK-END
