"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Hexagonal lattice with spacing sqrt(3) between centers
    # Flat-top hexagons: lattice vectors v1 = (sqrt(3), 0), v2 = (sqrt(3)/2, 1.5)
    # Position = q*v1 + r*v2 = (sqrt(3)*(q + r/2), 1.5*r)
    
    # 11 hexagons in a compact cluster
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (-2, 1), (-1, -1)
    ]
    
    centers = []
    for q, r in axial_coords:
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        centers.append((x, y))
    
    centers = np.array(centers)
    
    # Edge normals for flat-top hexagon: angles 0, 60, 120, 180, 240, 300
    angles = np.deg2rad(np.arange(0, 360, 60))
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Apothem must be >= max_c (c . n) + 1 for each normal n
    a = 0.0
    for n in normals:
        max_dot = np.max(centers @ n)
        a = max(a, max_dot + 1.0)
    
    side = 2.0 * a / np.sqrt(3)
    
    inner = [[x, y, 0.0] for x, y in centers]
    center = (0.0, 0.0)
    angle_outer = 0.0
    
    return inner, center, float(side), angle_outer
# EVOLVE-BLOCK-END
