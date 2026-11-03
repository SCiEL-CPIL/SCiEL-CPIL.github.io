"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    sqrt3 = np.sqrt(3.0)
    
    # Hexagonal lattice: flat-top hexagons, axial coords (q,r) -> (x,y)
    # x = 1.5*q, y = sqrt3*(r + q/2)
    def axial_to_xy(q, r):
        return 1.5 * q, sqrt3 * (r + q / 2.0)
    
    # 11 hexagons in a hexagonal cluster arrangement
    # Center + 6 in ring 1 + 4 from ring 2 (symmetric partial ring)
    positions_axial = [
        (0, 0),      # center
        (1, 0),      # ring 1
        (1, -1),
        (0, -1),
        (-1, 0),
        (-1, 1),
        (0, 1),
        # 4 more from ring 2, placed symmetrically
        (2, -1),
        (1, -2),
        (-1, -1),
        (-2, 0),
    ]
    
    inner = []
    centers = []
    for q, r in positions_axial:
        x, y = axial_to_xy(q, r)
        inner.append([float(x), float(y), 0.0])
        centers.append((x, y))
    
    # Compute centroid
    cx = sum(c[0] for c in centers) / len(centers)
    cy = sum(c[1] for c in centers) / len(centers)
    
    # Find minimum bounding hexagon side length
    # For a flat-top hexagon with center (cx,cy) and side s, apothem a = s*sqrt3/2
    # A point (px,py) is inside if for all 6 edge normals, projection <= a
    # Edge normals at angles 0, 60, 120, 180, 240, 300 degrees
    
    # Each inner hexagon has 6 vertices at distance 1 from its center
    # Collect all vertices of all inner hexagons
    all_vertices = []
    for px, py in centers:
        for k in range(6):
            angle = np.radians(60 * k)
            vx = px + np.cos(angle)
            vy = py + np.sin(angle)
            all_vertices.append((vx, vy))
    
    # For flat-top outer hexagon centered at (cx, cy), find min side
    # The 6 half-plane constraints: for each normal direction phi,
    # (vx-cx)*cos(phi) + (vy-cy)*sin(phi) <= a, where a = s*sqrt3/2
    max_proj = 0.0
    for vx, vy in all_vertices:
        for k in range(6):
            phi = np.radians(60 * k)
            proj = (vx - cx) * np.cos(phi) + (vy - cy) * np.sin(phi)
            if proj > max_proj:
                max_proj = proj
    
    side = 2.0 * max_proj / sqrt3
    
    return inner, (float(cx), float(cy)), float(side), 0.0
# EVOLVE-BLOCK-END
