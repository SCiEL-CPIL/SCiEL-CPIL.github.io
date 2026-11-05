"""Hexagon packing: 12 unit hexagons in a hexagonal cluster, outer side = 4."""
import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Hexagonal cluster: 1 center + 6 around + 5 in outer ring
    # Pointy-top hexagons, axial coordinates (q, r)
    # Center at (0,0), neighbors at distance 1 (flat-top) or √3/2 (pointy-top)
    
    # For pointy-top hexagons with side 1:
    # Axial to pixel: x = √3/2 * (q + r/2), y = 3/2 * r
    # Adjacent hexagons touch when distance = √3/2 (pointy-top)
    
    # Cluster positions (axial coords q, r):
    # Center: (0,0)
    # Ring 1 (6): (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Ring 2 (5 more to make 12): (2,0), (1,1), (-1,2), (-2,1), (2,-1)
    # Wait, let me count: 1 + 6 + 5 = 12
    
    # Actually for a hexagonal number pattern: 1, 7, 19...
    # 12 = 1 + 6 + 5 (partial ring 2)
    
    # Let me use a different arrangement: 2 rows of 6, or a 3-4-3-2 pattern
    # Better: use the known optimal for small n
    
    # For n=12, a hexagonal arrangement with rows 3+4+3+2 = 12
    # Or simply: place them in a hexagonal lattice pattern
    
    # Simplest: hexagonal cluster 1+6+5
    # Axial coordinates for 12 hexagons:
    positions_axial = [
        (0, 0),       # center
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),  # ring 1
        (2, 0), (1, 1), (-1, 2), (-2, 1), (2, -1),  # partial ring 2 (5)
    ]
    
    # Convert axial to pixel (pointy-top, side=1)
    sqrt3 = np.sqrt(3)
    inner = []
    for q, r in positions_axial:
        x = sqrt3 / 2.0 * (q + r / 2.0)
        y = 1.5 * r
        inner.append([x, y, 0.0])
    
    # Outer hexagon: center at origin, side length 4, pointy-top (0 degrees)
    # Verify: the cluster fits inside a hexagon of side 4
    # The farthest point from center in the cluster is at ring 2
    # Distance from center to farthest hex center: sqrt3/2 * sqrt(2^2 + 0^2) = sqrt3 ≈ 1.732
    # Plus hex radius (circumradius = 1 for unit hexagon) = 2.732
    # Outer hexagon of side 4 (pointy-top) has inradius = 4 * sqrt3/2 = 3.464
    # 2.732 < 3.464, so it fits with margin
    
    return inner, (0.0, 0.0), 4.0, 0.0
# EVOLVE-BLOCK-END
