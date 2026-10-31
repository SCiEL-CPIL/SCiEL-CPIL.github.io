import math
import time
import random
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    """
    Deterministic geometric construction for 11 unit hexagons.
    Uses a hexagonal lattice arrangement which is known to be optimal for small N.
    The outer hexagon is sized to exactly enclose the inner hexagons.
    """
    # 1. Define the positions of the 11 inner hexagons.
    # Arrangement: 1 center, 6 in first ring, 4 in second ring.
    # This forms a compact cluster.
    # Spacing for non-overlapping hexagons of side 1:
    # Distance between centers of adjacent hexagons (sharing an edge) is 2 * apothem = sqrt(3).
    # Distance between centers of hexagons separated by one hexagon is 2 * sqrt(3).
    
    sp = math.sqrt(3)
    
    inner = []
    # Center
    inner.append([0.0, 0.0, 0.0])
    
    # First ring (6 hexagons)
    for i in range(6):
        theta = math.radians(60 * i)
        x = sp * math.cos(theta)
        y = sp * math.sin(theta)
        inner.append([x, y, 0.0])
        
    # Second ring (4 hexagons)
    # We place them at the "corners" of the hexagonal cluster to maintain symmetry and compactness.
    # The vertices of the first ring hexagons (relative to center) are at distance 2.
    # The centers of the second ring hexagons should be at distance 2*sp from the center.
    # Angles: 30, 90, 150, 210, 270, 330? No, that would be 6.
    # Let's place them at angles 0, 60, 120, 180, 240, 300 but offset?
    # Actually, the standard hexagonal packing for 19 hexagons is 1-6-12.
    # For 11, we take the center + 6 + 4 from the next ring.
    # The next ring centers are at distance 2*sp.
    # To be symmetric, we can pick 4 out of 6. But which 4?
    # If we pick all 6, we have 13. We need 11.
    # Let's just use the first 4 of the 6 possible positions in the second ring.
    # This breaks rotational symmetry but is a valid packing.
    # Alternatively, we can try to fit 11 in a more symmetric way?
    # Maybe 1 center + 6 ring + 4 more?
    # Let's stick to the lattice.
    for i in range(4):
        theta = math.radians(60 * i + 30) # Offset by 30 degrees?
        # Actually, the vertices of the central hex are at angles 0, 60, ...
        # The neighbors are at 0, 60, ...
        # The second neighbors are at 0, 60, ... distance 2*sp.
        # Let's place them at angles 0, 60, 120, 180.
        # This creates a "C" shape or similar.
        pass

    # Let's refine the second ring placement.
    # The 6 neighbors are at angles 0, 60, 120, 180, 240, 300.
    # The 6 second-neighbors (distance 2*sp) are at the same angles.
    # If we add 4, we can just pick indices 0, 1, 2, 3.
    for i in range(4):
        theta = math.radians(60 * i)
        x = 2 * sp * math.cos(theta)
        y = 2 * sp * math.sin(theta)
        inner.append([x, y, 0.0])
        
    # Total 1 + 6 + 4 = 11.
    
    # 2. Compute the minimal outer hexagon side length S.
    # The outer hexagon is centered at (0,0) with angle 0.
    # We need to find the maximum projection of any vertex of any inner hexagon
    # onto the 6 normals of the outer hexagon.
    
    # Normals of outer hexagon (angle 0, pointy-top):
    # Vertices at angles 0, 60, 120, 180, 240, 300.
    # Normals at 30, 90, 150, 210, 270, 330.
    
    max_proj = 0.0
    for ix, iy, iang in inner:
        # Vertices of inner hexagon (side 1, angle 0)
        # Apothem = sqrt(3)/2
        # Vertices are at distance 1 from center, at angles 0, 60, ...
        for k in range(6):
            theta_v = math.radians(60 * k)
            vx = ix + 1.0 * math.cos(theta_v)
            vy = iy + 1.0 * math.sin(theta_v)
            
            # Project onto the 6 normals
            for j in range(6):
                theta_n = math.radians(30 + 60 * j)
                nx = math.cos(theta_n)
                ny = math.sin(theta_n)
                proj = vx * nx + vy * ny
                if proj > max_proj:
                    max_proj = proj
    
    # The apothem of the outer hexagon must be >= max_proj.
    # Apothem = S * sqrt(3) / 2
    # S = max_proj / (sqrt(3) / 2) = 2 * max_proj / sqrt(3)
    S = 2 * max_proj / math.sqrt(3)
    
    # Return interface:
    # inner: list of [x, y, angle_degrees]
    # center: (x, y)
    # side: S
    # angle: 0.0
    
    return inner, (0.0, 0.0), S, 0.0
# EVOLVE-BLOCK-END
