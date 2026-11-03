"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    # For 11 hexagons: 1 center + 6 ring1 + 4 ring2
    # Hexagonal lattice: ring k has 6k hexagons at distance k*sqrt(3)
    # Ring 1: 6 hexes at distance sqrt(3), angles 0,60,120,180,240,300
    # Ring 2: we place 4 hexes at distance 2*sqrt(3)
    
    sqrt3 = np.sqrt(3)
    
    # Center hexagon
    centers = [(0.0, 0.0)]
    
    # Ring 1: 6 hexagons
    for k in range(6):
        angle = k * np.pi / 3
        x = sqrt3 * np.cos(angle)
        y = sqrt3 * np.sin(angle)
        centers.append((x, y))
    
    # Ring 2: 4 hexagons - place them to minimize max extent
    # Ring 2 positions are at distance 2*sqrt(3)
    # Available angles: 0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330 (12 positions)
    # But in hex lattice, ring 2 has 12 positions at angles: 0, 30, 60, ..., 330? 
    # Actually in hex lattice, ring 2 positions are at:
    # 6 at same angles as ring1 (0,60,...) at dist 2*sqrt3, and 6 at 30,90,... at dist 2*sqrt3
    # Wait, let me reconsider. In a hexagonal lattice of hexagons:
    # The neighbors of the center are at 6 positions.
    # The next ring has 12 positions.
    
    # For minimizing the bounding hexagon, place ring 2 hexes symmetrically.
    # Place 4 ring-2 hexes at angles 0, 120, 240, and 300 (or similar)
    # Actually, to be symmetric and minimize max radius, place them at 4 of the 6 "corner" positions
    # of ring 2 that are at the same angles as ring 1: 0, 60, 120, 180, 240, 300
    # Pick 4: say 0, 120, 240, and one more. For symmetry, pick 0, 120, 240 and skip 60, 180, 300?
    # Better: pick 60, 180, 300, and 0? That's 4 positions.
    
    # Let me place 4 ring-2 hexes at angles: 0, 120, 240, and 60 (4 out of 6 main directions)
    # Actually for minimal bounding box, place them at 4 positions that are as balanced as possible.
    
    ring2_angles = [0, np.pi/3, 2*np.pi/3, np.pi]  # 0, 60, 120, 180 degrees - 4 positions
    for angle in ring2_angles:
        x = 2 * sqrt3 * np.cos(angle)
        y = 2 * sqrt3 * np.sin(angle)
        centers.append((x, y))
    
    # Now compute the minimum outer hexagon side length
    # For each inner hexagon, compute the farthest vertex distance from origin
    # The outer hexagon with side S has circumradius S, so we need S >= max vertex distance
    
    max_dist = 0.0
    for (cx, cy) in centers:
        # Vertices of unit hexagon centered at (cx, cy)
        for v in range(6):
            va = v * np.pi / 3
            vx = cx + np.cos(va)
            vy = cy + np.sin(va)
            d = np.sqrt(vx**2 + vy**2)
            if d > max_dist:
                max_dist = d
    
    # The outer hexagon side length must be at least max_dist (circumradius)
    # But we also need to ensure the outer hexagon actually contains all points.
    # A regular hexagon with circumradius R contains a point at distance r from center
    # if r <= R (since circumradius is the max distance to a vertex).
    # So S = max_dist should work.
    
    S = max_dist
    
    # Build inner list: (x, y, angle_degrees)
    inner = []
    for (cx, cy) in centers:
        inner.append([cx, cy, 0.0])
    
    return inner, (0.0, 0.0), S, 0.0
# EVOLVE-BLOCK-END
