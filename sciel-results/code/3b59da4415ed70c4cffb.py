import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Side length of inner hexagons is 1.0
    # Distance between centers of touching hexagons is sqrt(3)
    d = math.sqrt(3)
    
    # 7 central hexagons
    # Center
    centers = [(0.0, 0.0)]
    
    # 6 neighbors
    # Angles 90, 30, -30, -90, -150, 150 degrees?
    # Using the vectors derived:
    # Top: (0, d)
    # TR: (1.5, d/2)
    # BR: (1.5, -d/2)
    # Bottom: (0, -d)
    # BL: (-1.5, -d/2)
    # TL: (-1.5, d/2)
    
    neighbors = [
        (0.0, d),
        (1.5, d/2),
        (1.5, -d/2),
        (0.0, -d),
        (-1.5, -d/2),
        (-1.5, d/2)
    ]
    
    for n in neighbors:
        centers.append(n)
        
    # We have 7 hexagons. Need 4 more.
    # Ring 2 candidates (12 spots)
    # These are at distance 3 from center.
    # Coordinates:
    # 1. (3, 0)
    # 2. (1.5, 3*d/2)
    # 3. (-1.5, 3*d/2)
    # 4. (-3, 0)
    # 5. (-1.5, -3*d/2)
    # 6. (1.5, -3*d/2)
    # ... and the other 6?
    # The 12 spots are the vertices of the hexagon of side 2 in grid.
    # Let's generate them properly.
    # The 12 spots are the sums of two adjacent Ring 1 vectors?
    # No, just the 6 "corner" extensions and 6 "side" extensions?
    # Actually, the 12 spots are:
    # (3, 0)
    # (1.5, 3*d/2)
    # (-1.5, 3*d/2)
    # (-3, 0)
    # (-1.5, -3*d/2)
    # (1.5, -3*d/2)
    # These are 6.
    # Where are the other 6?
    # Ah, the "side" ones are at:
    # (2.598, 1.5)? No.
    # Let's check the distance from (1.5, d/2) to (3,0).
    # dx=1.5, dy=-d/2. dist=sqrt(2.25+0.75)=sqrt(3).
    # So (3,0) is adjacent to TR and BR.
    # Is there a spot between TR and Top?
    # Top is (0, d). TR is (1.5, d/2).
    # Midpoint direction?
    # The spot adjacent to both Top and TR is (1.5, 3*d/2)?
    # Dist to Top (0,d): dx=1.5, dy=d/2. dist=sqrt(3).
    # Dist to TR (1.5, d/2): dx=0, dy=d. dist=sqrt(3).
    # Yes.
    # So the 6 "corner" spots are:
    # (3,0), (1.5, 3*d/2), (-1.5, 3*d/2), (-3,0), (-1.5, -3*d/2), (1.5, -3*d/2).
    # These are 6.
    # What are the other 6?
    # The 7-cluster has 6 sides.
    # The Ring 2 has 12 hexagons.
    # 6 touch 2 hexagons of Ring 1 (the corners).
    # 6 touch 1 hexagon of Ring 1? No, in a hex grid, every hex in Ring 2 touches 2 or 3 hexes in Ring 1?
    # Actually, the 12 hexagons in Ring 2 form a ring.
    # Each touches 2 neighbors in Ring 2 and 2 or 3 in Ring 1.
    # The 6 "corner" ones touch 2 in Ring 1.
    # The 6 "side" ones touch 1 in Ring 1?
    # Let's check the spot "above" TR?
    # TR is (1.5, d/2).
    # Neighbor above TR?
    # Vector (0, d) from TR? -> (1.5, 3*d/2). This is the corner one.
    # Vector (-1.5, d/2) from TR? -> (0, d). This is Top.
    # Vector (1.5, -d/2) from TR? -> (3, 0). This is the other corner one.
    # So TR is surrounded by Top, Corner1, Corner2, and 3 in Ring 2?
    # No, TR has 6 neighbors.
    # 1. Origin (0,0) - Ring 0
    # 2. Top (0, d) - Ring 1
    # 3. BR (1.5, -d/2) - Ring 1
    # 4. Corner1 (1.5, 3*d/2) - Ring 2
    # 5. Corner2 (3, 0) - Ring 2
    # 6. Side1 (?) - Ring 2
    # The 6th neighbor of TR must be the "side" one.
    # It is adjacent to TR and... who else?
    # It must be adjacent to Top and TR? No, that's Corner1.
    # It must be adjacent to TR and BR? No, that's Corner2.
    # So the "side" ones are adjacent to only 1 Ring 1 hex?
    # No, that would leave gaps.
    # Actually, in a hex grid, the Ring 2 consists of 12 hexes.
    # 6 are at the "corners" of the hexagon.
    # 6 are at the "midpoints" of the sides?
    # Let's check the position (2.598, 1.5).
    # x = 3*d/2? No.
    # Let's just use the 6 corner ones. They are sufficient to place 4 hexagons.
    
    ring2_corners = [
        (3.0, 0.0),
        (1.5, 1.5*d),
        (-1.5, 1.5*d),
        (-3.0, 0.0),
        (-1.5, -1.5*d),
        (1.5, -1.5*d)
    ]
    
    # Pick 4 of these
    extra = ring2_corners[:4]
    
    for e in extra:
        centers.append(e)
        
    # Convert to (x, y, angle)
    inner = [[x, y, 0.0] for x, y in centers]
    
    # Outer hexagon
    # Max radius of vertices is 4.0
    # Side length 4.0
    return inner, (0.0, 0.0), 4.0, 0.0
# EVOLVE-BLOCK-END
