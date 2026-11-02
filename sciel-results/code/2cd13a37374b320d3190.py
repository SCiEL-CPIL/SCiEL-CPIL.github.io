import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Create a compact cluster of 12 hexagons
    # We use a 7-hex cluster (center + 6 neighbors) and add 5 more
    # to form a denser shape.
    
    # Centers of the 7-hex cluster (flat-top hexagons, distance sqrt(3) between centers)
    d = np.sqrt(3)
    centers = [
        (0.0, 0.0),
        (d, 0.0),
        (d/2, 1.5),
        (-d/2, 1.5),
        (-d, 0.0),
        (-d/2, -1.5),
        (d/2, -1.5)
    ]
    
    # Add 5 more hexagons to extend the cluster
    # Placing them in the "valleys" or extending the perimeter
    # To keep it compact, we place them in a ring around the 7-hex cluster
    # but we only have 5 slots in the next ring (which has 12 slots).
    # Let's place them at angles 0, 60, 120, 180, 240 degrees at distance 2*d?
    # No, that leaves gaps.
    
    # Let's try a 3x4 grid rotated to be more circular.
    # Axial coords for a 3x4 grid:
    # q in [0, 2], r in [0, 3]
    # This is 12 hexagons.
    # Convert to Cartesian:
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    
    points = []
    for q in range(3):
        for r in range(4):
            x = np.sqrt(3) * (q + r/2.0)
            y = 1.5 * r
            points.append((x, y))
            
    # Center the points
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    
    centered_points = [(x - cx, y - cy) for x, y in points]
    
    # The bounding box of this 3x4 grid is roughly:
    # Width: 3*sqrt(3) + sqrt(3) = 4*sqrt(3) ~ 6.9
    # Height: 3*1.5 + 3 = 7.5
    # This is elongated.
    
    # Let's rotate it by 30 degrees to make it more square/circular?
    # Or just use a S=5 hexagon which will easily contain it.
    # But we want lower S.
    
    # Let's try to fit it in S=4.5.
    # Max distance from center:
    # Corner (0,0) -> (0,0) relative to center?
    # Let's calculate max radius.
    max_r = 0
    for x, y in centered_points:
        r = np.sqrt(x**2 + y**2)
        if r > max_r:
            max_r = r
            
    # The required S for a hexagon to contain a point at distance r (if aligned)
    # is roughly r / (sqrt(3)/2) if r is in the direction of a vertex?
    # Actually, for a regular hexagon of side S, the circumradius is S.
    # So if max_r <= S, it fits? No, the hexagon is not a circle.
    # The inradius is S*sqrt(3)/2.
    # If the points are within the inradius circle, they fit.
    # max_r is likely > inradius.
    
    # Let's just set S=5.0 to be safe.
    # Score = 12/5 = 2.4.
    
    # Can we do better?
    # Let's use a 4x3 grid but rotated?
    
    # Let's try a different packing: 1 center, 6 around, 5 more in a line.
    # This is elongated.
    
    # Let's use a "flower" of 7 and 5 in the corners of the bounding hex of the 7.
    # The 7-hex cluster fits in S=3.
    # If we add 5 hexes at the corners of the S=3 hex, the new S will be larger.
    # Corner of S=3 hex is at distance 3.
    # New hex center at distance 3 + something?
    
    # Let's just output the 3x4 grid centered, with S=5.0.
    # This is a significant improvement over 12.
    
    inner = []
    for x, y in centered_points:
        inner.append([float(x), float(y), 0.0])
        
    # Estimate S
    # The 3x4 grid has height 7.5 and width 6.9.
    # A hexagon of side S has height 2*S (if pointy top) or S*sqrt(3) (if flat top).
    # If we align the long axis of the grid with the flat side of the hex:
    # Width of hex = S*sqrt(3). Height = 2*S.
    # If we rotate the grid by 90 deg, width 7.5, height 6.9.
    # We need S*sqrt(3) >= 7.5 => S >= 4.33
    # And 2*S >= 6.9 => S >= 3.45
    # So S=4.5 should work.
    
    S = 4.5
    return inner, (0.0, 0.0), S, 0.0
# EVOLVE-BLOCK-END
