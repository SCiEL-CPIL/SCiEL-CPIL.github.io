import numpy as np

N = 11

def construct():
    # Generate positions for 11 hexagons in a compact cluster
    # Using axial coordinates (q, r)
    # Center (0,0)
    # Ring 1: 6 hexes
    # Ring 2: 4 hexes (to make 11)
    
    positions = [(0, 0)]
    
    # Ring 1
    for i in range(6):
        angle = i * 60
        # Distance is sqrt(3)
        # In axial coords, neighbors are (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
        positions.append((1, 0))
        positions.append((0, 1))
        positions.append((-1, 1))
        positions.append((-1, 0))
        positions.append((0, -1))
        positions.append((1, -1))
        
    # We have 7. Need 4 more.
    # Pick 4 from Ring 2 to balance the shape.
    # Ring 2 positions are at distance 2 from center in axial coords?
    # (2, -1), (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1), (-2, 0), (-1, -1), (0, -2), (1, -2), (2, -1) is duplicate?
    # Let's just add 4 specific ones.
    positions.append((2, -1))
    positions.append((2, 1)) # Wait, (2,1) is not in standard ring 2?
    # Axial neighbors of (1,0) are (2,0), (2,-1), (1,1), (0,1), (0,0), (1,-1).
    # So (2,0) and (2,-1) are in Ring 2.
    # Let's add (2,0), (2,-1), (-2,0), (-2,-1) to be symmetric.
    # But (2,-1) was already added? No, I listed it.
    # Let's restart the list generation carefully.
    
    pts = [(0,0)]
    # 6 neighbors
    pts.extend([(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)])
    # 4 more from the next layer
    pts.extend([(2,-1), (2,0), (-2,0), (-2,-1)])
    
    # Convert axial to Cartesian
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    
    centers = []
    for q, r in pts:
        x = np.sqrt(3) * (q + r/2)
        y = 1.5 * r
        centers.append((x, y))
    
    # Calculate max distance from origin (center of packing)
    max_dist = 0
    for x, y in centers:
        # The hexagon at (x,y) extends to x+1, y+sqrt(3)/2 etc.
        # Max distance from origin to any point in this hexagon
        # is roughly distance to center + 1
        d = np.sqrt(x**2 + y**2) + 1
        if d > max_dist:
            max_dist = d
            
    # The container hexagon must contain a circle of radius max_dist?
    # No, it must contain the points.
    # A regular hexagon of side S contains a circle of radius S * sqrt(3) / 2.
    # So S * sqrt(3) / 2 >= max_dist
    # S >= max_dist * 2 / sqrt(3)
    
    S = max_dist * 2 / np.sqrt(3)
    # Add a small margin for safety
    S = S * 1.05
    
    # Format inner hexagons
    inner = []
    for x, y in centers:
        inner.append([x, y, 0.0])
        
    return inner, (0.0, 0.0), S, 0.0

# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    N = 11
    
    # Generate positions for 11 hexagons in a compact cluster
    pts = [(0,0)]
    # 6 neighbors
    pts.extend([(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)])
    # 4 more from the next layer (symmetric)
    pts.extend([(2,-1), (2,0), (-2,0), (-2,-1)])
    
    centers = []
    for q, r in pts:
        x = np.sqrt(3) * (q + r/2)
        y = 1.5 * r
        centers.append((x, y))
    
    max_dist = 0
    for x, y in centers:
        d = np.sqrt(x**2 + y**2) + 1
        if d > max_dist:
            max_dist = d
            
    S = max_dist * 2 / np.sqrt(3)
    S = S * 1.05
    
    inner = []
    for x, y in centers:
        inner.append([x, y, 0.0])
        
    return inner, (0.0, 0.0), S, 0.0
# EVOLVE-BLOCK-END
