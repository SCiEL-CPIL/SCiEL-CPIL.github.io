import math
import itertools

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Generate hex grid coordinates
    # Axial coordinates (q, r)
    # Convert to Cartesian (x, y) for pointy-top hexagons
    # x = sqrt(3) * (q + r/2)
    # y = 3/2 * r
    
    # We want the 11 closest hexagons to the origin
    # Distance 0: (0,0)
    # Distance sqrt(3): (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Distance 3: (1,1), (-1,2), (-2,1), (-1,-1), (1,-2), (2,-1)
    # Distance 2*sqrt(3): (2,0), (0,2), (-2,0), (0,-2), (-1,1) is used...
    
    # List of candidates for the 4 extra hexagons
    # The 6 hexes at distance 3 are:
    ring2 = [
        (1, 1), (-1, 2), (-2, 1), (-1, -1), (1, -2), (2, -1)
    ]
    
    # The 7 base hexes
    base = [
        (0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)
    ]
    
    best_L = float('inf')
    best_inner = None
    
    # Try all combinations of 4 from ring2
    for combo in itertools.combinations(ring2, 4):
        hexes = base + list(combo)
        
        # Calculate required L for this set
        # Outer hexagon is pointy-top, centered at (0,0)
        # Apothem A = L * sqrt(3) / 2
        # Constraints: x*cos(theta) + y*sin(theta) <= A
        # Thetas: 30, 90, 150, 210, 270, 330 degrees
        
        max_proj = 0.0
        
        for q, r in hexes:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            
            # Vertices of the unit pointy-top hexagon at (x, y)
            # Side length 1
            # Apothem of inner hex = sqrt(3)/2
            # Circumradius = 1
            # Vertices:
            # (x, y+1)
            # (x + sqrt(3)/2, y + 0.5)
            # (x + sqrt(3)/2, y - 0.5)
            # (x, y-1)
            # (x - sqrt(3)/2, y - 0.5)
            # (x - sqrt(3)/2, y + 0.5)
            
            verts = [
                (x, y + 1.0),
                (x + math.sqrt(3)/2.0, y + 0.5),
                (x + math.sqrt(3)/2.0, y - 0.5),
                (x, y - 1.0),
                (x - math.sqrt(3)/2.0, y - 0.5),
                (x - math.sqrt(3)/2.0, y + 0.5)
            ]
            
            for theta_deg in [30, 90, 150, 210, 270, 330]:
                theta = math.radians(theta_deg)
                c = math.cos(theta)
                s = math.sin(theta)
                
                for vx, vy in verts:
                    proj = vx * c + vy * s
                    if proj > max_proj:
                        max_proj = proj
        
        # L * sqrt(3) / 2 >= max_proj
        L = (2.0 / math.sqrt(3)) * max_proj
        
        if L < best_L:
            best_L = L
            best_inner = hexes

    # Convert best_inner to the required format: [x, y, angle]
    # Angle is 0.0 (aligned with outer)
    inner_list = []
    for q, r in best_inner:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        inner_list.append([x, y, 0.0])

    return inner_list, (0.0, 0.0), best_L, 0.0
# EVOLVE-BLOCK-END
