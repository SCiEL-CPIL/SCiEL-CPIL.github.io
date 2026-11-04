import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Generate 11 hexagon centers on a hexagonal lattice
    # Lattice vectors
    v1 = (math.sqrt(3), 0)
    v2 = (math.sqrt(3) / 2, 1.5)
    
    points = []
    # Center
    points.append((0.0, 0.0))
    
    # Ring 1 (6 points)
    for k in range(6):
        x = v1[0] * math.cos(k * math.pi / 3) + v2[0] * math.sin(k * math.pi / 3) # This is wrong, just use lattice coords
        # Better: use integer coords (i,j)
        pass
    
    # Use integer lattice coords (i, j)
    # (0,0)
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # (2,0), (0,2), (-2,0), (0,-2)
    
    lattice_points = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (0, 2), (-2, 0), (0, -2)
    ]
    
    centers = []
    for i, j in lattice_points:
        x = i * v1[0] + j * v2[0]
        y = i * v1[1] + j * v2[1]
        centers.append((x, y))
    
    # The inner hexagons are aligned with the outer hexagon (angle 0)
    # So the effective container for centers is a hexagon of side L-1
    # We need to find min L such that centers fit in hexagon of side L-1
    
    # Calculate required side length for centers
    # Hexagon with side S, center (0,0), vertex on x-axis
    # Normals at 30, 90, 150 degrees
    # Apothem A = S * sqrt(3) / 2
    # Condition: |x cos(n) + y sin(n)| <= A
    
    min_apothem = 0.0
    for x, y in centers:
        # Normal 30 deg
        d1 = abs(x * math.cos(math.pi / 6) + y * math.sin(math.pi / 6))
        # Normal 90 deg
        d2 = abs(x * math.cos(math.pi / 2) + y * math.sin(math.pi / 2))
        # Normal 150 deg
        d3 = abs(x * math.cos(5 * math.pi / 6) + y * math.sin(5 * math.pi / 6))
        
        max_d = max(d1, d2, d3)
        if max_d > min_apothem:
            min_apothem = max_d
    
    # Side length of point container
    S_points = 2 * min_apothem / math.sqrt(3)
    
    # Outer side length
    L_outer = S_points + 1.0
    
    # Construct return values
    inner = []
    for x, y in centers:
        inner.append([x, y, 0.0])
        
    return inner, (0.0, 0.0), L_outer, 0.0
# EVOLVE-BLOCK-END
