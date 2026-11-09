import math

N = 12

def construct():
    # Hexagonal lattice coordinates for 12 hexagons
    # Basis: u = (1, 0), v = (0.5, sqrt(3)/2)
    # We want a compact arrangement. The optimal arrangement for 12 on a hex lattice
    # within a hexagon is typically a "hexagonal number" shape or close to it.
    # For N=12, the hexagonal number H_3 = 1+3+6 = 10 is too small. H_4 = 1+3+6+9 = 19 is too big.
    # So we take a hexagonal cluster of 7 (center + 6 neighbors) and add 5 more.
    # Or we can use a specific set of 12 points that minimizes the max distance to center.
    # Let's list points in a hexagonal grid (i, j) where pos = i*u + j*v.
    # Center (0,0).
    # Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1) -> 6 points. Total 7.
    # Ring 2: (2,0), (1,1), (0,2), (-1,2), (-2,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-2), (2,-1) -> 12 points.
    # We need 5 more from Ring 2 to make 12.
    # To minimize the bounding hexagon, we should pick points that are "balanced".
    # However, the task says "minimizing the outer side length".
    # The outer hexagon contains the union of the 12 inner hexagons.
    # If the inner hexagons are aligned with the outer one, the containment is easier to calculate.
    # Let's pick 5 points from Ring 2 that are roughly symmetric.
    # The distance of a point (i,j) from the center in the "hex metric" is max(|i|, |j|, |i+j|)? No, that's for axial coordinates.
    # Let's just use Cartesian distance for initial heuristic, but the constraint is hexagonal containment.
    # Actually, for a hexagonal lattice of hexagons, the "radius" of the cluster determines the size.
    # Let's try to fit them into a hexagon of side length S.
    # If we place hexagons at lattice points, the minimum enclosing hexagon of the *centers* plus the hexagon size determines S.
    
    # Let's use a known good configuration for N=12.
    # A common dense packing is a hexagon of side 2 (which has 7 cells) plus 5 more attached.
    # Let's try points:
    # (0,0)
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # (2,0), (1,1), (-1,2), (-2,1), (0,-2)  -- picking 5 from the next ring.
    
    points = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (-1, 2), (-2, 1), (0, -2)
    ]
    
    # Convert to Cartesian
    u = (1.0, 0.0)
    v = (0.5, math.sqrt(3) / 2.0)
    
    coords = []
    for i, j in points:
        x = i * u[0] + j * v[0]
        y = i * u[1] + j * v[1]
        coords.append((x, y))
        
    # Calculate the required side length S of the outer hexagon.
    # The outer hexagon is centered at (0,0) and aligned with the lattice (angle 0).
    # A point (x,y) is inside a regular hexagon of side S centered at origin (flat-top) if:
    # |x| <= S * sqrt(3)/2 * (1 + 2*|y|/(3*S)) ? No, let's use the standard inequality.
    # For a hexagon with vertices at (S, 0), (S/2, S*sqrt(3)/2), etc. (pointy-top? or flat-top?)
    # Standard hexagon with side S, centered at origin, flat top:
    # Vertices: (S, 0), (S/2, S*sqrt(3)/2), (-S/2, S*sqrt(3)/2), (-S, 0), (-S/2, -S*sqrt(3)/2), (S/2, -S*sqrt(3)/2)
    # Inequalities:
    # 1. |x| <= S
    # 2. |y| <= S * sqrt(3) / 2
    # 3. |y| <= sqrt(3) * (S - |x|)  ->  |y| + sqrt(3)*|x| <= sqrt(3)*S ? No.
    # Let's derive: The lines are y = sqrt(3)(x - S) for x in [S/2, S].
    # Equation: sqrt(3)x + y = sqrt(3)S.
    # So |sqrt(3)x + y| <= sqrt(3)S is not quite right because of symmetry.
    # The conditions for (x,y) to be in hexagon (flat-top, side S) are:
    # |x| <= S
    # |y| <= S * sqrt(3) / 2
    # |y| <= sqrt(3) * (S - |x|)
    
    # We need the inner hexagons to be *inside* the outer one.
    # An inner hexagon of side 1 centered at (cx, cy) is inside the outer hexagon if its furthest point is inside.
    # Actually, it's easier to check if the *centers* are inside a shrunken hexagon.
    # The inner hexagon has "radius" (apothem) 1 * sqrt(3)/2.
    # If we shrink the outer hexagon by the apothem of the inner hexagon, the centers must lie within this shrunken hexagon.
    # Shrunken hexagon side S' = S - 1. (Since the apothem scales linearly with S, and apothem_inner = sqrt(3)/2 * 1, apothem_outer = sqrt(3)/2 * S. Distance between boundaries is S - 1).
    # So we need S - 1 to be large enough to contain the centers.
    # Let S_inner = S - 1.
    # Check if all (x,y) are inside hexagon of side S_inner.
    
    max_S_inner = 0
    for x, y in coords:
        # Find minimum S_inner such that (x,y) is inside.
        # Conditions:
        # |x| <= S_inner
        # |y| <= S_inner * sqrt(3) / 2
        # |y| <= sqrt(3) * (S_inner - |x|)
        
        # From 1: S_inner >= |x|
        # From 2: S_inner >= 2|y| / sqrt(3)
        # From 3: S_inner - |x| >= |y|/sqrt(3) => S_inner >= |x| + |y|/sqrt(3)
        
        S1 = abs(x)
        S2 = 2 * abs(y) / math.sqrt(3)
        S3 = abs(x) + abs(y) / math.sqrt(3)
        
        S_req = max(S1, S2, S3)
        max_S_inner = max(max_S_inner, S_req)
        
    S = max_S_inner + 1.0
    
    # Construct inner list
    inner = []
    for x, y in coords:
        inner.append([x, y, 0.0])
        
    return inner, (0.0, 0.0), S, 0.0
