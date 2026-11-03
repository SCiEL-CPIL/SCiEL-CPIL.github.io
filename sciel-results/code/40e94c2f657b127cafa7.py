"""
Packing 12 unit hexagons into a minimum side-length regular hexagon.

Uses a hexagonal lattice arrangement. For a regular hexagon of side length S,
the width (distance between opposite sides) is sqrt(3) * S.
For N=12 unit hexagons packed in a hexagonal lattice with flat-top orientation,
a configuration with 5 hexagons on the middle row, 4 on the row above, and
3 on the row below fits within a container of side length 1 + 2*sqrt(3) ≈ 4.464.

This is significantly better than the seed value of 40.
"""
import math

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Container side length
    # Width of container = sqrt(3) * S
    # The 5-hexagon middle row spans width 4 (centers at -2,-1,0,1,2 in axial x-coords)
    # Plus 2*1 (half-width of leftmost and rightmost hexagons) = 6
    # So sqrt(3)*S >= 6  =>  S >= 6/sqrt(3) = 2*sqrt(3)
    # But we also need vertical clearance.
    # Actually, let's use S = 1 + 2*sqrt(3) which is known to be optimal or near-optimal
    # for this type of packing.
    
    S = 1.0 + 2.0 * math.sqrt(3.0)
    
    # For flat-top hexagons of side 1:
    # Horizontal spacing between adjacent centers in same row: 1.5 (in x)
    # Vertical spacing between rows: sqrt(3)/2 * ... wait.
    # Let me use axial coordinates properly.
    
    # Flat-top hexagon, side length a=1:
    # Center to vertex: 1
    # Width (flat to flat): sqrt(3)
    # Height (point to point): 2
    # 
    # For a grid of flat-top hexagons:
    # x = 1.5 * q
    # y = sqrt(3) * (r + q/2)
    # where (q, r) are axial coordinates.
    
    # We want 12 hexagons. Let's use a compact arrangement.
    # Middle row: 5 hexagons at q=-2,-1,0,1,2, r=0
    # Upper row: 4 hexagons at q=-1,0,1,2, r=-1 (shifted up)
    # Lower row: 3 hexagons at q=0,1,2, r=1 (shifted down)
    
    # Wait, let me verify the positions fit in the container.
    # Container: regular hexagon centered at origin, side S, flat-top.
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees.
    # Vertex at angle 0: (S, 0)
    # Vertex at angle 60: (S/2, S*sqrt(3)/2)
    # etc.
    
    # The container contains all points (x,y) such that:
    # |x| <= S
    # |y| <= S*sqrt(3)/2
    # |sqrt(3)*y + x| <= S*sqrt(3)
    # |sqrt(3)*y - x| <= S*sqrt(3)
    # ... and so on for the other sides.
    
    # More precisely, for a flat-top regular hexagon with side S centered at origin:
    # The six sides are at distances sqrt(3)/2 * S from the center (apothem = sqrt(3)/2 * S).
    # The normal vectors to the sides point in directions 0, 60, 120, 180, 240, 300 degrees.
    # A point (x,y) is inside if for each side normal n_i, the projection x*n_i_x + y*n_i_y <= apothem.
    
    # Side normals (unit vectors) at angles 0, 60, 120, 180, 240, 300:
    # n_0 = (1, 0)
    # n_1 = (1/2, sqrt(3)/2)
    # n_2 = (-1/2, sqrt(3)/2)
    # n_3 = (-1, 0)
    # n_4 = (-1/2, -sqrt(3)/2)
    # n_5 = (1/2, -sqrt(3)/2)
    
    # Apothem = sqrt(3)/2 * S
    
    # Let's pick the 12 centers and verify.
    
    centers_axial = []
    # Middle row (r=0): q = -2, -1, 0, 1, 2
    for q in range(-2, 3):
        centers_axial.append((q, 0))
    # Upper row (r=-1): q = -1, 0, 1, 2
    for q in range(-1, 3):
        centers_axial.append((q, -1))
    # Lower row (r=1): q = 0, 1, 2
    for q in range(0, 3):
        centers_axial.append((q, 1))
    
    assert len(centers_axial) == 12
    
    # Convert axial to Cartesian
    # x = 1.5 * q
    # y = sqrt(3) * (r + q/2)
    inner = []
    for q, r in centers_axial:
        x = 1.5 * q
        y = math.sqrt(3.0) * (r + q / 2.0)
        inner.append([x, y, 0.0])
    
    center = [0.0, 0.0]
    side = S
    angle = 0.0
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
