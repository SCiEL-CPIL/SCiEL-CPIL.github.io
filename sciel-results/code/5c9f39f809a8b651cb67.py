import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Define the lattice basis for hexagonal packing
    # Unit hexagon side length 1
    # Distance between centers in x-direction: sqrt(3)
    # Distance between centers in y-direction (offset): 1.5
    # Lattice vectors:
    # v1 = (sqrt(3), 0)
    # v2 = (sqrt(3)/2, 1.5)
    
    sqrt3 = math.sqrt(3)
    
    # Lattice coordinates (u, v) for 11 hexagons
    # Center (0,0)
    # Ring 1: (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)
    # Ring 2 (selected 4 for compactness): (2,0), (-2,0), (1,1), (-1,-1)
    
    lattice_coords = [
        (0, 0),
        (1, 0), (-1, 0),
        (0, 1), (0, -1),
        (1, -1), (-1, 1),
        (2, 0), (-2, 0),
        (1, 1), (-1, -1)
    ]
    
    inner = []
    for u, v in lattice_coords:
        x = sqrt3 * (u + v / 2.0)
        y = 1.5 * v
        angle = 0.0
        inner.append([x, y, angle])
        
    # Calculate the required outer side length
    # The cluster is symmetric.
    # Max extent in x-direction: center at (2,0) -> x = 2*sqrt(3). Vertex at x = 2*sqrt(3) + 1.
    # Max extent in y-direction: center at (0,1) -> y = 1.5. Vertex at y = 1.5 + sqrt(3)/2.
    # However, we need to fit in a regular hexagon.
    # The "tightest" hexagon for this shape, aligned with the lattice (vertex at 0 deg),
    # has side length S such that the vertex of the outer hexagon touches the furthest vertex of the cluster.
    # Furthest vertex distance from center:
    # Point (2,0) hexagon has vertex at (2*sqrt(3) + 1, 0). Distance = 2*sqrt(3) + 1.
    # Let's verify if this is the max radius.
    # Point (1,1) hexagon center: (sqrt(3) + sqrt(3)/2, 1.5) = (1.5*sqrt(3), 1.5).
    # Its top-right vertex (angle 0 relative to center? No, hexagon is flat-top, vertex at 0).
    # Vertex at (center_x + 1, center_y).
    # (1.5*sqrt(3) + 1, 1.5).
    # Distance squared: (1.5*sqrt(3) + 1)^2 + 1.5^2
    # 1.5*sqrt(3) approx 2.598.
    # (3.598)^2 + 2.25 = 12.94 + 2.25 = 15.19.
    # sqrt(15.19) approx 3.90.
    # The point (2*sqrt(3)+1, 0) has distance 2*sqrt(3)+1 approx 4.464.
    # So the max radius is 2*sqrt(3)+1.
    
    # If we align the outer hexagon with a vertex at 0 degrees,
    # the side length S must be at least the max radius.
    # S = 2*sqrt(3) + 1
    
    side_length = 2 * sqrt3 + 1
    center = (0.0, 0.0)
    outer_angle = 0.0
    
    return inner, center, side_length, outer_angle
# EVOLVE-BLOCK-END
