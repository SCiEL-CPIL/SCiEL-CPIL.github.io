import time

def construct():
    """
    Constructs 32 points on the triangular lattice maximizing the number of edges.
    Uses the hexagonal spiral ordering, which is known to attain the Harborth bound.
    """
    n_points = 32
    
    # Generate points in hexagonal spiral order
    # The hexagonal spiral visits points in order of hexagonal distance,
    # breaking ties in a specific way to maximize edges.
    
    points = [(0, 0)]
    
    # For ring r, we walk the perimeter of the hexagon of radius r.
    # The 6 corner points of ring r are:
    # (r, 0), (0, r), (-r, r), (-r, 0), (0, -r), (r, -r)
    # We walk from (r,0) counterclockwise.
    
    # The 6 edge directions (each edge has length r in hex-distance):
    # Edge 1: (r,0) -> (0,r): direction (-1, 1), r steps
    # Edge 2: (0,r) -> (-r,r): direction (-1, 0), r steps  
    # Edge 3: (-r,r) -> (-r,0): direction (0, -1), r steps
    # Edge 4: (-r,0) -> (0,-r): direction (1, -1), r steps
    # Edge 5: (0,-r) -> (r,-r): direction (1, 0), r steps
    # Edge 6: (r,-r) -> (r,0): direction (0, 1), r steps
    
    directions = [
        (-1, 1),
        (-1, 0),
        (0, -1),
        (1, -1),
        (1, 0),
        (0, 1)
    ]
    
    r = 1
    while len(points) < n_points:
        # Start at (r, 0) for ring r
        a, b = r, 0
        for side in range(6):
            da, db = directions[side]
            for step in range(r):
                if len(points) >= n_points:
                    break
                points.append((a, b))
                a += da
                b += db
            if len(points) >= n_points:
                break
        r += 1
    
    result = [[a, b] for a, b in points[:n_points]]
    return result
