"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Generate points in a hexagonal spiral order on the triangular lattice.
    # Coordinates are (a, b) corresponding to a + b*omega.
    # Neighbors are (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1).
    
    points = [(0, 0)]
    
    # Directions for the 6 sides of the hexagon, in counter-clockwise order
    # starting from the positive a-axis (direction (1,0) is not a side direction, 
    # but the corners are defined by the cumulative sum of directions).
    # The side directions used to traverse the perimeter of a hexagon of radius k:
    # 1. (0, -1)
    # 2. (-1, 0)
    # 3. (-1, 1)
    # 4. (0, 1)
    # 5. (1, 0)
    # 6. (1, -1)
    
    side_dirs = [
        (0, -1),
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 0),
        (1, -1)
    ]
    
    k = 1
    while len(points) < 32:
        # Start of the ring k is at (k, 0)
        cur_a, cur_b = k, 0
        points.append((cur_a, cur_b))
        
        for i in range(6):
            da, db = side_dirs[i]
            for _ in range(k - 1):
                cur_a += da
                cur_b += db
                points.append((cur_a, cur_b))
                if len(points) >= 32:
                    break
            if len(points) >= 32:
                break
        k += 1
        
    # Convert to list of lists [a, b]
    return [[a, b] for a, b in points]
# EVOLVE-BLOCK-END
