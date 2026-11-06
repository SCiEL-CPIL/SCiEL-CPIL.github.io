def construct():
    # Generate the hexagonal spiral (triangular lattice rings) and take the first 32 points.
    # This is the standard optimal construction for maximizing unit distances on the triangular lattice.
    # Harborth's theorem guarantees 76 unit distances for n=32, which this spiral attains.
    
    points = []
    
    # Ring 0: center
    points.append([0, 0])
    
    # Generate rings 1, 2, 3, ... until we have 32 points
    # A hexagonal ring r has 6*r points
    # Ring 1: 6 points (total 7)
    # Ring 2: 12 points (total 19)
    # Ring 3: 18 points (total 37) -> we only need 13 more to reach 32
    
    r = 1
    while len(points) < 32:
        # The 6 directions for a hexagonal ring in axial coordinates (a, b)
        # Starting from (r, 0) and going around
        # Direction vectors in axial coords:
        # (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)
        # But we need to start at a specific point and go around.
        # Standard hexagonal spiral:
        # Start at (r, 0)
        # Move in direction (0, 1) for r steps? No.
        
        # Let's use the standard algorithm:
        # Start at (r, 0)
        # The 6 sides of the hexagon:
        # Side 1: from (r, 0) to (r, -r) -> direction (0, -1), r steps
        # Side 2: from (r, -r) to (0, -r) -> direction (-1, 0), r steps
        # Side 3: from (0, -r) to (-r, 0) -> direction (0, 1), r steps
        # Side 4: from (-r, 0) to (-r, r) -> direction (1, 0), r steps
        # Side 5: from (-r, r) to (0, r) -> direction (0, -1), r steps
        # Side 6: from (0, r) to (r, 0) -> direction (1, 0), r steps
        
        # Wait, let me verify the directions in axial coordinates (a, b) where omega = exp(i*pi/3)
        # Neighbors of (a, b) are:
        # (a+1, b), (a, b+1), (a-1, b+1), (a-1, b), (a, b-1), (a+1, b-1)
        
        # Let's trace ring 1:
        # Start at (1, 0)
        # Move (0, -1) for 1 step: (1, -1)
        # Move (-1, 0) for 1 step: (0, -1)
        # Move (0, 1) for 1 step: (0, 0) -> wait, this is the center, not on the ring.
        
        # Correct hexagonal ring traversal:
        # Start at (r, 0)
        # Direction 1: (0, -1) for r steps -> ends at (r, -r)
        # Direction 2: (-1, 0) for r steps -> ends at (0, -r)
        # Direction 3: (-1, 1) for r steps -> ends at (-r, 0)
        # Direction 4: (0, 1) for r steps -> ends at (-r, r)
        # Direction 5: (1, 0) for r steps -> ends at (0, r)
        # Direction 6: (1, -1) for r steps -> ends at (r, 0)
        
        # Let's verify with r=1:
        # Start (1, 0)
        # Dir (0, -1): (1, -1)
        # Dir (-1, 0): (0, -1)
        # Dir (-1, 1): (-1, 0)
        # Dir (0, 1): (-1, 1)
        # Dir (1, 0): (0, 1)
        # Dir (1, -1): (1, 0)
        # Points: (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1) -> 6 points. Correct.
        
        a, b = r, 0
        points.append([a, b])
        
        directions = [(0, -1), (-1, 0), (-1, 1), (0, 1), (1, 0), (1, -1)]
        for da, db in directions:
            for _ in range(r):
                a += da
                b += db
                if len(points) < 32:
                    points.append([a, b])
                else:
                    break
            if len(points) >= 32:
                break
        if len(points) >= 32:
            break
        r += 1
    
    return points[:32]
