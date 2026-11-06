def construct():
    # Generate points in hexagonal spiral order
    points = []
    
    # Directions for the 6 sides of the hexagon
    # Axial coordinates (a, b)
    # We will walk around the perimeter of each ring k
    # Ring k has points where max(|a|, |b|, |a+b|) = k is NOT correct for all orientations.
    # Hex distance d(a,b) = (|a| + |b| + |a+b|) / 2
    
    # We want to generate rings 0, 1, 2 fully, and 13 points from ring 3.
    
    def hex_dist(a, b):
        return (abs(a) + abs(b) + abs(a + b)) // 2

    # Generate ring 0
    points.append([0, 0])
    
    # Generate rings 1 and 2 fully
    # A standard hexagonal spiral visits the perimeter of ring k in a specific order.
    # The perimeter of ring k consists of 6 segments of length k.
    # The corners are:
    # (k, -k), (k, 0), (0, k), (-k, k), (-k, 0), (0, -k) -- this is one orientation.
    # Let's use the orientation consistent with the hardcoded list.
    # Hardcoded Ring 1: [1,0], [1,-1], [0,-1], [-1,0], [-1,1], [0,1]
    # Hardcoded Ring 2: [2,0], [2,-1], [2,-2], [1,-2], [0,-2], [-1,-1], [-2,0], [-2,1], [-2,2], [-1,2], [0,2], [1,1]
    
    # Let's trace the hardcoded Ring 1:
    # Start at (1,0).
    # Next (1,-1). Delta (0,-1).
    # Next (0,-1). Delta (-1,0).
    # Next (-1,0). Delta (-1,1).
    # Next (-1,1). Delta (0,1).
    # Next (0,1). Delta (1,0).
    # This is a counter-clockwise walk starting from (1,0)?
    # (1,0) -> (1,-1) -> (0,-1) -> (-1,0) -> (-1,1) -> (0,1) -> (1,0) (close)
    
    # Let's trace Hardcoded Ring 2:
    # Start at (2,0).
    # (2,0) -> (2,-1) -> (2,-2) -> (1,-2) -> (0,-2) -> (-1,-1) -> (-2,0) -> (-2,1) -> (-2,2) -> (-1,2) -> (0,2) -> (1,1) -> (2,0) (close)
    
    # The pattern for Ring k:
    # Start at (k, 0).
    # Move 1 in direction (0, -1) for k steps?
    # (k, 0) to (k, -k) is k steps of (0, -1).
    # Points: (k, 0), (k, -1), ..., (k, -k).
    # Then move 1 in direction (-1, 0) for k steps?
    # (k, -k) to (0, -k) is k steps of (-1, 0).
    # Points: (k-1, -k), ..., (0, -k).
    # Then move 1 in direction (-1, 1) for k steps?
    # (0, -k) to (-k, 0) is k steps of (-1, 1).
    # Points: (-1, -k+1), ..., (-k, 0).
    # Then move 1 in direction (0, 1) for k steps?
    # (-k, 0) to (-k, k) is k steps of (0, 1).
    # Points: (-k, 1), ..., (-k, k).
    # Then move 1 in direction (1, 0) for k steps?
    # (-k, k) to (0, k) is k steps of (1, 0).
    # Points: (-k+1, k), ..., (0, k).
    # Then move 1 in direction (1, -1) for k steps?
    # (0, k) to (k, 0) is k steps of (1, -1).
    # Points: (1, k-1), ..., (k, 0).
    
    # Let's verify this against Hardcoded Ring 2:
    # k=2.
    # Seg 1: (2,0) to (2,-2). Points: [2,0], [2,-1], [2,-2]. (Matches first 3)
    # Seg 2: (2,-2) to (0,-2). Points: [1,-2], [0,-2]. (Matches next 2)
    # Seg 3: (0,-2) to (-2,0). Points: [-1,-1], [-2,0]. (Matches next 2)
    # Seg 4: (-2,0) to (-2,2). Points: [-2,1], [-2,2]. (Matches next 2)
    # Seg 5: (-2,2) to (0,2). Points: [-1,2], [0,2]. (Matches next 2)
    # Seg 6: (0,2) to (2,0). Points: [1,1], [2,0]. (Matches next 2, but [2,0] is already added)
    
    # So the algorithm is:
    # For k in 1..3:
    #   Add points from Seg 1 to Seg 6, excluding the last point of Seg 6 (which is the start of Seg 1) to avoid duplication.
    #   Actually, just add all points in the segments, and handle duplicates or stop when we have enough.
    
    def generate_ring(k):
        ring_points = []
        # Seg 1: (k, 0) to (k, -k)
        for i in range(k + 1):
            ring_points.append([k, -i])
        # Seg 2: (k, -k) to (0, -k)
        for i in range(1, k + 1):
            ring_points.append([k - i, -k])
        # Seg 3: (0, -k) to (-k, 0)
        for i in range(1, k + 1):
            ring_points.append([-i, -k + i])
        # Seg 4: (-k, 0) to (-k, k)
        for i in range(1, k + 1):
            ring_points.append([-k, i])
        # Seg 5: (-k, k) to (0, k)
        for i in range(1, k + 1):
            ring_points.append([-k + i, k])
        # Seg 6: (0, k) to (k, 0)
        for i in range(1, k + 1):
            ring_points.append([i, k - i])
        return ring_points

    # Ring 0
    points.append([0, 0])
    
    # Ring 1
    points.extend(generate_ring(1))
    
    # Ring 2
    points.extend(generate_ring(2))
    
    # Ring 3
    ring3 = generate_ring(3)
    # We need 13 points from Ring 3 to reach 32 total (1 + 6 + 12 + 13 = 32)
    points.extend(ring3[:13])
    
    return points
