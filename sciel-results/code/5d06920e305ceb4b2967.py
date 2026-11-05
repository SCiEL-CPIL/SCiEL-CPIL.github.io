import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Hex spiral directions
    # (da, db)
    dirs = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    
    points = [(0, 0)]
    
    # Generate rings
    ring = 1
    while len(points) < 32:
        # Start of ring: (ring, 0)
        a, b = ring, 0
        # Traverse the ring
        for i in range(6):
            da, db = dirs[i]
            for step in range(ring):
                if len(points) >= 32:
                    break
                points.append((a, b))
                a += da
                b += db
            if len(points) >= 32:
                break
        ring += 1
        if time.time() > deadline:
            break
            
    # Ensure exactly 32 distinct points
    # The spiral generation ensures distinctness
    if len(points) > 32:
        points = points[:32]
        
    # Convert to list of lists
    return [[a, b] for a, b in points]
