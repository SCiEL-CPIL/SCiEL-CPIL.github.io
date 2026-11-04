import math

def construct():
    N = 11
    
    # Find minimal k
    k = 0
    while 1 + 3 * k * (k + 1) < N:
        k += 1
        
    if N == 1:
        return [[0.0, 0.0, 0.0]], (0.0, 0.0), 1.0, 0.0
        
    side_length = k * math.sqrt(3) + 1.0
    
    inner = []
    
    # Generate hexagons in rings
    # Ring 0
    inner.append([0.0, 0.0, 0.0])
    count = 1
    
    if count >= N:
        return inner, (0.0, 0.0), side_length, 0.0
        
    # Generate rings 1 to k
    for r in range(1, k + 1):
        # Get all axial coords in ring r
        ring_coords = []
        # Directions to traverse the ring
        # Start at (r, 0)
        q, r_ax = r, 0
        # We need to traverse the perimeter of the hexagon of radius r
        # The perimeter has 6 * r segments? No, 6 sides, each of length r.
        # Total 6r points.
        
        # Directions in axial coordinates:
        # E: (1, 0)
        # SE: (0, 1) -- Wait, let's verify standard axial.
        # Let's use the 6 neighbor offsets:
        dirs = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
        
        # Start at (r, 0)
        cur_q, cur_r = r, 0
        for i in range(6):
            for j in range(r):
                ring_coords.append((cur_q, cur_r))
                cur_q += dirs[i][0]
                cur_r += dirs[i][1]
                
        # Add to inner list until N is reached
        for q, r_ax in ring_coords:
            if count >= N:
                break
            # Convert axial to cartesian
            x = math.sqrt(3) * (q + r_ax / 2.0)
            y = 1.5 * r_ax
            inner.append([x, y, 0.0])
            count += 1
            
        if count >= N:
            break
            
    return inner, (0.0, 0.0), side_length, 0.0
