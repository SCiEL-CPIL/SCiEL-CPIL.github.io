def construct():
    """
    Construct 32 distinct integer pairs [a, b] on the triangular lattice
    (points a + b*omega, omega = exp(i*pi/3)) with |a|, |b| <= 64,
    mutual distance at least 1, maximizing the number of pairs at distance exactly 1.
    
    Harborth's theorem: optimum = floor(3n - sqrt(12n - 3)) = 76 for n=32.
    A hexagonal spiral (compact cluster) achieves this.
    """
    import time
    start_time = time.time()
    deadline = 150 - 10  # 10s safety margin
    
    n = 32
    points = [(0, 0)]
    if n == 1:
        return points
    
    # Hexagonal spiral: directions in order
    # In axial coordinates (q, r) for triangular lattice:
    # omega = exp(i*pi/3) = 1/2 + i*sqrt(3)/2
    # Point = q + r*omega
    # Distance squared between (q1,r1) and (q2,r2):
    # Let dq = q1-q2, dr = r1-r2
    # (dq + dr/2)^2 + (dr*sqrt(3)/2)^2 = dq^2 + dq*dr + dr^2
    # So distance^2 = dq^2 + dq*dr + dr^2
    # Distance = 1 iff dq^2 + dq*dr + dr^2 = 1
    # This happens when (dq,dr) is one of: (1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)
    
    # The 6 neighbor offsets
    dirs = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    
    q, r = 0, 0
    step = 1
    dir_idx = 0
    
    while len(points) < n:
        # Check deadline
        if time.time() > deadline:
            break
        
        # Two segments of length 'step' in consecutive directions
        for _ in range(2):
            for _ in range(step):
                q += dirs[dir_idx][0]
                r += dirs[dir_idx][1]
                pt = (q, r)
                if pt not in points:
                    points.append(pt)
                    if len(points) == n:
                        return points
            dir_idx = (dir_idx + 1) % 6
        step += 1
    
    # If we didn't get enough points, pad with valid points (shouldn't happen for n=32)
    # But ensure we have exactly n points
    if len(points) < n:
        # Generate more points in spiral
        q, r = 0, 0
        step = 1
        dir_idx = 0
        # Reset and try again with larger step
        while len(points) < n and time.time() < deadline:
            for _ in range(2):
                for _ in range(step):
                    q += dirs[dir_idx][0]
                    r += dirs[dir_idx][1]
                    pt = (q, r)
                    if pt not in points:
                        points.append(pt)
                        if len(points) == n:
                            return points
                dir_idx = (dir_idx + 1) % 6
            step += 1
    
    return points[:n]
