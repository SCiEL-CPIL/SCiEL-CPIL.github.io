def construct():
    import time
    start_time = time.time()
    deadline = start_time + 10.0  # Safety margin, construction is instant
    
    # Parameters for PG(2, 7)
    q = 7
    target_n = 40
    
    # Modular inverse for F_7
    inv = {1:1, 2:4, 3:5, 4:2, 5:3, 6:6}
    
    # Generate canonical points of PG(2, q)
    pts = []
    seen = set()
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize: first non-zero component is 1
                if x != 0:
                    nx = 1
                    ny = (y * inv[x]) % q
                    nz = (z * inv[x]) % q
                elif y != 0:
                    nx = 0
                    ny = 1
                    nz = (z * inv[y]) % q
                else:
                    nx = 0
                    ny = 0
                    nz = 1
                
                key = (nx, ny, nz)
                if key not in seen:
                    seen.add(key)
                    pts.append(key)
    
    # Use first target_n points
    if len(pts) < target_n:
        target_n = len(pts)
        
    selected_pts = pts[:target_n]
    n = target_n
    
    edges = []
    for i in range(n):
        p1 = selected_pts[i]
        for j in range(i + 1, n):
            p2 = selected_pts[j]
            dot = (p1[0]*p2[0] + p1[1]*p2[1] + p1[2]*p2[2]) % q
            if dot == 0:
                edges.append([i, j])
                
    return {'n': n, 'edges': edges}
