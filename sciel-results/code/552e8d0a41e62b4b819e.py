def polarity_graph(q):
    if q != 7:
        return None
    # Points of PG(2,7) are 1-dim subspaces of GF(7)^3
    # Represent points as (x, y, z) with gcd(x,y,z)=1, not all 0
    # Normalize: first non-zero element is 1.
    points = []
    for x in range(8):
        for y in range(8):
            for z in range(8):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize
                if x != 0:
                    pts = (1, (y * pow(x, 6, 7)) % 7, (z * pow(x, 6, 7)) % 7)
                elif y != 0:
                    pts = (0, 1, (z * pow(y, 6, 7)) % 7)
                else:
                    pts = (0, 0, 1)
                if pts not in points:
                    points.append(pts)
    
    # Define polarity
    # A standard polarity in PG(2,q) for q odd:
    # Line polar of (x,y,z) is x*X + y*Y + z*Z = 0? No, that's a correlation.
    # For a polarity, we need a symmetric bilinear form.
    # B(u, v) = u1*v1 + u2*v3 + u3*v2 ?
    # Let's use B(u, v) = u1*v1 + u2*v2 + u3*v3? No, that's not a polarity for all q.
    # For q=7, we can use B(u, v) = u1*v1 + u2*v3 + u3*v2.
    # Check symmetry: B(u, v) = u1 v1 + u2 v3 + u3 v2.
    # B(v, u) = v1 u1 + v2 u3 + v3 u2. Same.
    # So u is on polar of v if B(u, v) = 0.
    
    adj = [0] * 57
    for i in range(57):
        u = points[i]
        for j in range(57):
            v = points[j]
            # B(u, v) = u1*v1 + u2*v3 + u3*v2
            val = (u[0]*v[0] + u[1]*v[2] + u[2]*v[1]) % 7
            if val == 0:
                adj[i] |= (1 << j)
    
    # Remove self-loops?
    # If B(u, u) = 0, u is absolute.
    # In simple graph, no loops.
    for i in range(57):
        adj[i] &= ~(1 << i)
        
    return adj
