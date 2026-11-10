import time

def construct():
    start_time = time.time()
    
    q = 5
    points = []
    # Affine points (1, x, y)
    for x in range(q):
        for y in range(q):
            points.append((1, x, y))
    # Points at infinity (0, 1, z)
    for z in range(q):
        points.append((0, 1, z))
    # Point at infinity (0, 0, 1)
    points.append((0, 0, 1))
    
    n = len(points) # 31
    
    # Precompute dot products mod q
    # dot[i][j] = 1 if points[i] dot points[j] == 0 mod q
    # We need the adjacency list of the polarity graph
    # Edge (i, j) exists if dot(p_i, p_j) == 0
    
    def dot(p, r):
        return (p[0]*r[0] + p[1]*r[1] + p[2]*r[2]) % q

    adj = [0] * n
    edges = []
    
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            if dot(pi, points[j]) == 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                edges.append([i, j])
                
    # We have 90 edges. Need to add 1 more.
    # Find the non-edge (u, v) that minimizes the number of new C4s.
    # New C4s formed by adding (u, v) = sum_{x in N(u)} |N(x) intersect N(v)|
    
    best_u = -1
    best_v = -1
    best_cost = float('inf')
    
    # List of non-edges
    for u in range(n):
        if time.time() - start_time > 140: # Safety check
            break
        for v in range(u + 1, n):
            if time.time() - start_time > 140:
                break
            # Check if (u, v) is already an edge
            if (adj[u] >> v) & 1:
                continue
            
            # Calculate cost of adding (u, v)
            # cost = sum_{x in N(u)} popcount(N(x) & N(v))
            cost = 0
            mask_u = adj[u]
            
            # Iterate through neighbors of u
            # Using bit manipulation to iterate set bits
            x_mask = mask_u
            while x_mask:
                # Get lowest set bit
                lb = x_mask & (-x_mask)
                x = lb.bit_length() - 1
                # Count common neighbors of x and v
                # N(x) is adj[x], N(v) is adj[v]
                cost += (adj[x] & adj[v]).bit_count()
                x_mask ^= lb
                
            if cost < best_cost:
                best_cost = cost
                best_u = u
                best_v = v
                if best_cost == 0:
                    # Can't get better than 0
                    break
        if best_cost == 0:
            break

    if best_u != -1:
        edges.append([best_u, best_v])
        
    # Ensure exactly 91 edges
    if len(edges) != 91:
        # Fallback if something went wrong (should not happen)
        # Just add any non-edge
        for i in range(n):
            for j in range(i + 1, n):
                if len(edges) >= 91:
                    break
                if not ((adj[i] >> j) & 1):
                    edges.append([i, j])
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
            if len(edges) >= 91:
                break

    return edges[:91]
