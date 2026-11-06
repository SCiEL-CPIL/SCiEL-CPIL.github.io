def construct():
    # Inverses mod 5
    inv = {1: 1, 2: 3, 3: 2, 4: 4}
    
    points = []
    seen = set()
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Canonicalize: first non-zero component is 1
                if x != 0:
                    ix = inv[x]
                    pt = (1, (y * ix) % 5, (z * ix) % 5)
                elif y != 0:
                    iy = inv[y]
                    pt = (0, 1, (z * iy) % 5)
                else:
                    pt = (0, 0, 1)
                
                if pt not in seen:
                    seen.add(pt)
                    points.append(pt)
    
    n = len(points)
    
    adj = [[False] * n for _ in range(n)]
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            p_i = points[i]
            p_j = points[j]
            dot = (p_i[0] * p_j[0] + p_i[1] * p_j[1] + p_i[2] * p_j[2]) % 5
            if dot == 0:
                adj[i][j] = True
                adj[j][i] = True
                edges.append((i, j))
                
    # Find non-edge (u, v) and its common neighbor w
    target_u = -1
    target_v = -1
    target_w = -1
    
    for u in range(n):
        for v in range(u + 1, n):
            if not adj[u][v]:
                for w in range(n):
                    if w != u and w != v and adj[u][w] and adj[v][w]:
                        target_u = u
                        target_v = v
                        target_w = w
                        break
                if target_w != -1:
                    break
        if target_w != -1:
            break
            
    if target_w == -1:
        # Fallback, should not happen
        return [[i, j] for i, j in edges[:91]]
        
    edges_to_remove = []
    protected = {
        (min(target_u, target_w), max(target_u, target_w)),
        (min(target_v, target_w), max(target_v, target_w))
    }
    
    for e in edges:
        if e not in protected:
            edges_to_remove.append(e)
            if len(edges_to_remove) == 3:
                break
                
    final_edges = []
    for e in edges:
        if e not in edges_to_remove:
            final_edges.append([e[0], e[1]])
            
    final_edges.append([target_u, target_v])
    
    return final_edges
