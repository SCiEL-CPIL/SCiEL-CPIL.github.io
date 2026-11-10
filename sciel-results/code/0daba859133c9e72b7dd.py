import time

def construct():
    n = 16
    deadline = time.time() + 85.0  # safety margin before 90s kill
    
    # Construct K(8,8)
    # Partitions: 0-7 and 8-15
    edges = []
    for a in range(8):
        for b in range(8, 16):
            edges.append([a, b])
    
    # Verify K(8,8) is valid: diameter 2, every edge essential
    if _verify_k88(edges, n):
        return edges
    
    # Fallback: A star graph is a valid minimal solution (15 edges)
    # Diameter 2, all edges essential (removal disconnects leaves)
    return [[0, v] for v in range(1, n)]

def _verify_k88(edges, n):
    """
    Verify K(8,8) has diameter 2 and every edge is essential.
    
    Structural proof for K(8,8):
    - Diameter is 2: any two vertices in same part have distance 2 via any vertex in other part.
      Any two vertices in different parts have distance 1 (edge exists).
    - Edge essential: For edge (u,v) with u in L, v in R:
      After removal, no length-2 path between u and v exists (no vertex adjacent to both).
      Shortest path is length 3: u -> y (y in R, y!=v) -> x (x in L, x!=u) -> v.
      So diameter becomes 3 > 2. Edge is essential.
      
    We do a fast structural check rather than full BFS for each edge.
    """
    if len(edges) != 64:
        return False
    
    # Verify it's actually K(8,8) structure
    L = set(range(8))
    R = set(range(8, 16))
    edge_set = set()
    for u, v in edges:
        if u in L and v in R:
            edge_set.add((u, v))
        elif v in L and u in R:
            edge_set.add((v, u))
        else:
            return False
    
    if len(edge_set) != 64:
        return False
    
    # Check all 64 cross edges present
    for a in L:
        for b in R:
            if (a, b) not in edge_set:
                return False
    
    # Structural verification: diameter is 2
    # - Same part: distance 2 (via any vertex in other part, since |other| >= 2)
    # - Different parts: distance 1
    # This holds for K(8,8) with both parts size >= 2.
    
    # Edge essentiality: for each edge (u,v), u in L, v in R:
    # No common neighbor exists (L vertices not adjacent to each other, R vertices not adjacent to each other)
    # So no length-2 path between u and v after edge removal.
    # Distance becomes 3 (u->y->x->v for y in R\{v}, x in L\{u}).
    # Since |L|>=2 and |R|>=2, such y and x exist.
    # Diameter becomes 3 > 2, so edge is essential.
    
    return True
