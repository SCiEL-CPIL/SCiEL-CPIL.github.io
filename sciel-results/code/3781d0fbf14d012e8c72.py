import time
import itertools

def _count_c4s(adj, n):
    """Count C4s using adjacency matrix approach."""
    import numpy as np
    A = np.zeros((n, n), dtype=np.int64)
    for u in range(n):
        for v in adj[u]:
            A[u, v] = 1
    # Number of C4s = 0.5 * sum over all pairs (u,v) of (common neighbors of u and v) choose 2
    # More precisely: for each pair (u,v), let c = |N(u) ∩ N(v)|, then C4s through u-v as diagonal = C(c,2)
    # Total C4s = 0.5 * sum_{u<v} C(common_neighbors(u,v), 2)
    total = 0
    for u in range(n):
        for v in range(u + 1, n):
            common = len(adj[u] & adj[v])
            if common >= 2:
                total += common * (common - 1) // 2
    return total

def _count_c4s_fast(adj, n):
    """Faster C4 count using numpy."""
    import numpy as np
    A = np.zeros((n, n), dtype=np.int64)
    for u in range(n):
        for v in adj[u]:
            A[u, v] = 1
    # C4 count: for each pair of vertices (u,v), count paths of length 2 between them
    # M = A @ A, M[u,v] = number of length-2 paths from u to v
    M = A @ A
    # For each pair (u,v), C4s with u,v as opposite corners = C(M[u,v], 2)
    # Each C4 is counted twice (once for each pair of opposite corners)
    total = 0
    for u in range(n):
        for v in range(u + 1, n):
            m = M[u, v]
            if m >= 2:
                total += m * (m - 1) // 2
    return total // 2  # Each C4 counted twice (two pairs of opposite corners)

def construct():
    """
    Returns a list of 91 edges [u, v] on vertices 0..30 minimizing the number of C4s.
    Uses the polarity graph of order 5 as a seed (90 edges, 0 C4s) and adds
    the single edge that creates the fewest C4s, then performs local search.
    """
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    q = 5
    # Generate vertices of PG(2, q)
    vertices = []
    for i in range(q):
        for j in range(q):
            vertices.append((1, i, j))
    for k in range(q):
        vertices.append((0, 1, k))
    vertices.append((0, 0, 1))
    
    n = len(vertices)  # 31
    
    # Build adjacency list for the polarity graph
    adj = [set() for _ in range(n)]
    edges = []
    
    for i in range(n):
        for j in range(i + 1, n):
            u = vertices[i]
            v = vertices[j]
            dot = (u[0]*v[0] + u[1]*v[1] + u[2]*v[2]) % q
            if dot == 0:
                adj[i].add(j)
                adj[j].add(i)
                edges.append([i, j])
    
    # We have 90 edges. Find the non-edge that adds the fewest C4s.
    best_cost = float('inf')
    best_edge = None
    
    for u in range(n):
        for v in range(u + 1, n):
            if v in adj[u]:
                continue
            A = adj[u]
            B = adj[v]
            cost = 0
            for x in A:
                cost += len(adj[x] & B)
            if cost < best_cost:
                best_cost = cost
                best_edge = [u, v]
    
    edges.append(best_edge)
    adj[best_edge[0]].add(best_edge[1])
    adj[best_edge[1]].add(best_edge[0])
    
    # Current C4 count
    current_c4 = _count_c4s(adj, n)
    best_c4 = current_c4
    
    # Local search: try edge swaps to reduce C4 count
    # An edge swap: remove edge (a,b) and add edge (c,d) where (c,d) is a non-edge
    # We need to maintain exactly 91 edges
    
    # Build set of current edges for O(1) lookup
    edge_set = set(tuple(sorted(e)) for e in edges)
    
    # Precompute all possible non-edges
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try all edge swaps: for each current edge (a,b), for each non-edge (c,d),
        # check if removing (a,b) and adding (c,d) reduces C4 count
        # This is O(E * (V^2)) which is too slow for 91 * ~400 = 36400 swaps
        # Instead, use incremental C4 counting
        
        best_delta = 0  # We want negative delta (reduction)
        best_swap = None
        
        for ei, (a, b) in enumerate(edges):
            if time.time() > deadline:
                break
            # For each non-edge (c, d), consider swapping
            # Delta = C4s_after - C4s_before
            # C4s affected: those involving edge (a,b) or edge (c,d)
            # This is complex; use a simpler approach with periodic full recount
            
            # Try swapping (a,b) with each non-edge
            for c in range(n):
                if c == a or c == b:
                    continue
                for d in range(c + 1, n):
                    if d == a or d == b:
                        continue
                    if c == d:
                        continue
                    key = (c, d) if c < d else (d, c)
                    if key in edge_set:
                        continue
                    
                    # Compute delta in C4 count
                    # Remove (a,b): lose C4s that use this edge
                    # Add (c,d): gain C4s that use this edge
                    # But there's interaction if the new C4s use vertices affected by removal
                    
                    # Count C4s lost by removing (a,b):
                    # C4s containing edge (a,b): for each pair (x,y) where x~a, x~b, y~a, y~b, x!=y
                    # Actually, C4s containing edge (a,b) are paths a-x-y-b where x~a, y~b, x~y
                    lost = 0
                    for x in adj[a]:
                        if x == b:
                            continue
                        for y in adj[b]:
                            if y == a or y == x:
                                continue
                            if x in adj[y]:
                                lost += 1
                    
                    # Count C4s gained by adding (c,d):
                    # C4s containing edge (c,d): paths c-x-y-d where x~c, y~d, x~y
                    # But we need to account for the fact that (a,b) is removed
                    # For simplicity, approximate: count C4s in current graph minus those using (a,b)
                    gained = 0
                    for x in adj[c]:
                        if x == d:
                            continue
                        for y in adj[d]:
                            if y == c or y == x:
                                continue
                            if x in adj[y]:
                                # Check if this C4 uses edge (a,b) - if so, it's not really gained
                                # The C4 is c-x-y-d. It uses (a,b) only if {a,b} ⊆ {c,x,y,d}
                                # Since (a,b) is being removed, we need to be careful
                                # For now, count it and rely on the fact that if it uses (a,b),
                                # the actual gain is less
                                gained += 1
                    
                    delta = gained - lost
                    if delta < best_delta:
                        best_delta = delta
                        best_swap = (ei, c, d)
        
        if best_swap is not None and best_delta < 0:
            ei, c, d = best_swap
            a, b = edges[ei]
            # Remove edge (a,b)
            edges[ei] = [c, d]
            adj[a].remove(b)
            adj[b].remove(a)
            adj[c].add(d)
            adj[d].add(c)
            edge_set.discard((a, b) if a < b else (b, a))
            edge_set.add((c, d) if c < d else (d, c))
            
            current_c4 = _count_c4s(adj, n)
            if current_c4 < best_c4:
                best_c4 = current_c4
                improved = True
    
    # Ensure exactly 91 edges
    assert len(edges) == 91, f"Expected 91 edges, got {len(edges)}"
    assert len(set(tuple(sorted(e)) for e in edges)) == 91, "Edges are not distinct"
    
    return edges
