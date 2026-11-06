import itertools
import time

def construct():
    start_time = time.time()
    deadline = 150.0  # 150s limit, keep margin
    
    q = 5
    n = q * q + q + 1  # 31
    
    # Generate points of PG(2, q)
    # Points are 1D subspaces of GF(q)^3
    # Represent by normalized vectors (first non-zero is 1)
    points = []
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize
                # Find first non-zero
                if x != 0:
                    inv = pow(x, q - 2, q)
                    px = 1
                    py = (y * inv) % q
                    pz = (z * inv) % q
                elif y != 0:
                    inv = pow(y, q - 2, q)
                    px = 0
                    py = 1
                    pz = (z * inv) % q
                else:
                    px = 0
                    py = 0
                    pz = 1
                
                pt = (px, py, pz)
                if pt not in points:
                    points.append(pt)
    
    # Map points to indices 0..30
    point_to_idx = {p: i for i, p in enumerate(points)}
    
    # Build adjacency
    # P_i adjacent to P_j if dot product is 0
    adj = [[False] * n for _ in range(n)]
    edges = []
    
    for i in range(n):
        for j in range(i + 1, n):
            p1 = points[i]
            p2 = points[j]
            dot = (p1[0]*p2[0] + p1[1]*p2[1] + p1[2]*p2[2]) % q
            if dot == 0:
                adj[i][j] = True
                adj[j][i] = True
                edges.append([i, j])
    
    current_edges = len(edges)
    
    # If we have less than 90 edges, add more greedily without creating C4
    # If we have exactly 90, add 1 best edge
    # If we have more, we have a problem (should not happen for ex)
    
    target_edges = 90
    
    if current_edges < target_edges:
        # Greedy addition of edges to reach 90, minimizing C4s (which is 0 so far)
        # We just need any C4-free graph with 90 edges.
        # Since the polarity graph is C4-free, adding edges might create C4s.
        # However, if current_edges < 90, it means the polarity graph wasn't maximal?
        # Or my construction missed some?
        # Let's try to add edges that don't create C4s.
        
        def count_c4s_after_add(adj, u, v):
            # Returns number of C4s created by adding edge (u,v)
            # A C4 is u-w-x-v-u. w and x are common neighbors of u and v.
            # Number of such pairs (w,x) is C(common_neighbors, 2).
            common = [k for k in range(n) if adj[u][k] and adj[v][k]]
            return len(common) * (len(common) - 1) // 2

        # We want to reach 90 edges.
        # If we are below 90, we can add edges.
        # But if we add an edge that creates C4s, we lose the "C4-free" property of the base.
        # The task asks for min C4s in the final 91-edge graph.
        # If we start with <90 C4-free, we add edges.
        # Best strategy: add edges with min C4 creation.
        
        while len(edges) < target_edges:
            best_edge = None
            best_cost = float('inf')
            
            for u in range(n):
                for v in range(u + 1, n):
                    if not adj[u][v]:
                        cost = count_c4s_after_add(adj, u, v)
                        if cost < best_cost:
                            best_cost = cost
                            best_edge = (u, v)
                        if time.time() > deadline - 5:
                            break
                if time.time() > deadline - 5:
                    break
            if time.time() > deadline - 5:
                break
            
            if best_edge is None:
                break
                
            u, v = best_edge
            adj[u][v] = True
            adj[v][u] = True
            edges.append([u, v])
            
    # Now we have at least 90 edges (hopefully exactly 90 if polarity worked)
    # We need exactly 91 edges.
    # Add one more edge with minimum C4 creation.
    
    if len(edges) < 91:
        best_edge = None
        best_cost = float('inf')
        
        for u in range(n):
            for v in range(u + 1, n):
                if not adj[u][v]:
                    cost = count_c4s_after_add(adj, u, v)
                    if cost < best_cost:
                        best_cost = cost
                        best_edge = (u, v)
                    if time.time() > deadline - 5:
                        break
            if time.time() > deadline - 5:
                break
        
        if best_edge is not None:
            u, v = best_edge
            adj[u][v] = True
            adj[v][u] = True
            edges.append([u, v])
            
    return edges[:91]
