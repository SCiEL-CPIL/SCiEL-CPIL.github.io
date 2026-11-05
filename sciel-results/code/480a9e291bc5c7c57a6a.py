import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin

    def time_check():
        return time.time() > deadline

    # 1. Generate Base Paley(29)
    p = 29
    squares = {x * x % p for x in range(1, p)}
    
    # Colors for K_29
    # Lexicographic order (i, j) with i < j
    base_colors = []
    for i in range(p):
        for j in range(i + 1, p):
            diff = (j - i) % p
            if diff in squares:
                base_colors.append(0)
            else:
                base_colors.append(1)
    
    # 2. Attempt to extend to n=30
    n = 30
    num_edges = n * (n - 1) // 2
    
    # Edge index map
    edge_idx = {}
    idx_edge = [None] * num_edges
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_idx[(i, j)] = idx
            idx_edge[idx] = (i, j)
            idx += 1

    def build_structures(colors):
        """
        Builds adjacency bitmasks and common neighbor bitmasks.
        adj[c][u] is a bitmask of neighbors of u in color c.
        common[c][u][v] is a bitmask of common neighbors of u and v in color c.
        """
        adj = [[0, 0] for _ in range(n)]
        for e_idx, c in enumerate(colors):
            u, v = idx_edge[e_idx]
            adj[u][c] |= (1 << v)
            adj[v][c] |= (1 << u)
        
        # Precompute common neighbors
        # common[c] is a list of lists of bitmasks
        common = [[None] * n for _ in range(n)]
        for c in range(2):
            for u in range(n):
                for v in range(u + 1, n):
                    common[u][v] = adj[u][c] & adj[v][c]
                    common[v][u] = common[u][v]
        return adj, common

    def find_violation(adj, common):
        """
        Returns a list of edges (indices) involved in a K2,2,2 violation, or None.
        """
        for c in range(2):
            for u in range(n):
                for v in range(u + 1, n):
                    # Check if (u,v) is color c
                    if not (adj[u][c] & (1 << v)):
                        continue
                    
                    S = common[u][v]
                    if S == 0:
                        continue
                    
                    # Check if subgraph induced by S in color c contains K2,2
                    # K2,2 exists if there are a,b in S such that 
                    # they are connected in color c and have >=2 common neighbors in S.
                    # Actually, K2,2 in S means there exist a,b in S such that
                    # N(a) in S and N(b) in S intersect in at least 2 vertices?
                    # No. K2,2 is a complete bipartite graph with parts {a,b} and {x,y}.
                    # So we need a,b in S such that there exist x,y in S \ {a,b}
                    # such that x,y are adjacent to both a and b in color c.
                    # This is equivalent to: common[a][b] & S has at least 2 bits.
                    # AND a,b must be in S.
                    # Do a,b need to be connected to each other?
                    # K2,2 is a subgraph. The vertices are a,b,x,y.
                    # Edges: (a,x), (a,y), (b,x), (b,y).
                    # a and b do NOT need to be connected.
                    # x and y do NOT need to be connected.
                    # So we just need a,b in S such that |N_c(a) \cap N_c(b) \cap S| >= 2.
                    
                    mask = S
                    while mask:
                        lb = mask & (-mask)
                        a = lb.bit_length() - 1
                        mask ^= lb
                        
                        mask_b = S
                        while mask_b:
                            lb_b = mask_b & (-mask_b)
                            b = lb_b.bit_length() - 1
                            mask_b ^= lb_b
                            
                            if a == b:
                                continue
                            
                            T = common[a][b] & S
                            # T contains common neighbors of a and b in color c, restricted to S.
                            # We need at least 2 such neighbors.
                            if T & (T - 1): # at least 2 bits
                                # Violation found!
                                # Vertices: u, v, a, b, and two from T.
                                # Edges in K2,2,2:
                                # (u,v) is color c.
                                # (u,a), (u,b), (u,t1), (u,t2) are color c (since a,b,t1,t2 in S=N(u)capN(v))
                                # (v,a), (v,b), (v,t1), (v,t2) are color c.
                                # (a,t1), (a,t2), (b,t1), (b,t2) are color c (since t1,t2 in N(a)capN(b))
                                # We can return the edge (a,b) or (u,v) or others.
                                # Let's return the edge (a,b) if it exists in the graph (it does, all pairs exist).
                                # Flipping (a,b) might break the K2,2 structure.
                                i, j = (a, b) if a < b else (b, a)
                                return [edge_idx[(i, j)]]
        return None

    best_n = 29
    best_colors = base_colors[:]

    # Heuristics for the 29 edges incident to vertex 29
    # We try several patterns
    patterns = []
    # All 0, All 1
    patterns.append([0]*29)
    patterns.append([1]*29)
    # Alternating
    patterns.append([i % 2 for i in range(29)])
    # Random
    for _ in range(20):
        patterns.append([random.randint(0, 1) for _ in range(29)])
    
    # Also try patterns based on Paley structure of vertex 0?
    # Vertex 29 is new. Let's try to mimic vertex 0's connections?
    # In Paley(29), vertex 0 connects to squares.
    # squares = {1, 4, 9, 16, 25, 7, 20, 6, 23, 13, 5, 28, 12, 3, 22, 11, 27, 10, 26, 15, 8, 21, 2, 19, 17, 18, 14, 24, 18?}
    # Let's just use random and simple patterns.

    for pattern in patterns:
        if time_check():
            break
            
        colors_list = base_colors[:] + pattern
        
        # Local Search
        # We will run for a fixed number of iterations or until timeout
        # To allow multiple patterns to be tried, we limit iterations per pattern
        max_iters_per_pattern = 2000
        
        for it in range(max_iters_per_pattern):
            if time_check():
                break
                
            adj, common = build_structures(colors_list)
            
            violation_edges = find_violation(adj, common)
            
            if violation_edges is None:
                # Success!
                best_n = 30
                best_colors = colors_list[:]
                break
            
            # Flip a random edge from the violation
            # We can also try flipping other edges in the K222 structure
            # But we only returned one edge.
            # Let's flip that edge.
            e_idx = violation_edges[0]
            colors_list[e_idx] ^= 1
            
            # Optional: Simulated Annealing step
            # If stuck, we might need to flip random edges.
            # But targeted flipping is usually better for constraint satisfaction.
            # If we loop, we might oscillate.
            # We can keep a history or just rely on multiple restarts.

        if best_n == 30:
            break

    if best_n == 30:
        return {'n': 30, 'colors': best_colors}
    else:
        return {'n': 29, 'colors': best_colors}
