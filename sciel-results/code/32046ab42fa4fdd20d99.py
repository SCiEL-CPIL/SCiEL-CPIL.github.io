import time

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin (evaluator kills at 150s)
    
    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    all_edges_set = set(all_edges)
    
    # Adjacency as list of sets for fast updates
    def build_adj(edge_set):
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    # Fast Q3 detection using bitmasks
    # Precompute neighbor bitmasks for the current graph
    # But since graph changes, we update masks on the fly or rebuild?
    # Rebuilding 16 masks is cheap.
    
    def has_q3(adj):
        # Returns True if a Q3 is found, False otherwise.
        # For speed, we don't need to return the Q3 in the removal phase if we use a smart removal.
        # But we DO need the Q3 edges to remove an edge.
        # Let's return the edges of the first Q3 found.
        
        # Bitmask representation for adjacency
        masks = [0] * n
        for u in range(n):
            m = 0
            for v in adj[u]:
                m |= (1 << v)
            masks[u] = m
            
        # Structure of Q3:
        # Vertices 0,1,2,3,4,5,6,7
        # Edges: (0,1),(1,2),(2,3),(3,0) - Square 1
        #        (4,5),(5,6),(6,7),(7,4) - Square 2
        #        (0,4),(1,5),(2,6),(3,7) - Connections
        # Actually, it's easier to use the "center" logic from the seed but optimized.
        
        # A Q3 is determined by two disjoint triangles? No, it's bipartite.
        # It's determined by 4 disjoint edges? No.
        
        # Let's stick to the efficient search from the seed but optimize the inner loops.
        # The seed's find_q3 was reasonably fast.
        # Optimization: Pre-check degrees? Q3 vertices have degree >= 3 in the subgraph.
        
        for v in range(n):
            if len(adj[v]) < 3:
                continue
            neighbors = list(adj[v])
            # Pick 3 non-adjacent neighbors a, b, c
            # In Q3, v is connected to 3 vertices that form an independent set.
            # Let's iterate over triples of neighbors.
            # To reduce checks, we can require a < b < c? No, order doesn't matter for the set.
            
            # We need a, b, c to be pairwise non-adjacent.
            # Then x = common neighbor of a,b (excluding v, c)
            # y = common neighbor of a,c (excluding v, b)
            # z = common neighbor of b,c (excluding v, a)
            # x, y, z must be pairwise non-adjacent?
            # In Q3:
            # v is connected to a,b,c.
            # a is connected to v, x, y.
            # b is connected to v, x, z.
            # c is connected to v, y, z.
            # x is connected to a, b, w.
            # y is connected to a, c, w.
            # z is connected to b, c, w.
            # w is connected to x, y, z.
            
            # So:
            # x is in N(a) ∩ N(b) \ {v, c}
            # y is in N(a) ∩ N(c) \ {v, b}
            # z is in N(b) ∩ N(c) \ {v, a}
            # w is in N(x) ∩ N(y) ∩ N(z) \ {v, a, b, c, x, y, z}
            
            # And we must check:
            # a,b,c pairwise non-adjacent.
            # x,y,z pairwise non-adjacent.
            # w not adjacent to a,b,c.
            # (Edges a-x, b-x, a-y, c-y, b-z, c-z, x-w, y-w, z-w, v-a, v-b, v-c are the 12 edges).
            
            for i in range(len(neighbors)):
                a = neighbors[i]
                for j in range(i+1, len(neighbors)):
                    b = neighbors[j]
                    if b in adj[a]:
                        continue
                    for k in range(j+1, len(neighbors)):
                        c = neighbors[k]
                        if c in adj[a] or c in adj[b]:
                            continue
                        
                        # Find candidates for x, y, z
                        # x is neighbor of a and b, not v, not c
                        x_cands = adj[a] & adj[b]
                        x_cands.discard(v)
                        x_cands.discard(c)
                        
                        y_cands = adj[a] & adj[c]
                        y_cands.discard(v)
                        y_cands.discard(b)
                        
                        z_cands = adj[b] & adj[c]
                        z_cands.discard(v)
                        z_cands.discard(a)
                        
                        if not x_cands or not y_cands or not z_cands:
                            continue
                            
                        for x in x_cands:
                            for y in y_cands:
                                if x == y:
                                    continue
                                if y in adj[x]:
                                    continue
                                for z in z_cands:
                                    if x == z or y == z:
                                        continue
                                    if z in adj[x] or z in adj[y]:
                                        continue
                                    
                                    # Check w
                                    w_cands = adj[x] & adj[y] & adj[z]
                                    forbidden = {v, a, b, c, x, y, z}
                                    w_cands -= forbidden
                                    
                                    if not w_cands:
                                        continue
                                    
                                    for w in w_cands:
                                        # Check w not adjacent to a, b, c
                                        if w in adj[a] or w in adj[b] or w in adj[c]:
                                            continue
                                        
                                        # Found a Q3
                                        return [
                                            (v, a), (v, b), (v, c),
                                            (a, x), (b, x),
                                            (a, y), (c, y),
                                            (b, z), (c, z),
                                            (x, w), (y, w), (z, w)
                                        ]
        return None

    def remove_edge_from_q3(adj, q3_edges):
        # Heuristic: Remove edge with minimum degree sum
        min_score = float('inf')
        best_e = None
        for u, v in q3_edges:
            score = len(adj[u]) + len(adj[v])
            if score < min_score:
                min_score = score
                best_e = (min(u, v), max(u, v))
        return best_e

    best_count = 0
    best_edges = []
    
    # Initial dense graph
    # Start with a random graph of density ~0.7?
    # Or start with K16 and remove?
    # Starting with K16 (120 edges) and removing might be faster than adding?
    # Let's try starting with a random graph of p=0.8.
    
    import random
    random.seed(123)
    
    for restart in range(50):
        if time.time() > deadline - 2:
            break
            
        # Initialize
        current_edges = set()
        for e in all_edges:
            if random.random() < 0.7:
                current_edges.add(e)
                
        adj = build_adj(current_edges)
        
        # Phase 1: Remove Q3s
        # Use a counter to avoid infinite loops
        steps = 0
        while steps < 1000:
            if time.time() > deadline - 2:
                break
            q3_edges = has_q3(adj)
            if q3_edges is None:
                break
            
            e_to_remove = remove_edge_from_q3(adj, q3_edges)
            u, v = e_to_remove
            if (u, v) in current_edges:
                current_edges.remove((u, v))
                adj[u].remove(v)
                adj[v].remove(u)
            else:
                break
            steps += 1
            
        # Phase 2: Hill Climbing
        # Try adding missing edges
        missing = [e for e in all_edges if e not in current_edges]
        random.shuffle(missing)
        
        for e in missing:
            if time.time() > deadline - 2:
                break
            u, v = e
            # Tentatively add
            current_edges.add(e)
            adj[u].add(v)
            adj[v].add(u)
            
            q3_edges = has_q3(adj)
            
            if q3_edges is not None:
                # Check if the added edge is in the Q3
                # If so, we can't keep it (or we could remove another edge, but that's more complex)
                # For simplicity, reject if it creates any Q3
                # Optimization: Only reject if e is in the Q3?
                # If e is NOT in the Q3, then the Q3 existed before?
                # No, we just removed all Q3s in Phase 1.
                # So if a Q3 appears, it must involve the new edge e.
                # Wait, is that true?
                # Yes, because Phase 1 ended with no Q3s.
                # Adding one edge can only create Q3s that contain that edge.
                # So we can just check if e is in q3_edges.
                # Actually, if e is not in q3_edges, then q3_edges must have been there before, which is a contradiction.
                # So we can just reject.
                
                current_edges.remove(e)
                adj[u].remove(v)
                adj[v].remove(u)
            else:
                # Keep the edge
                pass
                
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = list(current_edges)
            
        # Phase 3: Local Optimization (Swap edges?)
        # Try swapping an edge in with an edge out?
        # Or just restart with a slightly modified initial graph?
        
    return [[u, v] for u, v in best_edges]
