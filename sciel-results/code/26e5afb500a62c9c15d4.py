import time
import random

def construct():
    start_time = time.time()
    DEADLINE = start_time + 145.0  # Safety margin

    def gen_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [0] * n
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.append((u, v))
        return adj, edges

    def popcount(x):
        return x.bit_count()

    def bronk_max_clique(P, X, adj, deadline):
        """
        Find a maximum clique in the graph defined by adj, restricted to vertices in P.
        Returns a bitmask of the vertices in the max clique found.
        Uses Bron-Kerbosch with pivot.
        """
        best_clique = 0
        best_size = 0
        
        # P and X are bitmasks
        def bk(P, X, R):
            nonlocal best_clique, best_size
            
            if time.time() > deadline:
                return
            
            if not P and not X:
                if R.bit_count() > best_size:
                    best_size = R.bit_count()
                    best_clique = R
                return
            
            # Choose pivot u from P ∪ X to minimize branching
            # u should maximize |P ∩ N(u)|
            union = P | X
            if not union:
                return
            
            # Find pivot
            pivot = 0
            max_common = -1
            temp = union
            while temp:
                u = (temp & -temp).bit_length() - 1
                temp &= temp - 1
                common = P & adj[u]
                count = common.bit_count()
                if count > max_common:
                    max_common = count
                    pivot = u
            
            # Branch on vertices in P \ N(pivot)
            candidates = P & ~adj[pivot]
            
            # If no candidates, return (all vertices in P are in N(pivot), 
            # so any maximal clique containing a vertex in P must also contain pivot? 
            # No, standard BK logic: if P \ N(pivot) is empty, 
            # then any maximal clique in P is a subset of N(pivot) ∪ {pivot}? 
            # Actually, if P is subset of N(pivot), we don't need to branch on P directly?
            # Standard optimization: if P is subset of N(pivot), we can return? 
            # No, that's for counting. For finding max clique, we still need to explore.
            # However, if candidates is empty, it means P is subset of N(pivot).
            # In that case, we can just recurse on P? No.
            # If candidates is empty, it implies P is contained in N(pivot).
            # This doesn't mean we stop. We just mean the branching set is empty?
            # Wait, if candidates is empty, the loop below doesn't run. 
            # But we must have found a clique? 
            # If P is not empty, and candidates is empty, then P is subset of N(pivot).
            # This is a valid state. We should still explore.
            # Actually, the standard BK says: iterate v in P \ N(u).
            # If this set is empty, we return. Why?
            # Because any maximal clique containing a vertex from P must also contain u?
            # No. The theorem is: every maximal clique in G[P] contains a vertex from P \ N(u) OR is a clique in G[P \ {v}]...
            # Let's stick to the standard implementation. If candidates is empty, we return.
            # This is correct because if P is subset of N(u), then u is connected to all in P.
            # But u is in X (excluded) or P? 
            # If u is in P, then u is a candidate? No, candidates = P \ N(u). u is not in N(u).
            # So if u in P, u is in candidates.
            # If u in X, u is not in P.
            
            # Let's just iterate candidates
            temp_cand = candidates
            while temp_cand:
                if time.time() > deadline:
                    return
                
                v = (temp_cand & -temp_cand).bit_length() - 1
                temp_cand &= temp_cand - 1
                
                # Recurse
                bk(P & adj[v], X, R | (1 << v))
                
                P &= ~(1 << v)
                X |= (1 << v)

        bk(P, X, 0)
        return best_clique

    def solve_instance(n, p, seed, deadline):
        adj, edges = gen_graph(n, p, seed)
        if not edges:
            return []
        
        best_cliques = [[u, v] for u, v in edges]
        best_count = len(best_cliques)
        
        rng_master = random.Random(12345)
        
        # Determine number of restarts
        # We have ~140s. Instance 1 (n=32) is fast. Instance 2 (n=64) is slower.
        # Let's allocate time based on n.
        
        elapsed = time.time() - start_time
        remaining_time = deadline - start_time - elapsed
        
        # Heuristic for number of restarts
        # n=32: very fast, can do many restarts
        # n=64: slower, fewer restarts
        if n <= 32:
            num_restarts = min(200, max(20, int(remaining_time / 0.1)))
        else:
            num_restarts = min(100, max(10, int(remaining_time / 0.3)))
            
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            rng = random.Random(rng_master.randint(0, 2**31))
            
            # Copy adjacency
            rem_adj = adj[:]
            remaining_edges_count = len(edges)
            
            # Track remaining edges as a set of tuples for easy removal? 
            # Or just rebuild adj? 
            # Removing edges from bitmasks is fast.
            # We need to know when to stop. 
            # We can just loop while any adj[i] is non-zero.
            
            cliques = []
            
            # Optimization: precompute list of vertices
            vertices = list(range(n))
            
            # Strategy: Alternate between "Max Degree + BK" and "Min Common + BK"
            # Or just use "Max Degree + BK" which is robust for dense graphs.
            
            while True:
                if time.time() > deadline:
                    break
                
                # Check if any edges remain
                has_edge = False
                for i in range(n):
                    if rem_adj[i]:
                        has_edge = True
                        break
                if not has_edge:
                    break
                
                # Pick a starting point
                # 50% chance: Max Degree Vertex
                # 50% chance: Edge with Min Common Neighbors
                
                if rng.random() < 0.5:
                    # Find vertex with max degree
                    max_deg = -1
                    best_v = -1
                    for v in vertices:
                        d = rem_adj[v].bit_count()
                        if d > max_deg:
                            max_deg = d
                            best_v = v
                    
                    if best_v == -1:
                        break
                        
                    # Find max clique in N(best_v)
                    # Candidates are neighbors of best_v
                    P = rem_adj[best_v]
                    X = 0
                    K = bronk_max_clique(P, X, rem_adj, deadline)
                    clique_mask = K | (1 << best_v)
                else:
                    # Find edge with min common neighbors
                    # Sample edges if many
                    # To find an edge, we need to iterate over pairs
                    # This is O(n^2). For n=64, 2000 pairs. Fast enough.
                    min_common = float('inf')
                    best_u = -1
                    best_v = -1
                    
                    # Optimization: Only check edges that exist
                    # Iterate u < v
                    for u in range(n):
                        if not rem_adj[u]:
                            continue
                        # Only check v > u to avoid duplicates
                        # But we need to ensure edge exists
                        temp = rem_adj[u] & ((1 << n) - 1) & ~((1 << (u+1)) - 1)
                        while temp:
                            v = (temp & -temp).bit_length() - 1
                            temp &= temp - 1
                            # Edge (u,v) exists
                            common = rem_adj[u] & rem_adj[v]
                            count = common.bit_count()
                            if count < min_common:
                                min_common = count
                                best_u = u
                                best_v = v
                                if count == 0:
                                    break
                        if min_common == 0:
                            break
                            
                    if best_u == -1:
                        break
                        
                    # Find max clique in N(best_u) ∩ N(best_v)
                    P = rem_adj[best_u] & rem_adj[best_v]
                    X = 0
                    K = bronk_max_clique(P, X, rem_adj, deadline)
                    clique_mask = K | (1 << best_u) | (1 << best_v)
                
                # Extract clique vertices
                clique_vertices = []
                cm = clique_mask
                while cm:
                    v = (cm & -cm).bit_length() - 1
                    cm &= cm - 1
                    clique_vertices.append(v)
                    
                if len(clique_vertices) < 2:
                    # Should not happen if edge exists, but safety
                    # Force a single edge
                    # Find any edge
                    found = False
                    for u in range(n):
                        if rem_adj[u]:
                            v = (rem_adj[u] & -rem_adj[u]).bit_length() - 1
                            clique_vertices = [u, v]
                            clique_mask = (1 << u) | (1 << v)
                            found = True
                            break
                    if not found:
                        break

                cliques.append(clique_vertices)
                
                # Remove edges within clique
                for i in range(len(clique_vertices)):
                    ui = clique_vertices[i]
                    for j in range(i+1, len(clique_vertices)):
                        vj = clique_vertices[j]
                        rem_adj[ui] &= ~(1 << vj)
                        rem_adj[vj] &= ~(1 << ui)
            
            if cliques and len(cliques) < best_count:
                best_count = len(cliques)
                best_cliques = cliques
        
        return best_cliques

    # Solve both instances
    # Allocate time: 60% to first, 40% to second? 
    # Or just run sequentially with a shared deadline.
    # n=32 is much faster. 
    # Let's give 50s to first, 90s to second? 
    # Total 140s.
    
    deadline1 = start_time + 50.0
    deadline2 = DEADLINE
    
    result1 = solve_instance(32, 0.6, 601, deadline1)
    
    if time.time() > DEADLINE:
        adj2, edges2 = gen_graph(64, 0.55, 602)
        result2 = [[u, v] for u, v in edges2]
    else:
        result2 = solve_instance(64, 0.55, 602, DEADLINE)
    
    return [result1, result2]
