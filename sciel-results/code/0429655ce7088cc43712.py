import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    def check_alpha(n, edges):
        # Simple greedy approximation for independence number
        # For exact check on small n, we can use branch and bound or just greedy for speed
        # But we need to be sure.
        # n <= 43.
        # We can use a simple recursive search for independent set of size 11.
        # If it finds one, alpha >= 11.
        # If it exhausts search, alpha < 11.
        
        adj = [[False]*n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = True
            adj[v][u] = True
            
        # Find independent set of size 11
        # Use backtracking
        def find_independent_set(k, current, candidates):
            if len(current) == k:
                return True
            if not candidates:
                return False
            # Pruning: if remaining candidates + current < k, fail
            if len(candidates) + len(current) < k:
                return False
            
            # Heuristic: pick vertex with min degree in induced subgraph?
            # Or just iterate
            for i in range(len(candidates)):
                v = candidates[i]
                # Try including v
                new_candidates = [u for u in candidates[i+1:] if not adj[v][u]]
                if find_independent_set(k, current + [v], new_candidates):
                    return True
            # Try excluding v? 
            # Standard independent set search:
            # We can just recurse on candidates[i+1:]
            # But this is slow.
            # For n=43, k=11, it might be too slow if done many times.
            # Let's use a faster heuristic or limit the search depth/time.
            pass
        
        # Better: Use a simple greedy check first.
        # If greedy finds 11, it's invalid.
        # If greedy doesn't, it might still be valid.
        # Given the time constraints, we might just rely on the construction process to ensure low alpha.
        # Or use a very fast exact solver for alpha.
        
        # Let's implement a simple exact solver with timeout
        import sys
        sys.setrecursionlimit(10000)
        
        def has_independent_set_k(k):
            # State: set of available vertices
            # We want to know if there is an independent set of size k.
            # Use bitsets for speed? n=43 fits in 64-bit int.
            avail = (1 << n) - 1
            target = k
            
            def search(avail, depth):
                if depth == 0:
                    return True
                if avail == 0:
                    return False
                # Pruning: popcount(avail) < depth
                if bin(avail).count('1') < depth:
                    return False
                
                # Pick a vertex
                # Heuristic: pick vertex with highest degree in induced subgraph to prune more?
                # Or lowest?
                # Let's just pick the lowest set bit for simplicity, or random.
                # To speed up, we can order vertices.
                # For now, just pick lowest.
                v_bit = avail & -avail
                v = v_bit.bit_length() - 1
                
                # Branch 1: Include v
                # Neighbors of v in avail
                neighbors = 0
                for u in range(n):
                    if adj[v][u] and (avail & (1 << u)):
                        neighbors |= (1 << u)
                new_avail = avail & ~(neighbors | (1 << v))
                if search(new_avail, depth - 1):
                    return True
                
                # Branch 2: Exclude v
                new_avail2 = avail & ~(1 << v)
                if search(new_avail2, depth):
                    return True
                
                return False
            
            return search(avail, target)
        
        return not has_independent_set_k(11)

    def get_common_neighbor_count(n, edges):
        adj = [[False]*n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = True
            adj[v][u] = True
        M = [[0]*n for _ in range(n)]
        for u in range(n):
            for v in range(u+1, n):
                cnt = 0
                for w in range(n):
                    if adj[u][w] and adj[v][w]:
                        cnt += 1
                M[u][v] = cnt
                M[v][u] = cnt
        return M

    def add_edge_safe(n, edges, M, u, v):
        if u == v: return False
        if M[u][v] >= 2: return False
        # Check if edge exists
        # We should maintain a set of edges
        # For simplicity in this snippet, we assume caller checks
        return True

    best_graph = None
    best_n = 30
    
    # Fallback: 10 triangles
    edges_fallback = []
    for t in range(10):
        a, b, c = 3*t, 3*t+1, 3*t+2
        edges_fallback += [[a,b], [a,c], [b,c]]
    best_graph = {'n': 30, 'edges': edges_fallback}

    for n in range(43, 30, -1):
        if time.time() > deadline - 20:
            break
            
        found = False
        for trial in range(5):
            if time.time() > deadline - 10:
                break
                
            # Initialize random graph
            edges = []
            for u in range(n):
                for v in range(u+1, n):
                    if random.random() < 0.5:
                        edges.append([u, v])
            
            M = get_common_neighbor_count(n, edges)
            edge_set = set(tuple(sorted(e)) for e in edges)
            
            # Phase 1: Remove edges to eliminate C4s
            while time.time() < deadline - 5:
                # Find a C4
                c4_found = False
                for u in range(n):
                    for v in range(u+1, n):
                        if M[u][v] >= 2:
                            c4_found = True
                            # Find a common neighbor
                            # We need to find w such that adj[u][w] and adj[v][w]
                            # We can just pick one and remove an edge
                            # Let's remove edge (u, w)
                            # Find w
                            for w in range(n):
                                if w != u and w != v:
                                    if (u, w) in edge_set or (w, u) in edge_set:
                                        if (v, w) in edge_set or (w, v) in edge_set:
                                            # Remove (u, w)
                                            if (u, w) in edge_set:
                                                edge_set.remove((u, w))
                                            else:
                                                edge_set.remove((w, u))
                                            edges.remove([u, w] if [u, w] in edges else [w, u])
                                            # Update M
                                            # M[x][y] changes for x,y involving u or w
                                            # Simplify: Recalculate M? Too slow.
                                            # Update M locally.
                                            # N(u) lost w. N(w) lost u.
                                            # For any z:
                                            # M[z][u] = |N(z) \cap N(u)|. N(u) changed.
                                            # M[z][w] = |N(z) \cap N(w)|. N(w) changed.
                                            # M[u][w] = 0 (not adjacent).
                                            for z in range(n):
                                                if z == u or z == w: continue
                                                # Check if z is neighbor of u and w previously?
                                                # This is getting complex.
                                                # Given n is small, let's just recalculate M if we do many updates.
                                                # Or, simpler: just recalculate M from scratch every 10 updates?
                                                # Or, just break and recalculate.
                                                pass
                                            # Let's just recalculate M from scratch to be safe and simple
                                            # It's O(n^3) = 43^3 ~ 80k ops. Fast enough.
                                            M = get_common_neighbor_count(n, edges)
                                            break
                            break
                    if c4_found:
                        break
                if not c4_found:
                    break
            
            # Phase 2: Check Alpha
            if check_alpha(n, edges):
                best_graph = {'n': n, 'edges': [list(e) for e in edge_set]}
                found = True
                break # Try next n
            
            # Phase 3: Try to add edges to reduce alpha
            # (Optional, might be risky)
            
    return best_graph

# EVOLVE-BLOCK-END
