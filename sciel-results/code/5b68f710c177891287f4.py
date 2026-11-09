"""C4-free graph construction maximizing n with min_degree >= n-39."""
import time
import random

def construct():
    deadline = time.time() + 140.0
    best_n = 39
    best_edges = []
    
    # Try to build C4-free graphs for n=47, then 46, etc.
    # For n=47: min_degree >= 8, so complement degree <= 38
    # For n=46: min_degree >= 7, so complement degree <= 38
    
    def try_build(n, min_deg, time_limit):
        """Try to build a C4-free graph on n vertices with min degree >= min_deg."""
        # Strategy: start with a random graph and try to add edges while maintaining C4-free
        # Use multiple restarts with different seeds
        
        for seed in range(200):
            if time.time() > time_limit:
                return None
            
            random.seed(seed)
            
        # Better approach: use a structured construction
        # Use a projective plane-like structure or random with C4 avoidance
        
        # Greedy edge addition with C4 check
        # For each pair (u,v), check if adding edge creates C4
        # C4 is created if there exist x,y such that x~u, x~v, y~u, y~v (x≠y)
        # i.e., |N(u) ∩ N(v)| >= 1 before adding edge (u,v) would create C4
        # Wait, no: C4 is u-x-v-y-u. So we need two common neighbors.
        # Actually: adding edge (u,v) creates a C4 iff there exist distinct x,y 
        # such that x is adjacent to both u and v, and y is adjacent to both u and v.
        # No wait: C4 is a 4-cycle. Adding edge (u,v) creates C4 if there's a path 
        # of length 3 from u to v: u-a-b-v. So we need |N(u) ∩ N(v)| >= 2? 
        # No: a C4 is u-x-v-y-u. If we add edge (u,v), a C4 exists if there are 
        # two distinct vertices x,y both adjacent to u and v. So |N(u)∩N(v)| >= 2.
        # Wait, that's not right either. Let me think again.
        # A C4 is a cycle of length 4: v1-v2-v3-v4-v1.
        # If we add edge (u,v), it creates a C4 if there exist vertices a,b (distinct from u,v, a≠b)
        # such that u-a, a-b, b-v, and v-u form a cycle. But v-u is the new edge.
        # So we need: a adjacent to u, b adjacent to v, a adjacent to b, and a≠b, a≠u,v, b≠u,v.
        # Hmm, that's more complex.
        # Actually simpler: adding edge (u,v) creates a C4 iff there exist two distinct 
        # vertices x, y such that x is adjacent to both u and v, and y is adjacent to both u and v.
        # No! That would be a C4: u-x-v-y-u. Yes! So |N(u)∩N(v)| >= 2 before adding (u,v) 
        # would create a C4.
        # Wait, but x and y need to be distinct from u and v, which they are since 
        # N(u) doesn't contain u.
        # So the condition is: |N(u) ∩ N(v)| >= 2 means adding (u,v) creates a C4.
        # Actually no: if |N(u)∩N(v)| = 1, say {x}, then we have u-x-v but no 4-cycle yet.
        # If |N(u)∩N(v)| >= 2, say {x,y}, then u-x-v-y-u is a C4.
        # So the constraint is: when adding edge (u,v), we need |N(u) ∩ N(v)| <= 1.
        
        # Let's use a different approach: build the complement (which should have no K_{1,39})
        # Actually, let's just do greedy with the C4 check.
        
        def build_greedy(n, min_deg, rng, max_iters=100000):
            # Start with empty graph, try to add edges
            # We want to end up with min degree >= min_deg
            # Strategy: try to add edges until we can't, or until all degrees >= min_deg
            
            adj = [[False]*n for _ in range(n)]
            degree = [0]*n
            # common[i][j] = number of common neighbors of i and j
            common = [[0]*n for _ in range(n)]
            
            # Precompute: for each pair, track common neighbors
            # When adding edge (u,v): 
            #   For all w != u,v: if adj[w][u] and adj[w][v], common[u][v]++
            #   Actually we need to update common for all pairs involving u or v
            
            # Let's use a simpler but potentially slower approach first
            # Check C4 by counting common neighbors on the fly
            
            def can_add(u, v):
                if adj[u][v]:
                    return False
                # Count common neighbors
                cnt = 0
                for w in range(n):
                    if w != u and w != v and adj[w][u] and adj[w][v]:
                        cnt += 1
                        if cnt >= 2:
                            return False
                return True
            
            def add_edge(u, v):
                adj[u][v] = adj[v][u] = True
                degree[u] += 1
                degree[v] += 1
            
            # Greedy: try to add edges to raise minimum degree
            # Repeat: find vertex with lowest degree, try to connect it to others
            for iteration in range(max_iters):
                if time.time() > time_limit:
                    return None
                
                # Find vertex with minimum degree
                min_v = min(range(n), key=lambda i: degree[i])
                if degree[min_v] >= min_deg:
                    # Check if all vertices meet the requirement
                    if all(degree[i] >= min_deg for i in range(n)):
                        return adj, degree
                
                # Try to add edges from min_v to other vertices
                candidates = [v for v in range(n) if v != min_v and not adj[min_v][v]]
                rng.shuffle(candidates)
                
                added = False
                for v in candidates:
                    if can_add(min_v, v):
                        add_edge(min_v, v)
                        added = True
                        break
                
                if not added:
                    # Try other vertices
                    for u in range(n):
                        if degree[u] >= min_deg:
                            continue
                        candidates = [v for v in range(n) if v != u and not adj[u][v]]
                        rng.shuffle(candidates)
                        for v in candidates:
                            if can_add(u, v):
                                add_edge(u, v)
                                added = True
                                break
                        if added:
                            break
                    if not added:
                        break  # Stuck
            
            # Final check
            if all(degree[i] >= min_deg for i in range(n)):
                return adj, degree
            return None
        
        # Try multiple random seeds
        for seed in range(50):
            if time.time() > time_limit:
                break
            rng = random.Random(seed * 7919 + 13)
            result = build_greedy(n, min_deg, rng, max_iters=50000)
            if result is not None:
                adj, degree = result
                edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        if adj[i][j]:
                            edges.append([i, j])
                return edges
        
        return None
    
    # Try n=47 first (min_deg = 8)
    for n in [47, 46]:
        min_deg = n - 39
        if min_deg < 0:
            min_deg = 0
        time_limit = deadline - 5.0
        edges = try_build(n, min_deg, time_limit)
        if edges is not None:
            # Verify
            adj = [[False]*n for _ in range(n)]
            for e in edges:
                adj[e[0]][e[1]] = adj[e[1]][e[0]] = True
            degree = [sum(adj[i]) for i in range(n)]
            if min(degree) >= min_deg:
                # Verify no C4
                has_c4 = False
                for i in range(n):
                    for j in range(i+1, n):
                        if not adj[i][j]:
                            continue
                        # Check for C4: need two common neighbors
                        common = [k for k in range(n) if k != i and k != j and adj[k][i] and adj[k][j]]
                        if len(common) >= 2:
                            has_c4 = True
                            break
                    if has_c4:
                        break
                if not has_c4:
                    return {'n': n, 'edges': edges}
    
    return {'n': best_n, 'edges': best_edges}
