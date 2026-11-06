def construct():
        import time
        import random
        
        N = 35
        start_time = time.time()
        time_limit = 100 # seconds
        
        # Initialize random graph
        # Or use a structured start?
        # Let's try random with p=0.5
        
        def make_random_graph(n):
            adj = [set() for _ in range(n)]
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.5:
                        adj[i].add(j)
                        adj[j].add(i)
            return adj
            
        def has_k4(adj, n):
            # Returns the K4 as a tuple if found, else None
            for u in range(n):
                N_u = adj[u]
                # Get list of neighbors
                neighbors = list(N_u)
                L = len(neighbors)
                if L < 3: continue
                # Check for triangle in neighbors
                for i in range(L):
                    v = neighbors[i]
                    for j in range(i+1, L):
                        w = neighbors[j]
                        if w in adj[v]:
                            # Triangle u-v-w
                            # Check for x connected to u, v, w
                            # x must be in N_u \cap N_v \cap N_w
                            # N_u is set, N_v is set, N_w is set
                            # Intersection
                            common = N_u.intersection(adj[v]).intersection(adj[w])
                            if common:
                                x = next(iter(common))
                                return (u, v, w, x)
            return None

        def has_indep6(adj, n):
            # Returns the independent set of 6 if found, else None
            # Complement graph logic
            # For each u, check non-neighbors for independent set of 5
            for u in range(n):
                M_u = [v for v in range(n) if v != u and v not in adj[u]]
                # Need independent set of 5 in M_u
                # Use backtracking
                def find_indep_set(vertices, k):
                    if k == 0:
                        return []
                    if not vertices:
                        return None
                    v = vertices[0]
                    rest = vertices[1:]
                    # Try including v
                    # v's non-neighbors in rest
                    candidates = [x for x in rest if x not in adj[v]]
                    res = find_indep_set(candidates, k-1)
                    if res is not None:
                        return [v] + res
                    # Try excluding v
                    res = find_indep_set(rest, k)
                    if res is not None:
                        return res
                    return None
                
                res = find_indep_set(M_u, 5)
                if res is not None:
                    return [u] + res
            return None

        best_graph = None
        # We don't really track "best" if we just want valid.
        # But if we don't find valid, we return seed.
        
        # Let's run multiple restarts
        for restart in range(1000):
            if time.time() - start_time > time_limit:
                break
                
            adj = make_random_graph(N)
            
            # Local search
            for step in range(10000):
                if time.time() - start_time > time_limit:
                    break
                    
                k4 = has_k4(adj, N)
                if k4:
                    # Remove an edge from the K4
                    u, v, w, x = k4
                    # Pick random edge to remove
                    edges = [(u,v), (u,w), (u,x), (v,w), (v,x), (w,x)]
                    eu, ev = random.choice(edges)
                    adj[eu].remove(ev)
                    adj[ev].remove(eu)
                    continue
                    
                ind6 = has_indep6(adj, N)
                if ind6:
                    # Add an edge within the independent set
                    # Pick two random vertices from ind6
                    u, v = random.sample(ind6, 2)
                    if v not in adj[u]:
                        adj[u].add(v)
                        adj[v].add(u)
                    continue
                    
                # If neither found, it's valid!
                return {'n': N, 'edges': [[u, v] for u in range(N) for v in range(u+1, N) if v in adj[u]]}
            
        # If not found, return seed
        squares = {x * x % 17 for x in range(1, 17)}
        return {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]}
