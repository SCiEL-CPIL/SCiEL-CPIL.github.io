import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_result = {'n': 39, 'edges': []}
    
    # Try n=47 first (min deg 8), then 46 (min deg 7)
    # The prompt says 46 or 47 is the answer.
    # 46 is safer.
    
    for n in [47, 46, 45]:
        if time.time() > deadline:
            break
            
        min_deg_needed = n - 39
        if min_deg_needed <= 0:
            # Trivial case, but we want max n
            best_result = {'n': n, 'edges': []}
            continue
            
        success = False
        # Increase restarts for higher n
        max_restarts = 200 if n == 47 else 300
        
        for attempt in range(max_restarts):
            if time.time() > deadline:
                break
                
            random.seed(attempt * 1000 + n)
            
            adj = [set() for _ in range(n)]
            edges = []
            
            # Greedy construction
            # We will keep adding edges until no safe edge exists
            # or we reach a high enough density
            # To ensure min degree, we prefer edges incident to low-degree vertices
            
            # List of all potential edges
            # We don't precompute, we scan each time?
            # n=47, n^2/2 = 1081 edges.
            # Scanning all edges each step is O(n^2 * E) which is O(n^4) ~ 5M.
            # 5M * 100 steps = 500M. Might be too slow in Python.
            
            # Optimization:
            # Only consider edges that can improve min degree?
            # Or just random sampling of safe edges?
            
            # Let's try a different strategy:
            # Randomly shuffle vertices.
            # Add edges in a random order, skipping unsafe ones.
            # This is O(n^2 * cost_check).
            
            vertex_order = list(range(n))
            random.shuffle(vertex_order)
            
            # We want to ensure min degree.
            # A simple random edge addition might leave some vertices isolated.
            # We should prioritize connecting low-degree vertices.
            
            # Let's use the min-degree sum heuristic but optimize the search.
            # Instead of scanning all pairs, we can maintain a list of "candidate" edges.
            # But degrees change.
            
            # Given n is small (47), scanning all pairs is actually fine if optimized.
            # 1000 edges * 47 steps * 200 restarts = 9.4M edge checks.
            # Each check is O(deg^2) ~ 81.
            # 9.4M * 81 = 760M operations.
            # This is likely too slow for Python in 120s.
            
            # We need a faster check or fewer restarts.
            # The check:
            # For u, v:
            #   for a in adj[u]:
            #     for b in adj[v]:
            #       if a == b: continue
            #       if b in adj[a]: # (a,b) is edge
            #          return False (C4 found)
            
            # We can speed this up by using bitsets or numpy?
            # Or just reduce restarts.
            
            # Let's try 50 restarts first.
            # And optimize the inner loop.
            
            for _ in range(n * 2): # Max edges added
                if time.time() > deadline:
                    break
                    
                best_edge = None
                best_score = float('inf')
                
                # Sample a subset of edges to speed up?
                # Or just scan all.
                # Let's scan all but break early if we find a very good one?
                # No, we want the best to balance degrees.
                
                # Optimization: Only check edges where deg[u] < min_deg_needed or deg[v] < min_deg_needed?
                # If both are >= min_deg_needed, adding the edge doesn't help min degree.
                # But it might be necessary for connectivity? No, min degree is the constraint.
                # If we already have min degree, we can stop.
                
                min_deg_current = min(len(adj[i]) for i in range(n))
                if min_deg_current >= min_deg_needed:
                    break
                
                # Find best edge
                # To speed up, we can iterate u, and for each u, iterate v.
                for u in range(n):
                    # If deg[u] is already high, we might not need to connect u further?
                    # But u might be connected to low degree v.
                    # So we still check.
                    for v in range(u + 1, n):
                        if v in adj[u]:
                            continue
                        
                        # Quick check: if deg[u] + deg[v] >= best_score, skip detailed check?
                        # No, we need to know if it's safe first.
                        # But if it's unsafe, we skip.
                        
                        # Check C4
                        # If adj[u] and adj[v] have common neighbors, or path of length 3.
                        # Let's implement the path of length 3 check.
                        
                        safe = True
                        # Optimize: 
                        # If len(adj[u]) == 0 or len(adj[v]) == 0, safe.
                        # If len(adj[u]) == 1, let a be the neighbor.
                        #   Check if a is connected to any neighbor of v?
                        
                        # General case:
                        # We need to check if there is an edge (a,b) with a in N(u), b in N(v), a != b.
                        # We can iterate over the smaller set.
                        
                        nu = adj[u]
                        nv = adj[v]
                        
                        if len(nu) < len(nv):
                            for a in nu:
                                # Check if a is connected to any b in nv
                                # And a != b
                                # If a in nv, then a is common neighbor.
                                # If a in nv, does it form C4?
                                # Path u-a-v-a-u? No, a=v? No a in N(u).
                                # If a in N(v), then u-a-v is path length 2.
                                # Adding u-v creates triangle u-a-v-u.
                                # Is triangle a C4? No.
                                # But does it create a C4 with other nodes?
                                # We need path v-b-a-u?
                                # If a in N(v), and there is b in N(a) \cap N(u)?
                                # This is getting complicated.
                                
                                # Let's stick to the definition:
                                # C4 exists if there is a path of length 3 between u and v in G.
                                # Path: u - a - b - v.
                                # a in N(u), b in N(v), (a,b) in E.
                                # a != b.
                                
                                # Iterate a in N(u):
                                #   For each b in N(a):
                                #      If b in N(v) and b != a:
                                #          C4 found.
                                
                                # This is O(deg(u) * deg(a)).
                                # deg(a) is small.
                                
                                for b in adj[a]:
                                    if b == a: continue
                                    if b in nv:
                                        # Found path u-a-b-v
                                        # Check if a != b (handled)
                                        # Check if edges are distinct?
                                        # u-a, a-b, b-v.
                                        # Are they distinct?
                                        # u != a, a != b, b != v.
                                        # If u == v, impossible (simple graph).
                                        # If a == v, then u-v-b-v? No.
                                        # If b == u, then u-a-u-v? No.
                                        # So yes, C4.
                                        safe = False
                                        break
                                if not safe:
                                    break
                        else:
                            for b in nv:
                                for a in adj[b]:
                                    if a == b: continue
                                    if a in nu:
                                        safe = False
                                        break
                                if not safe:
                                    break
                        
                        if not safe:
                            continue
                        
                        score = len(nu) + len(nv)
                        if score < best_score:
                            best_score = score
                            best_edge = (u, v)
                
                if best_edge is None:
                    break
                    
                u, v = best_edge
                adj[u].add(v)
                adj[v].add(u)
                edges.append([u, v])
                
            # Final check
            min_deg = min(len(adj[i]) for i in range(n))
            if min_deg >= min_deg_needed:
                # Verify C4-free
                has_c4 = False
                for u in range(n):
                    for v in range(u + 1, n):
                        if v in adj[u]:
                            continue
                        # Check if u, v have common neighbor?
                        # No, we need to check if adding (u,v) would create C4?
                        # No, we just check if the graph HAS a C4.
                        # A C4 is a cycle.
                        # We can just check all pairs (u,v) if they have >1 common neighbor?
                        # No, that detects C4s where u,v are opposite.
                        # What if u,v are adjacent?
                        # If u-v is an edge, and they have a common neighbor w, we have a triangle.
                        # C4 is a cycle of 4 distinct vertices.
                        # So we check all pairs (u,v) (adjacent or not) if they have >= 2 common neighbors?
                        # If u,v are adjacent and have 2 common neighbors w1, w2, then u-w1-v-w2-u is a C4.
                        # If u,v are NOT adjacent and have 2 common neighbors w1, w2, then u-w1-v-w2-u is a C4.
                        # So yes, checking if any pair has >= 2 common neighbors is sufficient.
                        
                        common = adj[u] & adj[v]
                        if len(common) >= 2:
                            has_c4 = True
                            break
                    if has_c4:
                        break
                if not has_c4:
                    success = True
                    best_result = {'n': n, 'edges': edges}
                    break
                    
        if success:
            break
            
    return best_result
