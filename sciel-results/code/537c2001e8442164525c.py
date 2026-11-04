import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0

    if k == 4 and g == 10:
        # 1. Try Circulant
        result = _search_circulant_optimized(320, k, g, deadline)
        if result is not None:
            return result
        
        # 2. Try Random Regular Graph
        result = _search_random_regular(320, k, g, deadline)
        if result is not None:
            return result
            
        # 3. Fallback to LUW
        return _luw_construct(k, g)

    # General fallback
    return _luw_construct(k, g)

def _search_random_regular(n, k, g, deadline):
    """
    Search for a k-regular graph on n vertices with girth >= g.
    Uses configuration model and girth check.
    """
    # We need to ensure the graph is simple.
    # Configuration model: 2n stubs? No, k*n stubs.
    # Pair them randomly.
    
    # Girth check function
    def check_girth(adj, n, g):
        # adj is a list of lists of neighbors
        # Check for cycles of length < g
        # For each edge (u, v), check if there is a path from u to v of length < g-1
        # that does not use the edge (u, v).
        
        # BFS from u, avoiding v at start? No.
        # Just BFS from u, and see if we reach v in < g-1 steps.
        # But we must exclude the direct edge?
        # If we find a path of length L < g-1, then L+1 < g is a cycle.
        
        # Optimization: Only check up to depth g//2 - 1?
        # No, girth is the shortest cycle.
        # We can just run BFS from each vertex up to depth g//2 - 1 and check for self-loops?
        # No, that finds even cycles.
        
        # Better: For each vertex u, run BFS up to depth (g-1)//2.
        # If we encounter a vertex at depth d such that the shortest path from u to that vertex is < d, we have a cycle.
        # Actually, simpler:
        # Run BFS from each vertex u.
        # Record dist[u][v].
        # If dist[u][v] < g, and there is an edge (u,v) that is not the last edge in the shortest path?
        # No, just check: if there is any cycle of length < g.
        
        # Efficient check:
        # For each vertex u, BFS up to depth g//2 - 1.
        # If we find a neighbor w of u such that dist[u][w] < g-1?
        # No.
        
        # Let's use the standard "for each edge, check alternative path" method.
        # It is O(m * (k-1)^(g/2)).
        
        for u in range(n):
            for v in adj[u]:
                if u > v: continue # Check each edge once
                
                # BFS from u to v, avoiding the edge (u,v)
                # We want a path of length < g-1.
                # Max depth = g-2.
                
                visited = [False] * n
                visited[u] = True
                queue = [(u, 0)]
                
                found_shorter = False
                while queue:
                    if time.time() > deadline:
                        return False # Timeout, assume invalid to be safe? Or return None?
                        # Better to handle timeout in outer loop.
                    
                    node, depth = queue.pop(0)
                    if depth >= g - 2:
                        continue
                    
                    for w in adj[node]:
                        if w == u and node == v: # Avoid going back to u via v? No, we started at u.
                            continue
                        if w == v and node == u: # Avoid the direct edge (u,v)
                            continue
                            
                        if not visited[w]:
                            visited[w] = True
                            if w == v:
                                # Found path of length depth + 1
                                # Cycle length = depth + 1 + 1 = depth + 2
                                if depth + 2 < g:
                                    return False
                            queue.append((w, depth + 1))
                
        return True

    # Generate random graph
    max_attempts = 1000
    for _ in range(max_attempts):
        if time.time() > deadline:
            break
            
        # Configuration Model
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        
        random.shuffle(stubs)
        
        adj = [[] for _ in range(n)]
        valid = True
        for i in range(0, len(stubs), 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                valid = False
                break
            if v in adj[u]:
                valid = False
                break
            adj[u].append(v)
            adj[v].append(u)
            
        if not valid:
            continue
            
        # Check girth
        if check_girth(adj, n, g):
            # Convert adj to edge list
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append((u, v))
            return n, edges
            
    return None

# ... keep other functions ...
