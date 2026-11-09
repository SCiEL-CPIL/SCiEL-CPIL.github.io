import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    n = 16
    
    # Grötzsch graph on vertices 0..10
    # C5: 0-1-2-3-4-0
    # Mycielski: vertices 5..9 are copies, vertex 10 is the new one
    edges_set = set()
    cycle = [(i, (i + 1) % 5) for i in range(5)]
    for u, v in cycle:
        edges_set.add(frozenset((u, v)))
    for u, v in cycle:
        edges_set.add(frozenset((u, 5 + v)))
        edges_set.add(frozenset((v, 5 + u)))
    for u in range(5):
        edges_set.add(frozenset((5 + u, 10)))
    
    # Convert to adjacency
    adj = [set() for _ in range(n)]
    for e in edges_set:
        u, v = tuple(e)
        adj[u].add(v)
        adj[v].add(u)
    
    def is_triangle_free(adj):
        for u in range(n):
            for v in adj[u]:
                for w in adj[v]:
                    if w != u and w in adj[u]:
                        return False
        return True
    
    def is_3_colorable(adj):
        color = [-1] * n
        def backtrack(v):
            if v == n:
                return True
            for c in range(3):
                if all(color[u] != c or u not in adj[v] for u in range(v)):
                    color[v] = c
                    if backtrack(v + 1):
                        return True
                    color[v] = -1
            return False
        return backtrack(0)
    
    best_edges = len(edges_set)
    best_adj = [set(s) for s in adj]
    
    # Try to add edges involving vertices 11-15
    # For each new vertex, try connecting to subsets of existing vertices
    new_vertices = list(range(11, 16))
    
    def try_add_edges(adj, edges_set, deadline):
        nonlocal best_edges, best_adj
        if time.time() > deadline:
            return
        
        # Try adding edges for vertex 11 first, then 12, etc.
        for v in new_vertices:
            # Get all possible neighbors (vertices not already adjacent, and adding won't create triangle)
            possible = []
            for u in range(n):
                if u == v:
                    continue
                if u in adj[v]:
                    continue
                # Check if adding (v,u) creates a triangle
                triangle = False
                for w in adj[u]:
                    if w in adj[v]:
                        triangle = True
                        break
                if not triangle:
                    possible.append(u)
            
            # Try all subsets of possible neighbors (but limit to reasonable size)
            # For efficiency, try larger subsets first
            possible.sort(key=lambda u: len(adj[u]), reverse=True)
            
            # Try adding all possible, then remove if needed
            # First try: add all possible edges
            new_adj = [set(s) for s in adj]
            new_edges = set(edges_set)
            for u in possible:
                new_adj[v].add(u)
                new_adj[u].add(v)
                new_edges.add(frozenset((v, u)))
            
            if is_triangle_free(new_adj) and not is_3_colorable(new_adj):
                if len(new_edges) > best_edges:
                    best_edges = len(new_edges)
                    best_adj = [set(s) for s in new_adj]
                    edges_set = new_edges
                    adj = new_adj
            else:
                # Try subsets
                # This is expensive, so limit
                if len(possible) <= 12:
                    for r in range(len(possible), max(len(possible)//2, 0), -1):
                        if time.time() > deadline:
                            return
                        for subset in combinations(possible, r):
                            new_adj = [set(s) for s in adj]
                            new_edges = set(edges_set)
                            for u in subset:
                                new_adj[v].add(u)
                                new_adj[u].add(v)
                                new_edges.add(frozenset((v, u)))
                            if is_triangle_free(new_adj) and not is_3_colorable(new_adj):
                                if len(new_edges) > best_edges:
                                    best_edges = len(new_edges)
                                    best_adj = [set(s) for s in new_adj]
                                    edges_set = new_edges
                                    adj = new_adj
                                break
                        if time.time() > deadline:
                            return
        
        return adj, edges_set
    
    # Iteratively add vertices
    for v in new_vertices:
        adj, edges_set = try_add_edges(adj, edges_set, deadline)
        if time.time() > deadline:
            break
    
    # Convert best_adj to edge list
    result = []
    for u in range(n):
        for v in best_adj[u]:
            if u < v:
                result.append([u, v])
    
    return result
