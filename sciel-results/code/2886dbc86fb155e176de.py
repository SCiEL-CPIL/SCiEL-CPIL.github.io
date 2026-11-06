import itertools
import time
import random

def construct():
    n = 18
    start_time = time.time()
    deadline = 140.0
    
    # Try multiple random K4-free graphs and pick the best one
    # A K4-free graph: no 4 vertices all mutually connected
    
    best_score = n + 1
    best_edges = None
    
    random.seed(42)
    
    for attempt in range(200):
        if time.time() > deadline - 5:
            break
        
        # Build a random K4-free graph
        # Start with empty graph, add edges randomly, reject if creates K4
        adj = [[False] * n for _ in range(n)]
        edges = []
        
        # Add edges in random order
        all_possible = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.shuffle(all_possible)
        
        for u, v in all_possible:
            # Check if adding (u,v) creates a K4
            creates_k4 = False
            # A K4 containing u,v: {u,v,x,y} where x,y are common neighbors of u and v
            common = []
            for w in range(n):
                if w != u and w != v and adj[u][w] and adj[v][w]:
                    common.append(w)
            # Check if any pair in common is connected
            for i in range(len(common)):
                for j in range(i+1, len(common)):
                    if adj[common[i]][common[j]]:
                        creates_k4 = True
                        break
                if creates_k4:
                    break
            
            if not creates_k4:
                adj[u][v] = True
                adj[v][u] = True
                edges.append([u, v])
        
        # Compute the size of the largest triangle-free induced subgraph
        # Use branch and bound
        def max_triangle_free(adj, n):
            # Find largest subset with no triangle
            best = 0
            all_vertices = set(range(n))
            
            # Precompute triangles
            triangles = []
            for u in range(n):
                for v in range(u+1, n):
                    if adj[u][v]:
                        for w in range(v+1, n):
                            if adj[u][w] and adj[v][w]:
                                triangles.append((u, v, w))
            
            # For each triangle, at least one vertex must be excluded
            # This is a hitting set problem
            
            # Branch and bound: try to build a triangle-free set
            # A set is triangle-free if it contains no triangle
            
            # Use recursive search with pruning
            def is_triangle_free(vertices):
                vset = set(vertices)
                for t in triangles:
                    if t[0] in vset and t[1] in vset and t[2] in vset:
                        return False
                return True
            
            # Find max triangle-free subset
            # Try all subsets of size 8, 7, ... until we find one that's triangle-free
            for size in range(8, 0, -1):
                found = False
                for combo in itertools.combinations(range(n), size):
                    if is_triangle_free(combo):
                        return size
                # If no triangle-free set of this size, continue
            return 0
        
        score = max_triangle_free(adj, n)
        
        if score < best_score:
            best_score = score
            best_edges = edges[:]
        
        if best_score <= 6:
            break
    
    if best_edges is None:
        # Fallback: return empty graph
        return []
    
    return best_edges
