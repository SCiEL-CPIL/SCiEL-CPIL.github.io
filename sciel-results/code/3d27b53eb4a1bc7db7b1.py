import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    deadline = time.time() + 85

    # Tomita's algorithm for Maximum Clique
    # R: current clique
    # P: candidates
    # X: excluded
    
    def expand(R, P, X, best_clique):
        if time.time() > deadline:
            return
        
        # Greedy coloring bound
        # Color P such that adjacent vertices have different colors
        # The number of colors is an upper bound on the clique size in P
        order = list(P)
        # Sort by degree in P for better coloring? Or just greedy
        # Simple greedy coloring
        colors = {}
        color = 0
        # To make the bound tighter, we can order vertices
        # Here we use a simple greedy approach
        colored = []
        while order:
            color += 1
            new_order = []
            for v in order:
                if all(colors.get(u, 0) != color for u in adj[v] if u in colors):
                    colors[v] = color
                    colored.append(v)
                else:
                    new_order.append(v)
            order = new_order
        
        # If |R| + max_color <= len(best_clique), prune
        if len(R) + color <= len(best_clique):
            return
            
        # Heuristic: pick vertex with highest color (or degree)
        # Actually, standard Tomita picks from the last color class
        # Let's pick a vertex from the highest color class
        if not order and not colored:
            return
            
        # We need to pick a vertex to add to R
        # Standard approach: iterate through color classes from highest to lowest
        # For each vertex in the class, recurse
        
        # To keep it simple and fast for n=24:
        # We can just iterate through P
        # But the coloring bound is key.
        
        # Let's re-implement the expansion more carefully
        # This simple version might be slow if not optimized, but n=24 is small.
        
        # Actually, let's just use a simple backtracking with the bound.
        # Pick v from P
        # R' = R + {v}
        # P' = P intersect adj[v]
        # X' = X intersect adj[v]
        
        # We need to pick v.
        # To ensure we find the max, we can just iterate all v in P.
        # But we want to prune.
        
        # Let's try a different order:
        # Sort P by color descending
        sorted_P = sorted(list(P), key=lambda x: -colors.get(x, 0))
        
        for v in sorted_P:
            if time.time() > deadline:
                return
            if v not in P: continue # might have been removed? No, P is static in this loop
            
            R_new = R | {v}
            P_new = P & adj[v]
            X_new = X & adj[v]
            
            expand(R_new, P_new, X_new, best_clique)
            
            # Remove v from P, add to X
            P.remove(v)
            X.add(v)
            
            # Recalculate bound? 
            # The bound calculation above was for the initial P.
            # We should update the bound dynamically or just rely on the recursion.
            # For n=24, this might be fast enough.
            
    best_clique = set()
    # Initial P is all vertices
    P_init = set(range(n))
    X_init = set()
    
    # To speed up, sort vertices by degree initially?
    # The coloring step inside expand handles some of this.
    
    expand(set(), P_init, X_init, best_clique)
    
    if not best_clique:
        # Fallback to a triangle or edge
        u, v = edges[0]
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
        
    branches = list(best_clique)
    # Ensure int types
    branches = [int(b) for b in branches]
    
    # Construct paths: direct edges for clique
    paths = []
    for i in range(len(branches)):
        for j in range(i+1, len(branches)):
            u = branches[i]
            v = branches[j]
            # Check if edge exists (it should, since it's a clique)
            if v in adj[u]:
                paths.append([u, v])
            else:
                # If not a clique (should not happen), find path
                # But we assume it is a clique
                pass
                
    return {'branches': branches, 'paths': paths}
