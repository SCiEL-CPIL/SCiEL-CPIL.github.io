def construct():
    import math
    from functools import lru_cache
    
    N = 512
    # Build adjacency: can[i][j] = True if lcm(i,j) <= 512
    # We'll use 0-indexed: vertex i represents number i+1
    
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    # Build adjacency list
    adj = [set() for _ in range(N)]
    for i in range(N):
        for j in range(i+1, N):
            a, b = i+1, j+1
            if lcm(a, b) <= N:
                adj[i].add(j)
                adj[j].add(i)
    
    # Greedy coloring + branch and bound for maximum clique
    # Use Bron-Kerbosch with pivot and greedy coloring
    
    import sys
    sys.setrecursionlimit(10000)
    
    # Greedy coloring for upper bound
    def greedy_color(vertices, adj):
        """Color vertices greedily, return color classes."""
        color_classes = []
        uncolored = set(vertices)
        color = 0
        while uncolored:
            color += 1
            cls = set()
            for v in list(uncolored):
                # v can join current color class if no neighbor in cls
                if not adj[v] & cls:
                    cls.add(v)
                    uncolored.discard(v)
            if cls:
                color_classes.append(cls)
        return color_classes
    
    # Bron-Kerbosch with pivot
    def bron_kerbosch(R, P, X, adj):
        """Find max clique size."""
        if not P and not X:
            return len(R)
        
        # Greedy coloring for upper bound
        if P:
            color_classes = greedy_color(P, adj)
            # Upper bound: len(R) + len(color_classes)
            if len(R) + len(color_classes) <= best[0]:
                return best[0]
        
        # Pivot selection: choose vertex in P∪X with max degree in P
        U = P | X
        if U:
            pivot = max(U, key=lambda v: len(adj[v] & P))
            candidates = P - adj[pivot]
        else:
            candidates = P
        
        best_clique = best[0]
        for v in list(candidates):
            R_new = R | {v}
            P_new = P & adj[v]
            X_new = X & adj[v]
            P.discard(v)
            X.add(v)
            result = bron_kerbosch(R_new, P_new, X_new, adj)
            if result > best_clique:
                best_clique = result
                best[0] = best_clique
            # Restore
            P.add(v)
            X.discard(v)
        
        return best_clique
    
    best = [0]
    # Start with all vertices
    all_vertices = set(range(N))
    bron_kerbosch(set(), all_vertices, set(), adj)
    
    # Now we know the max size, but we need the actual set. Let's do a second pass to extract it.
    # Or modify bron_kerbosch to track the actual clique.
    
    # Let me redo with tracking
    best_clique_set = [set()]
    
    def bron_kerbosch_track(R, P, X, adj):
        if not P and not X:
            if len(R) > len(best_clique_set[0]):
                best_clique_set[0] = set(R)
            return len(R)
        
        if P:
            color_classes = greedy_color(P, adj)
            if len(R) + len(color_classes) <= len(best_clique_set[0]):
                return len(best_clique_set[0])
        
        U = P | X
        if U:
            pivot = max(U, key=lambda v: len(adj[v] & P))
            candidates = P - adj[pivot]
        else:
            candidates = P
        
        for v in list(candidates):
            R_new = R | {v}
            P_new = P & adj[v]
            X_new = X & adj[v]
            P.discard(v)
            X.add(v)
            bron_kerbosch_track(R_new, P_new, X_new, adj)
            P.add(v)
            X.discard(v)
        
        return len(best_clique_set[0])
    
    best_clique_set = [set()]
    all_vertices = set(range(N))
    bron_kerbosch_track(set(), all_vertices, set(), adj)
    
    # Convert back to numbers
    result = sorted([v+1 for v in best_clique_set[0]])
    return result
