import random
from itertools import combinations

def construct():
    n = 18
    
    def is_k4_free(adj):
        # adj: list of bitmasks
        for u in range(n):
            for v in range(u+1, n):
                if adj[u] & (1 << v):
                    for w in range(v+1, n):
                        if adj[v] & (1 << w):
                            if adj[w] & (1 << u):
                                return False
        return True
    
    def max_triangle_free_size(adj):
        # adj: list of bitmasks
        # Check all subsets of size k from n down to 1
        # If any subset of size k is triangle-free, return k
        for k in range(n, 0, -1):
            for subset in combinations(range(n), k):
                # Check if subset is triangle-free
                is_tf = True
                for i in range(k):
                    for j in range(i+1, k):
                        u = subset[i]
                        v = subset[j]
                        if adj[u] & (1 << v):
                            # Edge exists, check if it forms a triangle with any other vertex in subset
                            # Actually, we just need to check if there is ANY triangle.
                            # If there's an edge, we need to check if there's a common neighbor in the subset.
                            # Let's just check all triples.
                            pass
                
                # Efficient check:
                # A subset is triangle-free if no three vertices form a triangle.
                # We can iterate over all triples in the subset.
                # But that's O(k^3).
                # Total complexity: C(n,k) * k^3.
                # For k=12, C(18,12)=18564, k^3=1728. ~32M ops. Too slow in Python.
                
                # Let's use a different approach for the check.
                # Precompute for each subset? No.
                # Let's just do the triple check but break early.
                is_tf = True
                for i in range(k):
                    for j in range(i+1, k):
                        u = subset[i]
                        v = subset[j]
                        if adj[u] & (1 << v):
                            # Check if there is a w in subset such that w is connected to both u and v
                            common = adj[u] & adj[v]
                            # Mask to only include vertices in subset
                            mask = 0
                            for x in subset:
                                mask |= (1 << x)
                            if common & mask:
                                is_tf = False
                                break
                    if not is_tf:
                        break
                if is_tf:
                    return k
        return 0
    
    # Generate candidates
    candidates = []
    
    # 1. Seed: K6,6,6 (complement of 3 disjoint K6s)
    # G is K6,6,6. Edges between different parts.
    adj_seed = [0]*n
    parts = [list(range(0,6)), list(range(6,12)), list(range(12,18))]
    for p1 in range(3):
        for p2 in range(p1+1, 3):
            for u in parts[p1]:
                for v in parts[p2]:
                    adj_seed[u] |= (1 << v)
                    adj_seed[v] |= (1 << u)
    candidates.append(adj_seed)
    
    # 2. Random K4-free graphs
    rng = random.Random(42)
    for _ in range(20):
        # Start with random graph
        adj = [0]*n
        for u in range(n):
            for v in range(u+1, n):
                if rng.random() < 0.5:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        # Remove edges to make K4-free
        # Find K4s and remove an edge
        changed = True
        while changed:
            changed = False
            for u in range(n):
                for v in range(u+1, n):
                    if adj[u] & (1 << v):
                        for w in range(v+1, n):
                            if adj[v] & (1 << w):
                                if adj[w] & (1 << u):
                                    # Found K4: u,v,w,x? No, u,v,w is a triangle.
                                    # K4 requires 4 vertices.
                                    # Check if there is x connected to u,v,w.
                                    common = adj[u] & adj[v] & adj[w]
                                    # x must be > w? No, x can be any other vertex.
                                    # But we are iterating u<v<w.
                                    # Let's just check all x.
                                    for x in range(n):
                                        if x != u and x != v and x != w and (common & (1 << x)):
                                            # Found K4: u,v,w,x
                                            # Remove a random edge
                                            # Edges: uv, uw, ux, vw, vx, wx
                                            # Pick random
                                            r = rng.random()
                                            if r < 1/6:
                                                adj[u] &= ~(1 << v); adj[v] &= ~(1 << u)
                                            elif r < 2/6:
                                                adj[u] &= ~(1 << w); adj[w] &= ~(1 << u)
                                            elif r < 3/6:
                                                adj[u] &= ~(1 << x); adj[x] &= ~(1 << u)
                                            elif r < 4/6:
                                                adj[v] &= ~(1 << w); adj[w] &= ~(1 << v)
                                            elif r < 5/6:
                                                adj[v] &= ~(1 << x); adj[x] &= ~(1 << v)
                                            else:
                                                adj[w] &= ~(1 << x); adj[x] &= ~(1 << w)
                                            changed = True
                                            break
                                    if changed:
                                        break
                                # Optimization: if we found a triangle, check for K4
                                # But the loop above checks x for each triangle.
                                # This is O(n^4) which is 18^4 = 104976.
                                # But we break early.
                        if changed:
                            break
                    if changed:
                        break
                if changed:
                    break
        
        if is_k4_free(adj):
            candidates.append(adj)
    
    # 3. Blow-ups of small K4-free graphs
    # Base graph: K4 minus edge (4 vertices)
    # Blow up each vertex to 4.5? No, must be integer.
    # 18 = 4+4+4+6? Or 5+5+4+4?
    # Let's try base graph C5 (5 vertices) -> blow up to 18? 5*3=15, 5*4=20. No.
    # Base graph: 3-vertex path? No, K4-free.
    # Let's stick to random and seed.
    
    best_score = float('inf')
    best_adj = None
    
    for adj in candidates:
        score = max_triangle_free_size(adj)
        if score < best_score:
            best_score = score
            best_adj = adj
    
    # Convert best_adj to edges
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if best_adj[u] & (1 << v):
                edges.append([u, v])
    
    return edges
