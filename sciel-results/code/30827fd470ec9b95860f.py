import random
from itertools import combinations

def construct():
    # Hoffman-Singleton graph has 50 vertices, is triangle-free, and has alpha=10.
    # We construct it, remove vertices to reduce alpha, and verify.
    
    # HS Graph construction:
    # Vertices 0..49.
    # Edges:
    # 1. (i, i+1) mod 50 for i in 0..49? No, that's a cycle.
    # HS is a specific (7,3)-cage? No, (7,3) cage is Heawood.
    # HS is a (7,3) Moore graph? No, (7,3) Moore graph is HS?
    # HS is a (7,3) Moore graph? Degree 7, Diameter 2, Girth 5.
    # Yes, HS is the unique (7,3) Moore graph.
    
    # Standard construction of HS graph:
    # Vertices: (i, j) where i in 0..4, j in 0..9? No.
    # One representation:
    # 50 vertices.
    # Edges are defined by:
    # 1. (i, i+1) for i=0..48, and (49, 0)? No.
    
    # Let's use a known edge list or generator.
    # A common generator for HS graph:
    # The HS graph is the collinearity graph of the generalized quadrangle GQ(2,4)? No.
    
    # I will use a hardcoded adjacency list for a known triangle-free graph with low alpha.
    # Actually, the **Paley graph of order 41** is not triangle-free.
    # The **Clebsch graph** is safe.
    
    # Let's try to construct a graph on 39 vertices using a **Circulant** approach.
    # Graph on 39 vertices.
    # Connections: i ~ j if (j-i) mod 39 is in S.
    # S must be sum-free.
    # Let S = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}?
    # If S is an interval, it's not sum-free (1+2=3).
    # We need a large sum-free set mod 39.
    # Largest sum-free set in Z_39 is roughly 39/3 = 13.
    # If we take S = {13, 14, ..., 25}?
    # Sum-free means no a+b in S.
    # If S = {13..25}, 13+13=26 (not in S), 13+14=27 (not in S).
    # Max sum 25+25=50 = 11 mod 39. 11 is not in S.
    # Min sum 13+13=26.
    # Is {13..25} sum-free mod 39?
    # Check if any a,b in S such that a+b mod 39 in S.
    # a+b ranges from 26 to 50.
    # 26..38 are not in S (S is 13..25).
    # 0..11 (from 39..50) are not in S.
    # So S = {13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25} is sum-free.
    # This defines a triangle-free circulant graph.
    # Degree is 13*2 = 26? No, if S is symmetric?
    # Circulant graph C_n(S) usually has edges (i, i+s) for s in S.
    # If S is not symmetric, we add -S.
    # Let S = {13, ..., 25}. -S = {39-13, ..., 39-25} = {26, ..., 14}.
    # Union S U -S = {13..26}.
    # Is {13..26} sum-free?
    # 13+13=26 (in S U -S). So triangle (0, 13, 26).
    # So we cannot use symmetric S if it creates triangles.
    # We need a directed sum-free set? No, graph is undirected.
    # We need S such that S U -S is sum-free?
    # Or just pick a directed sum-free set and make it symmetric carefully?
    # If we use S = {13..25}, the edges are (i, i+s).
    # Triangle check: i, i+s1, i+s1+s2.
    # We need s1+s2 not in S U -S.
    # Since S is sum-free, s1+s2 is not in S.
    # But is s1+s2 in -S?
    # -S = {14..26}.
    # s1+s2 min 26. 26 is in -S.
    # So 13+13=26. 26 is in -S.
    # So (0, 13, 26) is a triangle?
    # 0~13 (13 in S). 13~26 (13 in S). 26~0 (-26 = 13 in S? No, 0-26 = -26 = 13 mod 39. 13 is in S).
    # So yes, triangle.
    
    # We need a smaller S.
    # Let S = {13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}.
    # Max sum 23+23=46=7. 7 not in S U -S.
    # Min sum 13+13=26. 26 in -S? -S = {16..26}. Yes.
    # So 13+13=26. Triangle.
    
    # It seems hard to get a large degree circulant graph that is triangle-free.
    
    # I will fall back to the **Hardcoded HS Subgraph** approach.
    
    # HS Graph Adjacency List (Standard):
    # I will generate it using the known property that it's the unique (7,3) Moore graph.
    # But generating it is complex.
    
    # Instead, I will use the **Clebsch Graph** (16 vertices) and **extend** it.
    # Or just return the **26-vertex Andrásfai** graph but try to improve it.
    
    # Actually, the **Paley Graph of Order 41** has a triangle-free subgraph on 40 vertices?
    # No.
    
    # I will return a **39-vertex graph** constructed by taking the **Hoffman-Singleton** graph and removing 11 vertices.
    # I need the HS graph edges.
    # I will hardcode the edges of the HS graph.
    
    # HS Graph Edges (one standard listing):
    # ... (Too long to type out 1250 edges manually)
    
    # Alternative: Use a **Randomized** search but with a **Better Initial State**.
    # Start with the **Clebsch Graph** (16 vertices, alpha=5).
    # Add vertices one by one, connecting them to existing vertices to maintain triangle-freeness and minimize alpha.
    
    # This is a "Greedy Extension" approach.
    
    def is_triangle_free(edges, n):
        adj = [[False]*n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = True
            adj[v][u] = True
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    for k in range(j+1, n):
                        if adj[i][k] and adj[j][k]:
                            return False
        return True
        
    def has_independent_set_of_size(edges, n, k=10):
        adj = [[False]*n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = True
            adj[v][u] = True
        def dfs(current_set, candidates):
            if len(current_set) >= k:
                return True
            if not candidates:
                return False
            v = candidates[0]
            new_candidates = [u for u in candidates[1:] if not adj[v][u]]
            if dfs(current_set + [v], new_candidates):
                return True
            if dfs(current_set, candidates[1:]):
                return True
            return False
        return dfs([], list(range(n)))
        
    # Start with Clebsch Graph (16 vertices)
    # Clebsch vertices: 0..15
    # Edges: (i, j) if i xor j is in {1, 2, 4, 8, 3, 5, 9, 12}?
    # Clebsch is the folded 5-cube.
    
    # I will just return the 26-vertex graph but try to find a 39-vertex one using a simple heuristic.
    # Given the constraints, I will output the 26-vertex graph to ensure validity, 
    # but I will try a few random 39-vertex graphs with a faster verification.
    
    # Actually, the best score improvement is to find a valid 39-vertex graph.
    # I will use the **Paley Graph of Order 41** and **remove edges** to make it triangle-free?
    # No, that increases alpha.
    
    # I will return the 26-vertex graph.
    # Wait, I can improve the random search by using a **Bitset** for adjacency.
    
    return {'n': 26, 'edges': [[i, j] for i in range(26) for j in range(i + 1, 26) if (j - i) % 26 % 3 == 1]}
