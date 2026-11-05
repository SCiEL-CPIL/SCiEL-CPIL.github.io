"""Construct a C4-free graph with high minimum degree to certify R(C4,K1,39) > n."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict

def construct():
    # Target: n=46, min degree >= 46-39 = 7
    # Use polar graph construction for q=6 as base (n=43, min degree=7),
    # then extend to 46 vertices.
    
    def build_polar_graph(q):
        """Build polar graph for prime power q. n = q^2 + q + 1, min degree = q+1."""
        n = q * q + q + 1
        # Points are (x, y) in F_q^2 plus one point at infinity
        # Use a simple representation: points 0..q^2-1 are (i//q, i%q), point q^2 is infinity
        # Adjacency: two points are adjacent if the line through them is tangent to the conic
        # For simplicity, use the standard polar graph of the hyperbolic quadric in PG(3,q)
        # But easier: use the construction where vertices are 1-dim subspaces of F_q^3
        # Two vertices adjacent if the corresponding subspaces are orthogonal under a symplectic form
        
        # Simpler: use the known C4-free graph from the projective plane
        # Vertices: points of PG(2,q), edges: two points adjacent if... 
        # Actually, let's use the complement of the line graph of the projective plane
        
        # Most straightforward: for q prime, use the following
        # Vertices: 0..n-1 where n = q^2+q+1
        # Edge (i,j) exists based on a quadratic residue construction
        
        # Use the standard C4-free graph: vertices are elements of F_q^2 plus infinity
        # Two vertices (a,b) and (c,d) are adjacent iff (c-a)^2 + (d-b) is a non-zero square in F_q
        # This gives a C4-free graph with min degree q+1
        
        edges = []
        def is_square(x):
            if x == 0:
                return False
            for t in range(q):
                if (t * t) % q == x % q:
                    return True
            return False
        
        # Vertices 0..q^2-1 represent (i//q, i%q), vertex q^2 is infinity
        # For finite vertices (a,b) and (c,d): adjacent iff (c-a)^2 + (d-b) is a nonzero square
        # Infinity is adjacent to all vertices on the "line at infinity"
        # Actually, this construction needs care. Let me use a verified construction.
        
        # Use the construction from: C4-free graphs via finite fields
        # Vertices: F_q x F_q union {inf}
        # (x1,y1) ~ (x2,y2) iff (x1-x2)^2 + (y1-y2) is a nonzero quadratic residue mod q
        # inf ~ (x,y) for all (x,y)  -- this gives degree q^2 for inf, not balanced
        
        # Better: use the polarity graph of the parabola
        # Vertices: points on parabola y=x^2 in F_q^2, plus points at infinity
        # This is getting complex. Let me just use a known small example and verify.
        
        # For q=6 (not prime, but let's use q=5 which is prime): n=31, min degree=6
        # For q=7: n=57, too big
        # 
        # Let me use q=5: n=31, min degree=6. Then I need to extend to 46.
        # Or use a different approach entirely.
        
        # Alternative: just build a random-ish C4-free graph with min degree 7 on 46 vertices
        # using greedy edge addition with backtracking.
        return None  # We'll use a different approach below
    
    # Direct construction: greedy edge addition on 46 vertices
    # Start with empty graph, add edges while maintaining C4-freeness, 
    # aiming for min degree >= 7.
    
    n = 46
    min_deg_target = n - 39  # = 7
    
    # Use a structured starting point: 
    # Partition vertices into groups and use a blow-up of a small C4-free graph
    # 
    # Start with the 5-cycle C5 (C4-free, min degree 2), blow up each vertex to ~9 vertices
    # C5 blown up: 5 groups of 9, edges between consecutive groups (complete bipartite)
    # This is C4-free if the base is C4-free and we use complete bipartite between parts
    # Wait, complete bipartite K_{9,9} contains C4! So that doesn't work.
    #
    # Instead, use a C4-free base and match edges carefully.
    #
    # Better approach: use the polar graph for q=5 (n=31, min degree 6) 
    # and add 15 more vertices with appropriate edges.
    
    # Let me implement the polar graph for prime q using the standard construction:
    # For prime p, the graph on F_p^2 has vertices (x,y), and (x1,y1)~(x2,y2) 
    # iff (x1-x2)^2 + (y1-y2) is a nonzero square mod p.
    # This graph has n=p^2 vertices, and is C4-free with min degree ~ (p-1)/2.
    # For p=7: n=49, min degree ~3. Not enough.
    #
    # The correct polar graph has n = p^2 + p + 1 for prime p.
    # For p=5: n=31, min degree=6
    # For p=7: n=57, too large
    #
    # So max from polar graph is n=31 (p=5) or we need another construction.
    
    # Let me try: use p=5 polar graph (n=31, min degree 6) as core,
    # add 15 vertices, connect them to maintain C4-freeness and achieve min degree 7.
    
    # Actually, let me just do a smart greedy construction directly.
    
    # Greedy: build graph on n vertices, add edges in a way that avoids C4,
    # prioritize edges that increase the minimum degree.
    
    adj = [[False] * n for _ in range(n)]
    degree = [0] * n
    # For C4 detection: for each pair of vertices, track common neighbors
    common = [[0] * n for _ in range(n)]
    
    def can_add_edge(u, v):
        """Check if adding edge (u,v) creates a C4."""
        # A C4 is created if there exists a vertex w adjacent to both u and v
        # (then u-v-w-? no, C4 needs 4 distinct vertices)
        # Actually, adding (u,v) creates a C4 iff there exist distinct x,y 
        # such that x~u, x~v, y~u, y~v and x!=y -- no wait.
        # Adding edge (u,v) creates a C4 if there is a path of length 3 from u to v
        # that doesn't use edge (u,v). I.e., there exist x,y with 
        # u~x, x~y, y~v (and x!=v, y!=u, x!=y).
        # Equivalently, the number of common neighbors of u and v matters:
        # if u and v already have a common neighbor w, adding (u,v) creates triangle u-v-w, not C4.
        # C4: u-a-v-b-u. So adding (u,v) creates C4 if there exist a,b with 
        # u~a, a~v, v~b, b~u and a!=b, a!=v, b!=u.
        # This means: there exist a and b (distinct, not u or v) such that 
        # a is adjacent to both u and v, and b is adjacent to both u and v? No.
        # 
        # C4 after adding (u,v): vertices u, v, a, b with edges (u,v), (v,a), (a,b), (b,u).
        # So we need a adjacent to v and b adjacent to u and a adjacent to b, 
        # with a,b distinct from u,v and a!=b.
        # 
        # Simpler check: adding (u,v) creates a C4 iff there exist two distinct 
        # vertices a, b such that:
        #   a is adjacent to u and v (no, that would be triangle)
        # 
        # Let me think again. Current graph G, add edge (u,v). New C4 must use edge (u,v).
        # So C4 is u - v - x - y - u where x,y are existing vertices, x!=y, x!=u, y!=v,
        # and edges (v,x), (x,y), (y,u) exist in G.
        # So: exists x,y distinct, x in N(v), y in N(u), x!=u, y!=v, x!=y, and (x,y) in E(G).
        # 
        # Equivalent: exists x in N(v) \ {u} and y in N(u) \ {v} with x!=y and (x,y) in E.
        
        for x in range(n):
            if x == u or x == v:
                continue
            if not adj[v][x]:
                continue
            for y in range(n):
                if y == u or y == v or y == x:
                    continue
                if not adj[u][y]:
                    continue
                if adj[x][y]:
                    return False
        return True
    
    # This is O(n^2) per edge check, and we try O(n^2) edges, so O(n^4) total.
    # n=46, n^4 = 4.5M, should be fine.
    
    # Greedy: repeatedly add the edge that maximizes minimum degree improvement
    # while maintaining C4-freeness.
    
    # First, ensure all vertices have degree >= min_deg_target.
    # Start by adding edges to bring all vertices to at least min_deg_target.
    
    # Use a priority: add edges between low-degree vertices first.
    added = 0
    max_iterations = 10000
    it = 0
    
    while it < max_iterations:
        it += 1
        # Find the vertex with minimum degree
        min_deg = min(degree)
        if min_deg >= min_deg_target:
            break
        
        # Find the best edge to add: one that increases the minimum degree
        # and doesn't create a C4.
        best_edge = None
        best_score = -1
        
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u][v]:
                    continue
                if not can_add_edge(u, v):
                    continue
                # Score: how much does this help the minimum degree?
                # It increases degree[u] and degree[v] by 1.
                # Benefit is higher if u or v has low degree.
                score = min(degree[u], degree[v])  # higher is better (more urgent)
                # Actually, we want to fix the most under-degree vertex.
                # Score = how much the min degree improves.
                new_min = min(min_deg, min(degree[u] + 1, degree[v] + 1))
                # If u or v is the min-degree vertex, this helps.
                if degree[u] == min_deg or degree[v] == min_deg:
                    score = 1000 + min(degree[u], degree[v])
                else:
                    score = min(degree[u], degree[v])
                
                if score > best_score:
                    best_score = score
                    best_edge = (u, v)
        
        if best_edge is None:
            break
        
        u, v = best_edge
        adj[u][v] = adj[v][u] = True
        degree[u] += 1
        degree[v] += 1
        added += 1
    
    # Verify C4-freeness
    def has_c4():
        for u in range(n):
            for v in range(u + 1, n):
                if not adj[u][v]:
                    continue
                # Check if there's a C4 using edge (u,v)
                # u-v-x-y-u: x in N(v)\{u}, y in N(u)\{v}, x!=y, (x,y) in E
                for x in range(n):
                    if x == u or x == v or not adj[v][x]:
                        continue
                    for y in range(n):
                        if y == u or y == v or y == x or not adj[u][y]:
                            continue
                        if adj[x][y]:
                            return True
        return False
    
    if has_c4():
        # Fallback: use a smaller verified construction
        # Use the 5x5 grid torus? No, that has C4s.
        # Use the polar graph for p=5: n=31, min degree 6 = 31-25 >= 31-39? 
        # 31-39 = -8, so min degree >= -8 is trivially satisfied. 
        # But we want to maximize n.
        # For n=31, min degree >= 31-39 = -8, so any graph works.
        # But we want high n. Let's return n=39 (empty graph) as fallback.
        return {'n': 39, 'edges': []}
    
    # Build edge list
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u][v]:
                edges.append([u, v])
    
    # Verify min degree
    min_deg = min(degree)
    if min_deg < min_deg_target:
        # Try to fix by adding more edges
        # Or fall back
        return {'n': 39, 'edges': []}
    
    return {'n': n, 'edges': edges}

# EVOLVE-BLOCK-END
