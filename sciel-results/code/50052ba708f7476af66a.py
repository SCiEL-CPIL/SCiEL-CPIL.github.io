import time
import numpy as np
from itertools import combinations

def _has_c5(adj: list[set]) -> bool:
    """Exact check for a C5 subgraph."""
    n = len(adj)
    for v0 in range(n):
        for v4 in adj[v0]:
            if v4 <= v0:
                continue
            n_v0 = adj[v0] - {v4}
            n_v4 = adj[v4] - {v0}
            for v1 in n_v0:
                for v3 in n_v4:
                    if v1 == v3:
                        continue
                    common = adj[v1] & adj[v3]
                    common -= {v0, v1, v3, v4}
                    if common:
                        return True
    return False

def _independent_set_size(adj: list[set], target: int) -> bool:
    """Check if there exists an independent set of size >= target."""
    n = len(adj)
    comp = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if j not in adj[i]:
                comp[i].add(j)
                comp[j].add(i)
    
    found = [False]
    
    def search(clique_sz, candidates):
        if found[0]:
            return
        if clique_sz >= target:
            found[0] = True
            return
        if not candidates:
            return
        if clique_sz + len(candidates) < target:
            return
        v = max(candidates, key=lambda x: len(comp[x] & candidates))
        nc = candidates & comp[v]
        search(clique_sz + 1, nc)
        if found[0]:
            return
        c2 = candidates - {v}
        search(clique_sz, c2)
    
    search(0, frozenset(range(n)), )
    return found[0]

def _build_clebsch_plus_one():
    """
    Build the Clebsch graph (31 vertices, C5-free, alpha=8) plus one extra vertex.
    Clebsch graph: vertices are 5-bit strings with even parity (weight 0,2,4).
    Two vertices are adjacent if they differ in exactly 2 positions.
    This gives a strongly regular graph (31,12,5,6), C5-free, alpha=8.
    Add vertex 31 connected to all vertices of weight 2 (10 vertices).
    Result: 32 vertices, still C5-free (extra vertex has no common neighbors forming C5),
    alpha <= 8 (any independent set using vertex 31 can use at most 8-1=7 others from non-neighbors... 
    actually we need to verify).
    """
    # Generate all 5-bit strings with even parity
    vertices = []
    for bits in range(32):
        if bin(bits).count('1') % 2 == 0:
            vertices.append(bits)
    # vertices has 16 elements (weight 0, 2, or 4)
    # Wait: even parity 5-bit strings: C(5,0)+C(5,2)+C(5,4) = 1+10+5 = 16
    # That's not 31. Let me reconsider.
    # 
    # The Clebsch graph has 31 vertices. It's the complement of the Shrikhande graph... no.
    # Actually the Clebsch graph is the folded 5-cube, which has 2^4 = 16 vertices... no.
    # 
    # Let me use a different approach. The Clebsch graph is the graph on 31 vertices where
    # vertices are nonzero vectors in GF(2)^5, and two are adjacent if their dot product is 1.
    # This is the (31,12,5,6) strongly regular graph.
    
    n_clebsch = 31
    vertices = list(range(1, 32))  # nonzero 5-bit vectors
    
    edges = []
    for i in range(n_clebsch):
        for j in range(i+1, n_clebsch):
            # Dot product mod 2
            x = vertices[i]
            y = vertices[j]
            dot = bin(x & y).count('1') % 2
            if dot == 1:
                edges.append([i, j])
    
    # Now add vertex 31. Connect it to some subset to keep C5-free and alpha < 9.
    # The Clebsch graph has alpha = 8. If we add a vertex connected to all, alpha stays 8.
    # But we need to check C5-freeness. If vertex 31 is connected to all 31 vertices,
    # any C5 through vertex 31 would need 4 other vertices forming a P4 in the Clebsch graph
    # with endpoints both adjacent to 31 (which they are). So we need no P4 in Clebsch...
    # Actually a C5 through v31: v31-a-b-c-v31. Need a,c both adjacent to v31 (yes, all are),
    # a-b edge, b-c edge, and a != c. So any path of length 2 in Clebsch gives a C5.
    # The Clebsch graph certainly has paths of length 2, so connecting to all creates C5s.
    
    # Instead, connect vertex 31 to a small set. If we connect to 0 vertices, alpha could be 9.
    # Connect to a maximal set that doesn't create C5s.
    # 
    # Alternative: just use the 8xK4 construction which is known to work.
    # But let's try: connect vertex 31 to vertices of weight 1 (5 vertices: 1,2,4,8,16).
    # Check if this creates C5s.
    
    weight1 = [1, 2, 4, 8, 16]
    for w in weight1:
        edges.append([31, w])
    
    n = 32
    return n, edges

def construct():
    start = time.time()
    
    # Primary: algebraic construction (Clebsch + 1 vertex)
    n, edges = _build_clebsch_plus_one()
    
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    if not _has_c5(adj) and not _independent_set_size(adj, 9):
        return {'n': n, 'edges': edges}
    
    # Fallback: 8 disjoint K4s (known valid, n=32)
    n = 32
    edges = []
    for b in range(8):
        base = b * 4
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
    
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    if not _has_c5(adj) and not _independent_set_size(adj, 9):
        return {'n': n, 'edges': edges}
    else:
        return {'n': 0, 'edges': []}
