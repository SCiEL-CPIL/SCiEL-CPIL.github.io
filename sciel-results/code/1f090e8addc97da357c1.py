import time
import random
import numpy as np

def _has_c5(adj: list[set]) -> bool:
    """Exact check for a C5 subgraph using path-of-length-4 enumeration."""
    n = len(adj)
    for v0 in range(n):
        n0 = adj[v0]
        for v1 in n0:
            n1 = adj[v1] - {v0}
            for v2 in n1:
                n2 = adj[v2] - {v0, v1}
                for v3 in n2:
                    n3 = adj[v3] - {v0, v1, v2}
                    if n3 & n0:
                        return True
    return False

def _has_independent_set(adj: list[set], target: int) -> bool:
    """Check if there exists an independent set of size >= target."""
    n = len(adj)
    comp = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if j not in adj[i]:
                comp[i].add(j)
                comp[j].add(i)
    
    found = [False]
    
    def search(sz, candidates):
        if found[0]:
            return
        if sz >= target:
            found[0] = True
            return
        if not candidates:
            return
        if sz + len(candidates) < target:
            return
        v = max(candidates, key=lambda x: len(comp[x] & candidates))
        search(sz + 1, candidates & comp[v])
        if found[0]:
            return
        search(sz, candidates - {v})
    
    search(0, frozenset(range(n)))
    return found[0]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Base: 8 disjoint K4s (n=32), alpha=8, C5-free
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
    
    best_n = n
    best_edges = edges[:]
    
    # Try to add vertex 32: connect to one vertex per K4 block
    # This is C5-safe: any two neighbors are in different K4s, so no path of length 3 exists between them
    # Alpha constraint: no independent set of size 8 in base is disjoint from S
    # An independent set of size 8 picks exactly one vertex per K4 block
    # So S = {s_0, s_1, ..., s_7} where s_b is in block b
    # We need: for every choice (c_0, ..., c_7) with c_b in block b, at least one c_b == s_b
    # This is ALWAYS true since each c_b must be in block b, and s_b is the only vertex from block b in S
    # Wait, that's wrong. c_b can be any of the 4 vertices in block b, and s_b is one specific one.
    # So an independent set (c_0,...,c_7) is disjoint from S iff c_b != s_b for all b.
    # There are 3^8 = 6561 such sets. So we CANNOT guarantee alpha < 9 with just one per block.
    
    # We need to connect to MORE vertices to block all 3^8 independent sets.
    # Strategy: connect to 2 vertices per block. Then an independent set disjoint from S 
    # must pick from the other 2 vertices per block: 2^8 = 256 possibilities.
    # Still not enough. Need to cover all independent sets.
    
    # Better approach: use the structure. In K4, any 2 vertices are adjacent, so an independent
    # set picks at most 1 per K4. To prevent an independent set of size 9 with vertex 32,
    # we need: no independent set of size 8 in base is disjoint from N(32).
    # An independent set of size 8 in base = one vertex per K4 block.
    # N(32) intersects each block in some subset S_b.
    # An independent set is disjoint from N(32) iff for each b, it picks a vertex NOT in S_b.
    # This is possible iff for each b, S_b != all 4 vertices of block b.
    # So we need: for at least one block b, S_b = all 4 vertices. But then vertex 32 connects
    # to all 4 in that block, creating triangles with 32, which is fine for C5.
    
    # Actually wait - if S_b = all 4 vertices of block b, then any independent set of size 8
    # must pick one from block b, and that vertex IS in S_b, so the independent set intersects S.
    # So connecting to ALL 4 vertices of at least one block guarantees alpha constraint!
    
    # But does it create a C5? Vertex 32 connects to all 4 in block 0.
    # A C5 involving 32: 32 - a - b - c - d - 32 where a,d in N(32).
    # If a, d are both in block 0: a-b-c-d requires a path of length 3 from a to d.
    # In K4, a and d are adjacent (distance 1). No path of length 3 within K4.
    # Could go a - (in block 0) - (outside) - (outside) - d? 
    # a is in block 0, adjacent to 3 other vertices in block 0.
    # From those, they're only adjacent to vertices in block 0 (disjoint K4s).
    # So no path of length 3 from a to d through outside vertices.
    # If a in block 0, d in block j (j>0): path a - x - y - d.
    # a-x: x in block 0 (only neighbors of a). x-y: y must be neighbor of x.
    # x is in block 0, so y in block 0. y-d: d in block j, y in block 0. No edge.
    # So NO C5 is created!
    
    # Connect vertex 32 to all 4 vertices of block 0.
    S = {0, 1, 2, 3}
    new_adj = [set(s) for s in adj] + [set(S)]
    for u in S:
        new_adj[u].add(32)
    
    if not _has_c5(new_adj) and not _has_independent_set(new_adj, 9):
        best_n = 33
        best_edges = edges[:] + [[32, u] for u in sorted(S)]
    
    # Try to add vertex 33 as well
    if best_n == 33:
        # Vertex 33: connect to all 4 of another block, say block 1
        S2 = {4, 5, 6, 7}
        new_adj2 = [set(s) for s in new_adj] + [set(S2)]
        for u in S2:
            new_adj2[u].add(33)
        
        if not _has_c5(new_adj2) and not _has_independent_set(new_adj2, 9):
            best_n = 34
            best_edges = best_edges[:] + [[33, u] for u in sorted(S2)]
    
    # Try vertex 34
    if best_n == 34:
        S3 = {8, 9, 10, 11}
        new_adj3 = [set(s) for s in new_adj2] + [set(S3)]
        for u in S3:
            new_adj3[u].add(34)
        
        if not _has_c5(new_adj3) and not _has_independent_set(new_adj3, 9):
            best_n = 35
            best_edges = best_edges[:] + [[34, u] for u in sorted(S3)]
    
    return {'n': best_n, 'edges': best_edges}
