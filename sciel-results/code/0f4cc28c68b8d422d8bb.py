import time
import random
from itertools import combinations

N = 16
MAX_EDGES = N * (N - 1) // 2

all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
edge_idx = {e: i for i, e in enumerate(all_edges)}

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    # Start with a strong seed: the complete graph minus a perfect matching?
    # Or use a known Q3-free construction.
    
    # K16 has 120 edges. We need to remove edges to destroy all Q3's.
    # Each Q3 has 12 edges. How many Q3's are in K16?
    # Number of labeled Q3's in K16: choose 8 vertices, then number of Q3 labelings.
    # Number of ways to label 8 vertices as Q3: 8! / |Aut(Q3)| ... 
    # Actually, the number of subgraphs isomorphic to Q3 in K_n is C(n,8) * (8! / 8!) ... 
    # Let's just compute: for each 8-subset, how many Q3 subgraphs?
    # The number of distinct Q3's on 8 labeled vertices is 8! / |Aut(Q3)|.
    # |Aut(Q3)| = 48 (the symmetry group of the cube).
    # So C(16,8) * 48 = 12870 * 48 = 617,760 Q3's in K16.
    
    # We need to hit all of these. This is a hitting set problem.
    
    # Let's try a randomized local search:
    # Start with K16, remove edges to destroy Q3's.
    
    import numpy as np
    
    # Start with all edges present
    adj = np.ones((N, N), dtype=bool)
    np.fill_diagonal(adj, False)
    
    # Find all Q3's and remove edges
    # For each Q3, remove one edge from it
    
    def find_q3(adj):
        """Find one Q3 in the graph, return its 8 vertices or None."""
        for vertices in combinations(range(N), 8):
            # Check if induced subgraph is Q3
            # Q3: 8 vertices, 12 edges, 3-regular, bipartite with parts 4,4
            degs = []
            for v in vertices:
                d = sum(1 for u in vertices if u != v and adj[v][u])
                if d != 3:
                    degs = None
                    break
                degs.append(d)
            if degs is None:
                continue
            
            # Check bipartite
            color = {}
            queue = [vertices[0]]
            color[vertices[0]] = 0
            bip = True
            while queue:
                u = queue.pop(0)
                for w in vertices:
                    if w != u and adj[u][w]:
                        if w not in color:
                            color[w] = 1 - color[u]
                            queue.append(w)
                        elif color[w] == color[u]:
                            bip = False
                            break
                if not bip:
                    break
            if not bip:
                continue
            
            # Now check if it's actually Q3 (not just any 3-regular bipartite on 8 vertices)
            # Q3 is the 3-cube. Let's check by trying to map it.
            # Q3 adjacency in binary: v ~ v^1, v^2, v^4
            q3_edges = set()
            for v in range(8):
                for w in [v^1, v^2, v^4]:
                    if v < w:
                        q3_edges.add((v, w))
            
            # Try to find isomorphism
            mapping = [None] * 8
            used = set()
            
            def bt(i):
                if i == 8:
                    return True
                for v in vertices:
                    if v in used:
                        continue
                    ok = True
                    for j in range(i):
                        if j in [v2^1, v2^2, v2^4] if False else True:
                            pass
                        # Check edge consistency
                        for k in range(i):
                            should_be_adj = (mapping[k] in [v2^1, v2^2, v2^4]) if False else None
                        # This is getting complicated, let's simplify
                        should = 1 if (mapping[k] in [(i^1), (i^2), (i^4)]) else 0
                        actual = 1 if adj[mapping[k]][v] else 0
                        if should != actual:
                            ok = False
                            break
                    if not ok:
                        continue
                    used.add(v)
                    mapping[i] = v
                    if bt(i + 1):
                        return True
                    mapping[i] = None
                    used.remove(v)
                return False
            
            if bt(0):
                return list(vertices)
        return None
    
    # Remove edges to destroy Q3's
    for iteration in range(500):
        if time.time() > deadline:
            break
        q3 = find_q3(adj)
        if q3 is None:
            break
        # Remove a random edge from this Q3
        random.shuffle(q3)
        # Find an edge in the Q3 to remove
        removed = False
        for i in range(8):
            for j in range(i+1, 8):
                u, v = q3[i], q3[j]
                if adj[u][v]:
                    adj[u][v] = False
                    adj[v][u] = False
                    removed = True
                    break
            if removed:
                break
    
    # Count edges
    edge_count = sum(adj[i][j] for i in range(N) for j in range(i+1, N))
    
    # Try to add back some edges
    random.seed(42)
    edge_order = list(range(MAX_EDGES))
    random.shuffle(edge_order)
    
    for eidx in edge_order:
        if time.time() > deadline:
            break
        u, v = all_edges[eidx]
        if adj[u][v]:
            continue
        adj[u][v] = True
        adj[v][u] = True
        if find_q3(adj) is not None:
            adj[u][v] = False
            adj[v][u] = False
    
    result = []
    for i in range(N):
        for j in range(i + 1, N):
            if adj[i][j]:
                result.append([i, j])
    
    return result
