"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    vertices = list(range(n))
    
    all_4sets = list(itertools.combinations(vertices, 4))
    
    def make_mask(s):
        m = 0
        for x in s:
            m |= (1 << x)
        return m
    
    masks = [make_mask(s) for s in all_4sets]
    
    disjoint_pairs = []
    for i in range(len(all_4sets)):
        Am = masks[i]
        for j in range(i+1, len(all_4sets)):
            Bm = masks[j]
            if not (Am & Bm):
                disjoint_pairs.append((all_4sets[i], all_4sets[j], Am, Bm))
    
    def is_q3_free(adj):
        for (A, B, Am, Bm) in disjoint_pairs:
            ok = True
            for a in A:
                if bin(Bm & ~adj[a]).count('1') > 1:
                    ok = False
                    break
            if not ok:
                continue
            for b in B:
                if bin(Am & ~adj[b]).count('1') > 1:
                    ok = False
                    break
            if ok:
                return False
        return True
    
    # Generate candidates
    candidates = []
    
    # K_{3,13}
    candidates.append([(a, b) for a in range(3) for b in range(3, 16)])
    
    # Random graphs
    for seed in range(30):
        if time.time() > deadline - 30.0:
            break
        random.seed(seed)
        edges = []
        for (u, v) in all_edges:
            if random.random() < 0.55:
                edges.append((u, v))
        candidates.append(edges)
    
    # 3-partite K_{4,4,8}
    p1, p2, p3 = set(range(4)), set(range(4,8)), set(range(8,16))
    e3 = []
    for i in range(16):
        for j in range(i+1, 16):
            pi = 0 if i in p1 else (1 if i in p2 else 2)
            pj = 0 if j in p1 else (1 if j in p2 else 2)
            if pi != pj:
                e3.append((i, j))
    candidates.append(e3)
    
    best_edges = None
    best_count = 0
    
    for edges in candidates:
        if time.time() > deadline - 10.0:
            break
        adj = [0] * n
        for (u, v) in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        if is_q3_free(adj):
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = edges[:]
    
    if best_edges is None:
        best_edges = candidates[0]
    
    # Local improvement
    if time.time() < deadline - 15.0:
        adj = [0] * n
        edge_set = set(best_edges)
        for (u, v) in best_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        for (u, v) in all_edges:
            if time.time() > deadline - 5.0:
                break
            if (u, v) in edge_set:
                continue
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            if is_q3_free(adj):
                edge_set.add((u, v))
                best_edges.append((u, v))
            else:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
    
    return [[u, v] for u, v in best_edges]

# EVOLVE-BLOCK-END
