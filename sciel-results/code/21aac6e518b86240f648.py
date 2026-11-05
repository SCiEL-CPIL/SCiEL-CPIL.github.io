import random
import time

def construct():
    d = 7
    n = 2 ** d
    total_edges = d * (n // 2)
    
    # Generate all edges
    edges = []
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                edges.append((x, y))
                
    # Precompute for each edge, the list of edges that form a C4 with it
    # Edge (u, v) with v = u ^ (1<<i)
    # C4 is formed by (u, u^j), (v, v^j), (u^j, v^j) where j != i
    # We need to check if these 3 edges are in the selected set.
    
    edge_to_idx = {e: i for i, e in enumerate(edges)}
    
    # For each edge, store the 3 edges that would complete a C4 with it
    # Actually, there are 6 pairs of such edges? No, for a fixed edge (u,v) along dimension i,
    # the other dimension j defines a unique C4.
    # The 3 other edges are (u, u^j), (v, v^j), (u^j, v^j).
    # So for each edge, there are d-1 = 6 such C4s.
    # We can store for each edge, a list of tuples of 3 edge indices.
    
    c4_conflicts = [[] for _ in range(len(edges))]
    for idx, (u, v) in enumerate(edges):
        # Determine dimension i
        diff = u ^ v
        i = diff.bit_length() - 1
        for j in range(d):
            if j == i:
                continue
            e1 = (min(u, u ^ (1 << j)), max(u, u ^ (1 << j)))
            e2 = (min(v, v ^ (1 << j)), max(v, v ^ (1 << j)))
            e3 = (min(u ^ (1 << j), v ^ (1 << j)), max(u ^ (1 << j), v ^ (1 << j)))
            
            idx1 = edge_to_idx.get(e1, -1)
            idx2 = edge_to_idx.get(e2, -1)
            idx3 = edge_to_idx.get(e3, -1)
            
            if idx1 != -1 and idx2 != -1 and idx3 != -1:
                c4_conflicts[idx].append((idx1, idx2, idx3))
                
    best_count = 0
    best_edges = []
    
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    while time.time() < deadline:
        random.shuffle(edges)
        selected = [False] * len(edges)
        count = 0
        
        for idx in range(len(edges)):
            ok = True
            for conflict in c4_conflicts[idx]:
                if selected[conflict[0]] and selected[conflict[1]] and selected[conflict[2]]:
                    ok = False
                    break
            if ok:
                selected[idx] = True
                count += 1
                
        if count > best_count:
            best_count = count
            best_edges = [edges[i] for i in range(len(edges)) if selected[i]]
            
    return [[u, v] for u, v in best_edges]
