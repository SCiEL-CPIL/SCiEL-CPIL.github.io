import time

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    edge_set = set()
    for u, v in P['edges']:
        e = (min(u,v), max(u,v))
        edge_set.add(e)
    
    # Greedy: pick largest biclique from remaining edges
    def greedy_partition():
        remaining = set(edge_set)
        bicliques = []
        
        while remaining:
            verts = set()
            for e in remaining:
                verts.add(e[0])
                verts.add(e[1])
            
            best_L = None
            best_R = None
            best_count = 0
            
            for u in verts:
                right_cands = set()
                for e in remaining:
                    if e[0] == u:
                        right_cands.add(e[1])
                    elif e[1] == u:
                        right_cands.add(e[0])
                
                if not right_cands:
                    continue
                
                # Greedy max clique in right_cands within remaining
                # Simple greedy: iteratively add vertices that connect to all existing
                rc_list = list(right_cands)
                best_subset = set()
                
                # Try starting from each vertex
                for start in rc_list:
                    current = {start}
                    for v in rc_list:
                        if v == start:
                            continue
                        # Check if v connects to all in current
                        ok = True
                        for w in current:
                            e = (min(v,w), max(v,w))
                            if e not in remaining:
                                ok = False
                                break
                        if ok:
                            current.add(v)
                    if len(current) > len(best_subset):
                        best_subset = current
                
                if len(best_subset) > best_count:
                    best_count = len(best_subset)
                    best_L = [u]
                    best_R = list(best_subset)
            
            if best_L is None:
                e = next(iter(remaining))
                best_L = [e[0]]
                best_R = [e[1]]
            
            for u in best_L:
                for v in best_R:
                    e = (min(u,v), max(u,v))
                    remaining.discard(e)
            
            bicliques.append((best_L, best_R))
        
        return bicliques
    
    bicliques = greedy_partition()
    best = [list(b) for b in bicliques]
    best_count = len(best)
    
    # Local search: try to merge bicliques
    def can_merge(bi, bj):
        Li, Ri = bi
        Lj, Rj = bj
        new_L = set(Li) | set(Lj)
        new_R = set(Ri) | set(Rj)
        if new_L & new_R:
            return False
        for u in new_L:
            for v in new_R:
                if v not in adj[u]:
                    return False
        merged_edges = set()
        for u in new_L:
            for v in new_R:
                merged_edges.add((min(u,v), max(u,v)))
        edges_i = set()
        for u in Li:
            for v in Ri:
                edges_i.add((min(u,v), max(u,v)))
        edges_j = set()
        for u in Lj:
            for v in Rj:
                edges_j.add((min(u,v), max(u,v)))
        return merged_edges == edges_i | edges_j
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(bicliques)):
            if time.time() > deadline:
                break
            for j in range(i+1, len(bicliques)):
                if can_merge(bicliques[i], bicliques[j]):
                    Li, Ri = bicliques[i]
                    Lj, Rj = bicliques[j]
                    bicliques[i] = (list(set(Li)|set(Lj)), list(set(Ri)|set(Rj)))
                    bicliques.pop(j)
                    improved = True
                    break
            if improved:
                break
        if len(bicliques) < best_count:
            best = [list(b) for b in bicliques]
            best_count = len(bicliques)
    
    result = []
    for L, R in best:
        result.append([list(L), list(R)])
    return result
