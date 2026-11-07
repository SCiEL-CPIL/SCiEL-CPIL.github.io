import time

def construct():
    start_time = time.time()
    deadline = 50.0
    
    P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}
    
    n = P['n']
    edges = P['edges']
    
    remaining = set(map(tuple, edges))
    
    out_neigh = [set() for _ in range(n)]
    in_neigh = [set() for _ in range(n)]
    
    for u, v in edges:
        out_neigh[u].add(v)
        in_neigh[v].add(u)
        
    solution = []
    
    def find_best_biclique():
        best_L = None
        best_R = None
        best_size = -1
        
        candidates = [u for u in range(n) if out_neigh[u]]
        candidates.sort(key=lambda x: len(out_neigh[x]), reverse=True)
        
        for u in candidates:
            if not out_neigh[u]:
                continue
                
            L = {u}
            R = set(out_neigh[u])
            
            # Alternating expansion
            for _ in range(10):
                # Expand L
                old_L_size = len(L)
                # Find all v such that R is subset of out_neigh[v]
                # Optimization: only check vertices with out-degree >= len(R)
                for v in range(n):
                    if v not in L and len(out_neigh[v]) >= len(R) and R.issubset(out_neigh[v]):
                        L.add(v)
                if len(L) == old_L_size:
                    break
                
                # Update R
                if L:
                    R = set(out_neigh[list(L)[0]])
                    for v in L:
                        R.intersection_update(out_neigh[v])
                else:
                    R = set()
                    break
                
                # Expand R? 
                # The definition of biclique L x R subset E means:
                # 1. For all u in L, v in R, (u, v) in E.
                # Expanding R: add w if for all u in L, (u, w) in E.
                # This is equivalent to: N^+(L) contains w.
                # i.e., w is in intersection of out_neigh[u] for all u in L.
                # Which is exactly how we calculated R.
                # So R is already maximal for the current L.
                # And L is maximal for the current R.
                # The loop converges.
                
            size = len(L) * len(R)
            if size > best_size:
                best_size = size
                best_L = L
                best_R = R
                
        return best_L, best_R

    while remaining:
        if time.time() > deadline:
            break
            
        L, R = find_best_biclique()
        
        if not L or not R:
            edge = next(iter(remaining))
            L = {edge[0]}
            R = {edge[1]}
            
        solution.append([sorted(L), sorted(R)])
        
        for u in L:
            for v in R:
                remaining.discard((u, v))
                
        for u in L:
            for v in R:
                if v in out_neigh[u]:
                    out_neigh[u].discard(v)
                if u in in_neigh[v]:
                    in_neigh[v].discard(u)
                    
    if remaining:
        for u, v in remaining:
            solution.append([u, [v]])
            
    final_solution = []
    for biclique in solution:
        left = [int(x) for x in biclique[0]]
        right = [int(x) for x in biclique[1]]
        final_solution.append([left, right])
        
    return final_solution
