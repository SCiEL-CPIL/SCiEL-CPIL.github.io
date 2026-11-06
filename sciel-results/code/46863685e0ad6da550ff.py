import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin for safety
    
    n = 16
    # Q3 edges
    q3_edges = [(0,1), (0,2), (0,4), (1,3), (1,5), (2,3), (2,6), (3,7), (4,5), (4,6), (5,7), (6,7)]
    q3_adj = [0] * 8
    for u, v in q3_edges:
        q3_adj[u] |= (1 << v)
        q3_adj[v] |= (1 << u)
        
    # Precompute all 8-vertex subsets
    all_subsets = list(combinations(range(n), 8))
    
    # Precompute subsets containing each pair
    subsets_containing = [[[] for _ in range(n)] for _ in range(n)]
    for idx, subset in enumerate(all_subsets):
        for i in range(8):
            for j in range(i+1, 8):
                u, v = subset[i], subset[j]
                subsets_containing[u][v].append(idx)
                subsets_containing[v][u].append(idx)
                
    # Graph representation: list of bitmasks
    adj = [0] * n
    
    # Initialize K_3,13
    for a in range(3):
        mask = 0
        for b in range(3, 16):
            mask |= (1 << b)
        adj[a] = mask
        for b in range(3, 16):
            adj[b] |= (1 << a)
            
    current_edges = 39
    best_edges = current_edges
    best_adj = adj[:]
    
    def subset_has_q3(subset_idx, adj):
        subset = all_subsets[subset_idx]
        local_adj = [0] * 8
        degs = [0] * 8
        
        for i in range(8):
            u = subset[i]
            for j in range(8):
                if i == j: continue
                v = subset[j]
                if adj[u] & (1 << v):
                    local_adj[i] |= (1 << j)
                    degs[i] += 1
                    
        for d in degs:
            if d < 3:
                return False
                
        # Backtracking
        mapping = [-1] * 8
        used_local = [False] * 8
        
        # Fix Q3 0 to Local 0
        mapping[0] = 0
        used_local[0] = True
        
        def backtrack(q3_idx):
            if q3_idx == 8:
                return True
            
            for l in range(8):
                if used_local[l]:
                    continue
                
                valid = True
                for n_idx in range(8):
                    if n_idx == q3_idx: continue
                    if mapping[n_idx] == -1: continue
                    
                    is_adj_q3 = (q3_adj[q3_idx] & (1 << n_idx)) != 0
                    is_adj_local = (local_adj[l] & (1 << mapping[n_idx])) != 0
                    
                    if is_adj_q3 != is_adj_local:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                mapping[q3_idx] = l
                used_local[l] = True
                
                if backtrack(q3_idx + 1):
                    return True
                    
                mapping[q3_idx] = -1
                used_local[l] = False
                
            return False
        
        return backtrack(1)

    # List of non-edges
    non_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if not (adj[i] & (1 << j)):
                non_edges.append((i, j))
                
    random.shuffle(non_edges)
    
    for u, v in non_edges:
        if time.time() > deadline:
            break
            
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
        is_free = True
        for subset_idx in subsets_containing[u][v]:
            if subset_has_q3(subset_idx, adj):
                is_free = False
                break
                
        if is_free:
            current_edges += 1
            if current_edges > best_edges:
                best_edges = current_edges
                best_adj = adj[:]
        else:
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i] & (1 << j):
                edges.append([i, j])
                
    return edges
