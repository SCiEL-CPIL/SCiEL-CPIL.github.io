import numpy as np
import time
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    D = 11
    
    # Phase 1: All D_11 vectors (2-unit vectors) - all mutually compatible
    d11_vectors = []
    for i in range(D):
        for j in range(i + 1, D):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = np.zeros(D, dtype=np.int8)
                    v[i] = si
                    v[j] = sj
                    d11_vectors.append(v)
    
    # Phase 2: Find MIS among 4-unit supports
    supports = list(combinations(range(D), 4))
    n_sup = len(supports)  # 330
    
    # Build conflict: two supports conflict if |intersection| >= 3
    conflict = np.zeros((n_sup, n_sup), dtype=bool)
    for a in range(n_sup):
        sa = set(supports[a])
        for b in range(a + 1, n_sup):
            sb = set(supports[b])
            inter = len(sa & sb)
            if inter >= 3:
                conflict[a, b] = True
                conflict[b, a] = True
    
    # Greedy MIS with min-degree heuristic
    def greedy_mis(conflict_matrix, time_limit):
        n = len(conflict_matrix)
        in_mis = np.zeros(n, dtype=bool)
        remaining = set(range(n))
        deg = conflict_matrix.sum(axis=1).astype(int)
        
        while remaining and time.time() < time_limit:
            min_deg = min(deg[i] for i in remaining)
            candidates = [i for i in remaining if deg[i] == min_deg]
            node = candidates[0]
            
            in_mis[node] = True
            neighbors = set(np.where(conflict_matrix[node])[0])
            remaining -= neighbors
            remaining.discard(node)
            for i in remaining:
                deg[i] -= 1
        
        return in_mis
    
    # Local search: try to improve by swaps
    def local_search_mis(conflict_matrix, in_mis, time_limit):
        n = len(conflict_matrix)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try adding a node not in MIS
            for i in range(n):
                if in_mis[i]:
                    continue
                if not np.any(conflict_matrix[i] & in_mis):
                    in_mis[i] = True
                    improved = True
                    break
            
            if time.time() >= time_limit:
                break
            
            # Try 1-for-1 or 1-for-2 swaps
            for i in range(n):
                if not in_mis[i]:
                    continue
                # Temporarily remove i
                in_mis[i] = False
                # Find all nodes that can be added
                addable = []
                for j in range(n):
                    if in_mis[j]:
                        continue
                    if not np.any(conflict_matrix[j] & in_mis):
                        addable.append(j)
                if len(addable) > 1:
                    for j in addable:
                        in_mis[j] = True
                    improved = True
                else:
                    in_mis[i] = True
                if time.time() >= time_limit:
                    break
        
        return in_mis
    
    # Run greedy then local search
    mis_result = greedy_mis(conflict, deadline - 15.0)
    mis_result = local_search_mis(conflict, mis_result, deadline - 5.0)
    
    # Collect all vectors
    all_vectors = list(d11_vectors)
    
    for idx in range(n_sup):
        if mis_result[idx]:
            support = supports[idx]
            for signs in product([1, -1], repeat=4):
                v = np.zeros(D, dtype=np.int8)
                for s_idx, sign in zip(support, signs):
                    v[s_idx] = sign
                all_vectors.append(v)
    
    if not all_vectors:
        return np.array([], dtype=np.int8).reshape(0, D)
    
    final_vectors = np.array(all_vectors, dtype=np.int8)
    return final_vectors
