import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    deadline = time.time() + 55
    n = P['n']
    edges = P['edges']
    
    # Build bitmask adjacency
    adj_mask = [0] * n
    edge_set = set()
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        edge_set.add((min(u,v), max(u,v)))
    
    all_edges = list(edge_set)
    total_edges = len(all_edges)
    
    # Helper: check if a set of edges forms a biclique
    def edges_to_biclique(edge_subset):
        if not edge_subset:
            return None
        vertices = set()
        for u, v in edge_subset:
            vertices.add(u)
            vertices.add(v)
        
        color = {}
        for v in vertices:
            if v not in color:
                stack = [v]
                color[v] = 0
                while stack:
                    curr = stack.pop()
                    for nb in vertices:
                        if nb != curr and (min(curr, nb), max(curr, nb)) in edge_subset:
                            if nb not in color:
                                color[nb] = 1 - color[curr]
                                stack.append(nb)
                            elif color[nb] == color[curr]:
                                return None
        
        L = [v for v in vertices if color[v] == 0]
        R = [v for v in vertices if color[v] == 1]
        
        for u in L:
            for v in R:
                if (min(u,v), max(u,v)) not in edge_subset:
                    return None
        
        return [L, R]
    
    # Greedy with bitmask-based maximal biclique expansion
    def greedy_init():
        remaining = set(all_edges)
        bicliques = []
        
        while remaining:
            best_bic = None
            best_count = 0
            
            sample_edges = list(remaining)
            random.shuffle(sample_edges)
            
            for e in sample_edges[:25]:
                u, v = e
                L_mask = (1 << u)
                R_mask = (1 << v)
                
                changed = True
                while changed:
                    changed = False
                    # Expand L: find vertices adjacent to all in R
                    common_R = (1 << n) - 1
                    m = R_mask
                    while m:
                        bit = m & (-m)
                        r = bit.bit_length() - 1
                        common_R &= adj_mask[r]
                        m &= ~bit
                    candidates_L = common_R & ~(L_mask | R_mask)
                    while candidates_L:
                        bit = candidates_L & (-candidates_L)
                        L_mask |= bit
                        candidates_L &= ~bit
                        changed = True
                    
                    # Expand R: find vertices adjacent to all in L
                    common_L = (1 << n) - 1
                    m = L_mask
                    while m:
                        bit = m & (-m)
                        l = bit.bit_length() - 1
                        common_L &= adj_mask[l]
                        m &= ~bit
                    candidates_R = common_L & ~(L_mask | R_mask)
                    while candidates_R:
                        bit = candidates_R & (-candidates_R)
                        R_mask |= bit
                        candidates_R &= ~bit
                        changed = True
                
                # Count edges covered
                count = 0
                m1 = L_mask
                while m1:
                    bit1 = m1 & (-m1)
                    l = bit1.bit_length() - 1
                    m2 = R_mask
                    while m2:
                        bit2 = m2 & (-m2)
                        r = bit2.bit_length() - 1
                        if (min(l,r), max(l,r)) in remaining:
                            count += 1
                        m2 &= ~bit2
                    m1 &= ~bit1
                
                if count > best_count:
                    best_count = count
                    best_L = []
                    m1 = L_mask
                    while m1:
                        bit = m1 & (-m1)
                        best_L.append(bit.bit_length() - 1)
                        m1 &= ~bit
                    best_R = []
                    m2 = R_mask
                    while m2:
                        bit = m2 & (-m2)
                        best_R.append(bit.bit_length() - 1)
                        m2 &= ~bit
                    best_bic = [best_L, best_R]
            
            if best_bic is None:
                u, v = next(iter(remaining))
                best_bic = [[u], [v]]
            
            bicliques.append(best_bic)
            covered = set()
            L, R = best_bic
            for l in L:
                for r in R:
                    covered.add((min(l,r), max(l,r)))
            remaining -= covered
        
        return bicliques
    
    # Try to merge two bicliques
    def try_merge(bicliques):
        n_bic = len(bicliques)
        for i in range(n_bic):
            for j in range(i+1, n_bic):
                L1, R1 = bicliques[i]
                L2, R2 = bicliques[j]
                
                edges_i = set()
                for u in L1:
                    for v in R1:
                        edges_i.add((min(u,v), max(u,v)))
                
                edges_j = set()
                for u in L2:
                    for v in R2:
                        edges_j.add((min(u,v), max(u,v)))
                
                union_edges = edges_i | edges_j
                merged = edges_to_biclique(union_edges)
                if merged:
                    new_bicliques = [b for k, b in enumerate(bicliques) if k != i and k != j]
                    new_bicliques.append(merged)
                    return new_bicliques
        return None
    
    best_bicliques = None
    best_count = float('inf')
    
    for restart in range(50):
        if time.time() > deadline - 8:
            break
        
        bicliques = greedy_init()
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            new_bic = try_merge(bicliques)
            if new_bic is not None:
                bicliques = new_bic
                improved = True
        
        if len(bicliques) < best_count:
            best_count = len(bicliques)
            best_bicliques = bicliques
    
    if best_bicliques is None:
        best_bicliques = [[[u], [v]] for u, v in all_edges]
    
    result = []
    for left, right in best_bicliques:
        result.append([list(left), list(right)])
    
    return result

# EVOLVE-BLOCK-END
