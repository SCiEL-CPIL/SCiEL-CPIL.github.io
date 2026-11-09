"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    n = P['n']
    edges = P['edges']
    deadline = time.time() + 55.0  # safety margin before 60s limit
    
    # Build adjacency as bitmasks for fast operations
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    # Track uncovered edges as a set of tuples
    uncovered = set()
    for u, v in edges:
        uncovered.add((min(u,v), max(u,v)))
    
    if not uncovered:
        return []
    
    bicliques = []
    
    def mask_to_list(mask):
        return [i for i in range(n) if (mask >> i) & 1]

    def get_maximal_biclique(u, v, uncov_set):
        """
        Find a maximal biclique containing edge (u, v) using only edges in uncov_set.
        Returns (L_list, R_list).
        """
        # Start with u in L, v in R
        L_mask = 1 << u
        R_mask = 1 << v
        
        changed = True
        while changed:
            changed = False
            
            # Try to add vertices to L
            # A vertex w can join L if it is connected to ALL vertices in R
            # and (w, r) is in uncov_set for all r in R.
            # Also w must not be in R.
            for w in range(n):
                if (R_mask >> w) & 1:
                    continue
                if (L_mask >> w) & 1:
                    continue
                
                can_add = True
                # Check if w connects to all in R via uncovered edges
                r_tmp = R_mask
                while r_tmp:
                    lsb = r_tmp & -r_tmp
                    r = lsb.bit_length() - 1
                    a, b = (w, r) if w < r else (r, w)
                    if (a, b) not in uncov_set:
                        can_add = False
                        break
                    r_tmp -= lsb
                if can_add:
                    L_mask |= (1 << w)
                    changed = True
            
            # Try to add vertices to R
            for w in range(n):
                if (L_mask >> w) & 1:
                    continue
                if (R_mask >> w) & 1:
                    continue
                
                can_add = True
                l_tmp = L_mask
                while l_tmp:
                    lsb = l_tmp & -l_tmp
                    l = lsb.bit_length() - 1
                    a, b = (w, l) if w < l else (l, w)
                    if (a, b) not in uncov_set:
                        can_add = False
                        break
                    l_tmp -= lsb
                if can_add:
                    R_mask |= (1 << w)
                    changed = True
        
        L_list = mask_to_list(L_mask)
        R_list = mask_to_list(R_mask)
        
        # Ensure disjointness (should be by construction, but safe)
        L_set = set(L_list)
        R_list = [r for r in R_list if r not in L_set]
        
        return L_list, R_list

    while uncovered:
        if time.time() > deadline:
            # Fallback: cover remaining edges one by one
            for e in list(uncovered):
                u, v = e
                bicliques.append([[u], [v]])
            break
        
        best_edge = None
        best_coverage = -1
        
        # Heuristic: Pick the edge that allows covering the most edges in a single biclique
        # We estimate this by finding the maximal biclique for a sample of edges or all edges if small
        # Given N=30, E~200, trying all edges might be okay if expansion is fast.
        # But expansion is iterative. Let's just pick the edge with max degree in uncovered graph first,
        # or simply iterate. To be safe on time, we'll pick the first edge but try to find a good one.
        
        # Better heuristic: Pick edge (u,v) maximizing (deg_u * deg_v) in uncovered subgraph?
        # Or just pick the edge with the most common neighbors?
        
        # Let's stick to a robust greedy: 
        # 1. Compute degrees in uncovered graph
        deg = [0] * n
        for u, v in uncovered:
            deg[u] += 1
            deg[v] += 1
            
        # Pick edge with highest sum of degrees
        best_score = -1
        for u, v in uncovered:
            score = deg[u] + deg[v]
            if score > best_score:
                best_score = score
                best_edge = (u, v)
                
        u, v = best_edge
        
        L, R = get_maximal_biclique(u, v, uncovered)
        
        if not L or not R:
            bicliques.append([[u], [v]])
            uncovered.discard((min(u,v), max(u,v)))
            continue
        
        bicliques.append([list(L), list(R)])
        
        # Remove all edges in this biclique from uncovered
        for l in L:
            for r in R:
                a, b = (l, r) if l < r else (r, l)
                uncovered.discard((a, b))
    
    # Convert to proper types
    result = []
    for bic in bicliques:
        left = list(bic[0])
        right = list(bic[1])
        if left and right and not (set(left) & set(right)):
            result.append([left, right])
    
    return result
# EVOLVE-BLOCK-END
