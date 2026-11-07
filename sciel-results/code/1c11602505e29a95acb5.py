import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    n = P['n']
    edges = P['edges']
    
    # Build adjacency mask
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    # Seed Solution (Baseline)
    best_bicliques = []
    for u in range(n):
        right_side = [v for a, v in edges if a == u]
        if right_side:
            best_bicliques.append([[u], right_side])
            
    best_count = len(best_bicliques)

    def try_greedy(restarts=200):
        nonlocal best_bicliques, best_count
        
        for r in range(restarts):
            if time.time() > deadline:
                return
            
            rem_adj = adj[:]
            current_bicliques = []
            edges_left = len(edges)
            
            while edges_left > 0:
                if time.time() > deadline:
                    return
                
                found_edge = False
                for _ in range(50):
                    u = random.randint(0, n-1)
                    mask = rem_adj[u]
                    if mask:
                        m = mask
                        candidates = []
                        while m:
                            lsb = m & -m
                            v = lsb.bit_length() - 1
                            candidates.append(v)
                            m ^= lsb
                        v = random.choice(candidates)
                        
                        if rem_adj[u] & (1 << v):
                            found_edge = True
                            break
                
                if not found_edge:
                    break
                
                L = {u}
                R = {v}
                changed = True
                for _ in range(20):
                    if not changed:
                        break
                    changed = False
                    
                    candidates_L = [w for w in range(n) if w not in L]
                    random.shuffle(candidates_L)
                    
                    for w in candidates_L:
                        mask_R = 0
                        for rv in R:
                            mask_R |= (1 << rv)
                        if (rem_adj[w] & mask_R) == mask_R:
                            L.add(w)
                            changed = True
                            
                    candidates_R = [w for w in range(n) if w not in R]
                    random.shuffle(candidates_R)
                    
                    for w in candidates_R:
                        valid = True
                        for lu in L:
                            if not (rem_adj[lu] & (1 << w)):
                                valid = False
                                break
                        if valid:
                            R.add(w)
                            changed = True

                current_bicliques.append([list(L), list(R)])
                
                mask_L = 0
                for lu in L:
                    mask_L |= (1 << lu)
                mask_R = 0
                for rv in R:
                    mask_R |= (1 << rv)
                
                for lu in L:
                    rem_adj[lu] &= ~mask_R
                for rv in R:
                    rem_adj[rv] &= ~mask_L
                
                edges_left -= len(L) * len(R)
                
                if edges_left < 0:
                    break

            if len(current_bicliques) < best_count:
                best_count = len(current_bicliques)
                best_bicliques = current_bicliques
                if best_count == 1:
                    return

    def local_search_refine(bicliques, max_iters=50):
        """Try to reduce biclique count by splitting one biclique into two."""
        if time.time() > deadline:
            return bicliques
        
        bl = [list(b) for b in bicliques]
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() > deadline:
                return bl
            
            # Try each biclique
            for i in range(len(bl)):
                if time.time() > deadline:
                    return bl
                L, R = bl[i][0], bl[i][1]
                if len(L) < 2 and len(R) < 2:
                    continue
                
                # Try to split L into L1, L2 such that both (L1,R) and (L2,R) are valid
                # and the union covers all edges of (L,R)
                # Actually, we need to partition the edge set L×R into two bicliques.
                # Simplest: pick a subset L1 ⊂ L, then L2 = L \ L1.
                # (L1, R) and (L2, R) are both valid bicliques iff:
                #   - Every pair in L1×R is an edge (true, since L×R is a biclique)
                #   - Every pair in L2×R is an edge (true)
                # But we need the cross-edge sets to partition ALL pinned edges exactly once.
                # The edges in L1×R and L2×R are disjoint and their union is L×R. ✓
                # However, we also need to ensure that no edge appears in two different bicliques
                # in the overall partition. Since we're replacing one biclique with two, and the
                # two new bicliques' edge sets are disjoint subsets of the original, this is fine
                # as long as no other biclique shares edges with them.
                
                # The key question: can we merge two existing bicliques? Or split one to allow
                # a better global arrangement?
                # 
                # Alternative local search: try to merge two bicliques if possible.
                # Bicliques (L1,R1) and (L2,R2) can be merged if:
                #   - L1∩L2 = ∅ and R1∩R2 = ∅ (sides must be disjoint)
                #   - All cross pairs in (L1∪L2)×(R1∪R2) are edges
                #   - The edge sets partition correctly
                #
                # This is complex. Instead, try a simpler move:
                # Pick two bicliques, try to swap vertices between them to enable merging.
                
                # For now, try: can we merge biclique i with any other biclique j?
                for j in range(len(bl)):
                    if i == j:
                        continue
                    if time.time() > deadline:
                        return bl
                    
                    L1, R1 = bl[i][0], bl[i][1]
                    L2, R2 = bl[j][0], bl[j][1]
                    
                    # Check if sides are disjoint
                    if set(L1) & set(L2) or set(R1) & set(R2):
                        continue
                    # Also check cross-disjointness: L1∩R2, L2∩R1
                    if set(L1) & set(R2) or set(L2) & set(R1):
                        continue
                    
                    # Check if (L1∪L2) × (R1∪R2) is a valid biclique
                    # i.e., every pair (u,v) with u in L1∪L2, v in R1∪R2 is an edge
                    # Note: within-side edges are allowed, so we only check cross pairs.
                    valid = True
                    for u in L1:
                        for v in R1:
                            if not (adj[u] & (1 << v)):
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        for u in L1:
                            for v in R2:
                                if not (adj[u] & (1 << v)):
                                    valid = False
                                    break
                            if not valid:
                                break
                    if valid:
                        for u in L2:
                            for v in R1:
                                if not (adj[u] & (1 << v)):
                                    valid = False
                                    break
                            if not valid:
                                break
                    if valid:
                        for u in L2:
                            for v in R2:
                                if not (adj[u] & (1 << v)):
                                    valid = False
                                    break
                            if not valid:
                                break
                    
                    if valid:
                        # Merge: remove j, update i
                        new_L = L1 + L2
                        new_R = R1 + R2
                        bl[i] = [new_L, new_R]
                        del bl[j]
                        improved = True
                        break
                
                if improved:
                    break
        
        return bl

    try_greedy()

    # Apply local search refinement
    if best_bicliques:
        refined = local_search_refine(best_bicliques, max_iters=100)
        if len(refined) < best_count:
            best_count = len(refined)
            best_bicliques = refined

    # Fallback if something went wrong
    if not best_bicliques:
        for u in range(n):
            right_side = [v for a, v in edges if a == u]
            if right_side:
                best_bicliques.append([[u], right_side])

    # Convert to required format
    result = []
    for b in best_bicliques:
        left = [int(x) for x in b[0]]
        right = [int(x) for x in b[1]]
        result.append([left, right])
        
    return result
