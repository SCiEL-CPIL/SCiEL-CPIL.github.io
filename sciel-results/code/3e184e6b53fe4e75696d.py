import time
import random
from itertools import combinations

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 80.0

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
    total_edges = len(edges)

    # Precompute popcount for 16-bit chunks for speed
    popcount = [0] * 65536
    for i in range(65536):
        popcount[i] = bin(i).count('1')
    
    def popcount_int(x):
        c = 0
        while x:
            c += popcount[x & 0xFFFF]
            x >>= 16
        return c

    def count_edges(rem_adj):
        s = 0
        for i in range(n):
            s += popcount_int(rem_adj[i])
        return s // 2

    def try_greedy(restarts=500):
        nonlocal best_bicliques, best_count
        
        for r in range(restarts):
            if time.time() > deadline:
                return
            
            rem_adj = adj[:]
            current_bicliques = []
            edges_left = total_edges
            
            while edges_left > 0:
                if time.time() > deadline:
                    return
                
                # Pick a random edge to start the biclique
                found_edge = False
                u = -1
                v = -1
                
                for _ in range(100):
                    u = random.randint(0, n-1)
                    mask = rem_adj[u]
                    if mask:
                        m = mask
                        candidates = []
                        while m:
                            lsb = m & -m
                            v_cand = lsb.bit_length() - 1
                            candidates.append(v_cand)
                            m ^= lsb
                        v = random.choice(candidates)
                        if rem_adj[u] & (1 << v):
                            found_edge = True
                            break
                
                if not found_edge:
                    for i in range(n):
                        if rem_adj[i]:
                            u = i
                            m = rem_adj[i]
                            lsb = m & -m
                            v = lsb.bit_length() - 1
                            found_edge = True
                            break
                
                if not found_edge:
                    break

                # Expand biclique from (u, v)
                L = {u}
                R = {v}
                
                # Greedy expansion
                changed = True
                for _ in range(n):
                    if not changed:
                        break
                    changed = False
                    
                    # Expand L
                    mask_R = 0
                    for rv in R:
                        mask_R |= (1 << rv)
                    
                    candidates_L = []
                    for w in range(n):
                        if w in L:
                            continue
                        if (rem_adj[w] & mask_R) == mask_R:
                            candidates_L.append(w)
                    
                    if candidates_L:
                        random.shuffle(candidates_L)
                        for w in candidates_L:
                            L.add(w)
                            changed = True

                    # Expand R
                    mask_L = 0
                    for lu in L:
                        mask_L |= (1 << lu)
                    
                    candidates_R = []
                    for w in range(n):
                        if w in R:
                            continue
                        valid = True
                        for lu in L:
                            if not (rem_adj[lu] & (1 << w)):
                                valid = False
                                break
                        if valid:
                            candidates_R.append(w)
                            
                    if candidates_R:
                        random.shuffle(candidates_R)
                        for w in candidates_R:
                            R.add(w)
                            changed = True

                # Commit this biclique
                current_bicliques.append([list(L), list(R)])
                
                # Remove edges covered by this biclique
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
                best_bicliques = [ [list(l), list(r)] for l, r in current_bicliques ]
                if best_count == 1:
                    return

    def can_merge(bi, bj):
        """Check if biclique i and j can be merged into one biclique."""
        Li, Ri = bi
        Lj, Rj = bj
        
        # Check Li vs Rj
        for u in Li:
            for v in Rj:
                if not (adj[u] & (1 << v)):
                    return False
        
        # Check Lj vs Ri
        for u in Lj:
            for v in Ri:
                if not (adj[u] & (1 << v)):
                    return False
        
        return True

    def local_search(bicliques, max_iters=200):
        """Try to merge pairs of bicliques to reduce count."""
        if time.time() > deadline:
            return bicliques
        
        current = [[list(b[0]), list(b[1])] for b in bicliques]
        improved = True
        
        while improved and time.time() < deadline:
            improved = False
            n_bic = len(current)
            
            for i in range(n_bic):
                if time.time() > deadline:
                    break
                for j in range(i+1, n_bic):
                    if can_merge(current[i], current[j]):
                        # Merge j into i
                        Li = set(current[i][0]) | set(current[j][0])
                        Ri = set(current[i][1]) | set(current[j][1])
                        current[i] = [list(Li), list(Ri)]
                        current.pop(j)
                        improved = True
                        break
                if improved:
                    break
        
        return current

    # --- New: Branch and Bound for Minimum Biclique Cover ---
    
    def get_maximal_bicliques():
        """Enumerate all maximal bicliques in the graph."""
        bicliques = set()
        
        # For each pair of vertices (u, v) that form an edge, try to expand to maximal biclique
        for u in range(n):
            for v in range(u+1, n):
                if adj[u] & (1 << v):
                    # Start with L={u}, R={v}
                    L = {u}
                    R = {v}
                    changed = True
                    while changed:
                        changed = False
                        # Expand L: find w not in L such that w is connected to all in R
                        for w in range(n):
                            if w not in L:
                                connected_to_all_R = True
                                for rv in R:
                                    if not (adj[w] & (1 << rv)):
                                        connected_to_all_R = False
                                        break
                                if connected_to_all_R:
                                    L.add(w)
                                    changed = True
                        # Expand R: find w not in R such that w is connected to all in L
                        for w in range(n):
                            if w not in R:
                                connected_to_all_L = True
                                for lu in L:
                                    if not (adj[lu] & (1 << w)):
                                        connected_to_all_L = False
                                        break
                                if connected_to_all_L:
                                    R.add(w)
                                    changed = True
                    
                    # Only add if it's a proper biclique (both sides non-empty)
                    if L and R:
                        # Normalize: sort both sides, and ensure L[0] < R[0] or similar canonical form
                        L_sorted = tuple(sorted(L))
                        R_sorted = tuple(sorted(R))
                        # To avoid duplicates from different starting points, we can use a canonical key
                        # Since (L,R) and (R,L) are different bicliques in our formulation, we keep them distinct
                        # But we might generate the same (L,R) from different starting edges.
                        bicliques.add((L_sorted, R_sorted))
        
        # Also consider single vertex bicliques? No, they cover 0 edges.
        # What about bicliques where one side has 1 vertex?
        # The loop above covers all edges, so any maximal biclique with >=1 edge will be found.
        
        return list(bicliques)

    def solve_bnb(max_bicliques, time_limit=30.0):
        """Solve minimum set cover using branch and bound."""
        # Each biclique is a set of edges it covers
        # We need to cover all edges in P['edges']
        
        # Map each edge to a bit index
        edge_index = {}
        for i, (u, v) in enumerate(edges):
            if u > v:
                u, v = v, u
            edge_index[(u, v)] = i
        
        num_edges = len(edges)
        all_edges_mask = (1 << num_edges) - 1
        
        # For each biclique, compute the mask of edges it covers
        biclique_masks = []
        for L, R in max_bicliques:
            mask = 0
            for u in L:
                for v in R:
                    if u > v:
                        u, v = v, u
                    if (u, v) in edge_index:
                        mask |= (1 << edge_index[(u, v)])
            if mask > 0:
                biclique_masks.append(mask)
        
        # Remove dominated bicliques
        # If biclique A's mask is a subset of biclique B's mask, A is dominated
        # But we want to keep the superset.
        # Actually, for set cover, we want to pick sets. If A subset B, picking B is always at least as good as picking A.
        # So we can remove A.
        nondominated = []
        for i in range(len(biclique_masks)):
            dominated = False
            for j in range(len(biclique_masks)):
                if i == j:
                    continue
                if biclique_masks[i] & ~biclique_masks[j] == 0:
                    # i is subset of j
                    if biclique_masks[i] == biclique_masks[j]:
                        if i < j: # keep one
                            dominated = True
                            break
                    else:
                        dominated = True
                        break
            if not dominated:
                nondominated.append(biclique_masks[i])
        
        biclique_masks = nondominated
        num_bics = len(biclique_masks)
        
        if num_bics == 0:
            return []
        
        best_solution = None
        best_count = float('inf')
        
        # Precompute for each edge, which bicliques cover it
        edge_to_bics = [[] for _ in range(num_edges)]
        for i, mask in enumerate(biclique_masks):
            m = mask
            while m:
                lsb = m & -m
                edge_idx = lsb.bit_length() - 1
                edge_to_bics[edge_idx].append(i)
                m ^= lsb
        
        # Sort bicliques by size (larger first) for better pruning?
        # Or sort by "importance"
        
        # Branch and Bound
        # State: covered_mask, current_selection, current_count
        # Lower bound: number of uncovered edges / max_biclique_size_remaining?
        # Or simpler: greedy lower bound on remaining edges.
        
        def greedy_lb(remaining_mask, available_bics):
            """Lower bound: greedy set cover on remaining mask."""
            rem = remaining_mask
            count = 0
            while rem:
                best_cover = 0
                best_bic = -1
                for b in available_bics:
                    cover = biclique_masks[b] & rem
                    if cover > best_cover:
                        best_cover = cover
                        best_bic = b
                if best_bic == -1:
                    return float('inf') # Should not happen if solution exists
                count += 1
                rem ^= best_cover
            return count
        
        # We need to be careful with the "available_bics" in LB.
        # A simple LB is: max(1, ceil(popcount(rem) / max_bic_size))
        # Let's use a stronger one if time permits.
        
        max_bic_size = max(popcount_int(m) for m in biclique_masks) if biclique_masks else 1
        
        def bb(covered_mask, selected, count, start_idx):
            nonlocal best_solution, best_count
            
            if time.time() > time_limit:
                return
            
            if covered_mask == all_edges_mask:
                if count < best_count:
                    best_count = count
                    best_solution = selected[:]
                return
            
            # Lower bound
            remaining = all_edges_mask ^ covered_mask
            rem_count = popcount_int(remaining)
            lb = (rem_count + max_bic_size - 1) // max_bic_size
            
            if count + lb >= best_count:
                return
            
            # Choose the next biclique to branch on
            # Heuristic: pick the biclique that covers the most uncovered edges
            # Or pick an uncovered edge, and branch on the bicliques that cover it
            
            # Find an uncovered edge
            m = remaining
            if m == 0:
                return
            lsb = m & -m
            edge_idx = lsb.bit_length() - 1
            
            # Branch on bicliques that cover this edge
            candidates = edge_to_bics[edge_idx]
            
            # Sort candidates by number of new edges covered
            candidates.sort(key=lambda b: popcount_int(biclique_masks[b] & remaining), reverse=True)
            
            for b in candidates:
                if time.time() > time_limit:
                    return
                new_covered = covered_mask | biclique_masks[b]
                selected.append(b)
                bb(new_covered, selected, count + 1, start_idx)
                selected.pop()
                
                if count + 1 >= best_count:
                    break # Prune
        
        bb(0, [], 0, 0)
        
        if best_solution is None:
            return []
            
        return [max_bicliques[i] for i in best_solution]

    # Run BnB
    if time.time() < deadline:
        try:
            max_bics = get_maximal_bicliques()
            # Limit number of bicliques for BnB to avoid explosion
            if len(max_bics) < 500:
                bnb_solution = solve_bnb(max_bics, time_limit=30.0)
                if bnb_solution:
                    # Verify and convert
                    # Check if it's a valid partition
                    covered_edges = set()
                    valid = True
                    for L, R in bnb_solution:
                        for u in L:
                            for v in R:
                                if u > v:
                                    u, v = v, u
                                if (u, v) not in set(map(lambda e: (min(e[0], e[1]), max(e[0], e[1])), edges)):
                                    valid = False
                                    break
                                if (u, v) in covered_edges:
                                    valid = False
                                    break
                                covered_edges.add((u, v))
                        if not valid:
                            break
                    
                    if valid and covered_edges == set(map(lambda e: (min(e[0], e[1]), max(e[0], e[1])), edges)):
                        if len(bnb_solution) < best_count:
                            best_count = len(bnb_solution)
                            best_bicliques = [[list(L), list(R)] for L, R in bnb_solution]
        except:
            pass

    try_greedy()
    
    # Apply local search to improve the best solution
    if len(best_bicliques) > 1 and time.time() < deadline:
        improved = local_search(best_bicliques)
        if len(improved) < best_count:
            best_bicliques = improved
            best_count = len(improved)

    # Fallback if something went wrong (shouldn't happen)
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
