import time
import random
import sys

# EVOLVE-BLOCK-START
P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin for 90s limit
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency bitmasks
    adj_mask = [0] * n
    edge_set = set()
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        edge_set.add((min(u, v), max(u, v)))
    
    all_edges = list(edge_set)
    m = len(all_edges)
    
    # Helper: get list of vertices from mask
    def mask_to_list(mask):
        return [i for i in range(n) if (mask >> i) & 1]
    
    # Helper: compute cross edges of a biclique as a frozenset
    def cross_edges(L_mask, R_mask):
        es = set()
        lm = L_mask
        while lm:
            low = lm & (-lm)
            u = low.bit_length() - 1
            rm = R_mask
            while rm:
                low_r = rm & (-rm)
                v = low_r.bit_length() - 1
                es.add((min(u, v), max(u, v)))
                rm ^= low_r
            lm ^= low
        return es
    
    # Check if (L_mask, R_mask) is a valid biclique: all cross pairs are graph edges
    def is_valid_biclique(L_mask, R_mask):
        if L_mask == 0 or R_mask == 0:
            return False
        lm = L_mask
        while lm:
            low = lm & (-lm)
            u = low.bit_length() - 1
            if (adj_mask[u] & R_mask) != R_mask:
                return False
            lm ^= low
        return True
    
    # Expand a biclique to be maximal w.r.t. remaining edges
    # A vertex can be added to L if it's connected to all of R and all those edges are in remaining
    def expand_biclique(u, v, remaining):
        L_mask = 1 << u
        R_mask = 1 << v
        
        improved = True
        while improved:
            improved = False
            
            # Try adding to L
            for x in range(n):
                if (L_mask >> x) & 1 or (R_mask >> x) & 1:
                    continue
                
                # Check if x can join L: x must be connected to all r in R, and all (x,r) in remaining
                ok = True
                rm = R_mask
                while rm:
                    low_r = rm & (-rm)
                    r = low_r.bit_length() - 1
                    if not ((adj_mask[x] >> r) & 1):
                        ok = False
                        break
                    e = (min(x, r), max(x, r))
                    if e not in remaining:
                        ok = False
                        break
                    rm ^= low_r
                
                if ok:
                    L_mask |= (1 << x)
                    improved = True
            
            # Try adding to R
            for y in range(n):
                if (L_mask >> y) & 1 or (R_mask >> y) & 1:
                    continue
                
                # Check if y can join R: y must be connected to all l in L, and all (l,y) in remaining
                ok = True
                lm = L_mask
                while lm:
                    low_l = lm & (-lm)
                    l = low_l.bit_length() - 1
                    if not ((adj_mask[y] >> l) & 1):
                        ok = False
                        break
                    e = (min(l, y), max(l, y))
                    if e not in remaining:
                        ok = False
                        break
                    lm ^= low_l
                
                if ok:
                    R_mask |= (1 << y)
                    improved = True
                    
        return L_mask, R_mask
    
    # Greedy set-cover: repeatedly pick the biclique that covers the most uncovered edges
    def greedy_set_cover():
        remaining = set(all_edges)
        bicliques = []
        
        while remaining:
            if time.time() > deadline - 2:
                break
                
            best_L = 0
            best_R = 0
            best_count = 0
            
            # Try expanding from several seed edges
            # Use deterministic ordering for reproducibility, but try multiple seeds
            seed_edges = list(remaining)
            # Sort by degree sum for better heuristic
            seed_edges.sort(key=lambda e: -((adj_mask[e[0]] & adj_mask[e[1]]).bit_count()))
            
            # Try up to 30 seed edges
            for e in seed_edges[:30]:
                L_mask, R_mask = expand_biclique(e[0], e[1], remaining)
                ce = cross_edges(L_mask, R_mask)
                # Count how many of these are in remaining
                cnt = sum(1 for x in ce if x in remaining)
                if cnt > best_count:
                    best_count = cnt
                    best_L = L_mask
                    best_R = R_mask
            
            if best_L == 0:
                # Fallback
                e = next(iter(remaining))
                best_L = 1 << e[0]
                best_R = 1 << e[1]
                best_count = 1
                
            L_list = mask_to_list(best_L)
            R_list = mask_to_list(best_R)
            bicliques.append([L_list, R_list])
            
            # Remove covered edges
            ce = cross_edges(best_L, best_R)
            for e in ce:
                remaining.discard(e)
                    
        return bicliques
    
    # Local search: try to replace one biclique with a better decomposition
    def local_search(bicliques):
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            nb = len(bicliques)
            for i in range(nb):
                if time.time() > deadline - 3:
                    break
                # Try to split biclique i into two smaller bicliques
                Li, Ri = bicliques[i]
                L_set = set(Li)
                R_set = set(Ri)
                
                # Try splitting L: pick a subset of L and see if the rest forms a valid biclique
                # This is expensive, so try a few random splits
                for attempt in range(5):
                    if time.time() > deadline - 3:
                        break
                    # Randomly split L
                    if len(Li) < 2:
                        continue
                    random.shuffle(Li)
                    split_idx = random.randint(1, len(Li) - 1)
                    L1 = Li[:split_idx]
                    L2 = Li[split_idx:]
                    
                    # Check if (L1, R) and (L2, R) are both valid bicliques
                    # They are valid if all cross pairs are graph edges
                    # Since (L1 ∪ L2, R) was valid, (L1, R) and (L2, R) are automatically valid
                    # But we need to check that the union of their cross edges equals the original
                    # which it does by construction.
                    
                    # However, we need to check that no edge is double-covered or missed.
                    # The cross edges of (L1,R) and (L2,R) partition the cross edges of (L,R)
                    # if L1 and L2 are disjoint and their union is L.
                    
                    # This split always works structurally, but might not help.
                    # We need a different approach: try to find a biclique that covers
                    # edges from multiple existing bicliques.
                    
                    pass
                
                # Alternative: try to merge two bicliques
                for j in range(i + 1, nb):
                    if time.time() > deadline - 3:
                        break
                    Lj, Rj = bicliques[j]
                    
                    # Try all 4 ways to combine: (Li ∪ Lj, Ri ∩ Rj), etc.
                    # But this is complex. Let's try a simpler merge:
                    # Check if (Li, Rj) ∪ (Lj, Ri) covers the same edges as (Li,Ri) ∪ (Lj,Rj)
                    # No, that's not right.
                    
                    # Simpler: just check if the union of vertices forms a biclique
                    # with some partition.
                    
                    # For now, skip complex merges and just do the greedy restart
                    pass
        
        return bicliques
    
    # Better local search: try to find a biclique that covers edges from multiple existing bicliques
    def advanced_local_search(bicliques):
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            nb = len(bicliques)
            
            # Compute all edges covered by each biclique
            biclique_edges = []
            for Li, Ri in bicliques:
                ce = set()
                for u in Li:
                    for v in Ri:
                        ce.add((min(u, v), max(u, v)))
                biclique_edges.append(ce)
            
            all_covered = set()
            for ce in biclique_edges:
                all_covered |= ce
            
            # Try to find a new biclique that covers edges from at least 2 different bicliques
            # and allows us to remove at least 2 old bicliques
            for i in range(nb):
                if time.time() > deadline - 3:
                    break
                for j in range(i + 1, nb):
                    if time.time() > deadline - 3:
                        break
                    # Get the union of edges from bicliques i and j
                    union_edges = biclique_edges[i] | biclique_edges[j]
                    
                    # Try to find a biclique that covers all of union_edges
                    # Get all vertices involved
                    vertices = set()
                    for u, v in union_edges:
                        vertices.add(u)
                        vertices.add(v)
                    
                    # Try all bipartitions of vertices
                    # This is exponential, so only do it for small vertex sets
                    if len(vertices) > 10:
                        continue
                    
                    vertex_list = list(vertices)
                    # Try each vertex as a potential L-side starter
                    for start in vertex_list:
                        # Try to build a biclique containing start
                        # Use a greedy expansion
                        L_mask = 1 << start
                        R_mask = 0
                        
                        # Find all neighbors of start that are in vertices
                        candidates = [v for v in vertex_list if v != start and (adj_mask[start] >> v) & 1]
                        
                        # Try different R sets
                        # For small sets, try all subsets of candidates
                        if len(candidates) <= 8:
                            from itertools import combinations
                            for r_size in range(1, len(candidates) + 1):
                                for r_combo in combinations(candidates, r_size):
                                    R_mask = 0
                                    for r in r_combo:
                                        R_mask |= (1 << r)
                                    
                                    # Check if (L_mask, R_mask) is valid and covers all union_edges
                                    if not is_valid_biclique(L_mask, R_mask):
                                        continue
                                    
                                    ce = cross_edges(L_mask, R_mask)
                                    if union_edges.issubset(ce):
                                        # Check if ce is subset of all_covered (no new edges)
                                        if ce.issubset(all_covered):
                                            # This biclique covers edges from i and j
                                            # Check if we can remove i and j
                                            remaining_after = set()
                                            for k in range(nb):
                                                if k != i and k != j:
                                                    remaining_after |= biclique_edges[k]
                                            
                                            # Check that remaining_after ∪ ce == all_covered
                                            # and ce ∩ remaining_after == empty (no overlap)
                                            if (remaining_after | ce) == all_covered and (remaining_after & ce) == set():
                                                # Valid replacement
                                                new_bicliques = []
                                                for k in range(nb):
                                                    if k != i and k != j:
                                                        new_bicliques.append(bicliques[k])
                                                L_list = mask_to_list(L_mask)
                                                R_list = mask_to_list(R_mask)
                                                new_bicliques.append([L_list, R_list])
                                                bicliques = new_bicliques
                                                improved = True
                                                break
                                if improved:
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            
        return bicliques
    
    best_bicliques = None
    best_count = float('inf')
    
    # Multiple restarts with different random seeds
    for restart in range(50):
        if time.time() > deadline - 5:
            break
        
        random.seed(restart * 12345 + 67890)
        bicliques = greedy_set_cover()
        
        if not bicliques:
            continue
        
        bicliques = advanced_local_search(bicliques)
        
        if len(bicliques) < best_count:
            best_count = len(bicliques)
            best_bicliques = bicliques
            
            if best_count == 1:
                break

    if best_bicliques is None:
        # Fallback: single edges
        best_bicliques = [[[u], [v]] for u, v in all_edges]
        
    # Final conversion to ensure types are correct
    result = []
    for left, right in best_bicliques:
        result.append([list(left), list(right)])
        
    return result

# EVOLVE-BLOCK-END
