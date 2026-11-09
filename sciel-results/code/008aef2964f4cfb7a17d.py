import time
import sys
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    n = P['n']
    edges = P['edges']
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin before 90s kill
    
    # Build original adjacency as bitmasks
    adj_orig = [0] * n
    for u, v in edges:
        adj_orig[u] |= (1 << v)
        adj_orig[v] |= (1 << u)
    
    # Edge set as frozenset of frozensets for quick lookup
    edge_set = set()
    for u, v in edges:
        edge_set.add(frozenset((u, v)))
    
    total_edges = len(edges)
    
    def is_biclique(L_mask, R_mask):
        """Check if all cross pairs between L and R are edges in original graph."""
        if L_mask == 0 or R_mask == 0:
            return False
        m = L_mask
        while m:
            lsb = m & -m
            u = lsb.bit_length() - 1
            if (adj_orig[u] & R_mask) != R_mask:
                return False
            m ^= lsb
        return True
    
    def biclique_edges(L_mask, R_mask):
        """Return set of edges covered by biclique [L, R]."""
        es = set()
        m = L_mask
        while m:
            lsb = m & -m
            u = lsb.bit_length() - 1
            rm = R_mask
            while rm:
                lsb2 = rm & -rm
                v = lsb2.bit_length() - 1
                es.add(frozenset((u, v)))
                rm ^= lsb2
            m ^= lsb
        return es
    
    def greedy_biclique(remaining_edges, adj_rem, seed_rng=None):
        """Find a large biclique from remaining edges using greedy expansion."""
        if not remaining_edges:
            return None
        
        # Pick a random edge to start
        if seed_rng is not None:
            edge = seed_rng.choice(list(remaining_edges))
        else:
            edge = random.choice(list(remaining_edges))
        
        u, v = tuple(edge)
        L_mask = (1 << u)
        R_mask = (1 << v)
        
        # Compute common neighbors for expansion
        # A vertex w can join L if w is connected to all of R
        # A vertex w can join R if w is connected to all of L
        
        # Find all vertices that are "candidates" for L (connected to all in R)
        C_L = 0
        for w in range(n):
            if not (L_mask & (1 << w)) and not (R_mask & (1 << w)):
                if (adj_orig[w] & R_mask) == R_mask:
                    C_L |= (1 << w)
        
        # Find all vertices that are "candidates" for R (connected to all in L)
        C_R = 0
        for w in range(n):
            if not (L_mask & (1 << w)) and not (R_mask & (1 << w)):
                if (adj_orig[w] & L_mask) == L_mask:
                    C_R |= (1 << w)
        
        # Greedily expand: at each step, add the vertex that maximizes new edges covered
        while True:
            if time.time() > deadline:
                break
            
            best_gain = 0
            best_w = -1
            best_side = -1  # 0 = L, 1 = R
            
            # Try adding to L
            m = C_L
            while m:
                lsb = m & -m
                w = lsb.bit_length() - 1
                # Gain = number of edges from w to current R that are in remaining
                gain = 0
                rm = R_mask
                while rm:
                    lsb2 = rm & -rm
                    v = lsb2.bit_length() - 1
                    if frozenset((w, v)) in remaining_edges:
                        gain += 1
                    rm ^= lsb2
                if gain > best_gain:
                    best_gain = gain
                    best_w = w
                    best_side = 0
                m ^= lsb
            
            # Try adding to R
            m = C_R
            while m:
                lsb = m & -m
                w = lsb.bit_length() - 1
                gain = 0
                lm = L_mask
                while lm:
                    lsb2 = lm & -lm
                    u2 = lsb2.bit_length() - 1
                    if frozenset((u2, w)) in remaining_edges:
                        gain += 1
                    lm ^= lsb2
                if gain > best_gain:
                    best_gain = gain
                    best_w = w
                    best_side = 1
                m ^= lsb
            
            if best_w == -1 or best_gain == 0:
                break
            
            if best_side == 0:
                L_mask |= (1 << best_w)
                # Update C_R: w must be connected to all in L
                C_R &= adj_orig[best_w]
                C_R &= ~(1 << best_w)
                C_L &= ~(1 << best_w)
            else:
                R_mask |= (1 << best_w)
                C_L &= adj_orig[best_w]
                C_L &= ~(1 << best_w)
                C_R &= ~(1 << best_w)
        
        L_list = [i for i in range(n) if (L_mask & (1 << i))]
        R_list = [i for i in range(n) if (R_mask & (1 << i))]
        
        if not L_list or not R_list:
            return None
        
        return [L_list, R_list]
    
    def greedy_partition(rng):
        """Run full greedy extraction with randomization."""
        remaining = set(edge_set)
        adj_rem = [0] * n
        for u, v in edges:
            adj_rem[u] |= (1 << v)
            adj_rem[v] |= (1 << u)
        
        bicliques = []
        
        while remaining and time.time() < deadline:
            # Try multiple random seeds for starting edge, pick best
            best_bic = None
            best_cover = 0
            
            num_tries = min(5, len(remaining))
            for _ in range(num_tries):
                bic = greedy_biclique(remaining, adj_rem, rng)
                if bic is None:
                    continue
                L_list, R_list = bic
                # Count how many remaining edges this biclique covers
                cover = 0
                for u in L_list:
                    for v in R_list:
                        if frozenset((u, v)) in remaining:
                            cover += 1
                if cover > best_cover:
                    best_cover = cover
                    best_bic = bic
            
            if best_bic is None:
                # Fallback: take a single edge
                edge = rng.choice(list(remaining))
                u, v = tuple(edge)
                best_bic = [[u], [v]]
            
            L_list, R_list = best_bic
            bicliques.append(best_bic)
            
            # Remove covered edges
            for u in L_list:
                for v in R_list:
                    e = frozenset((u, v))
                    if e in remaining:
                        remaining.discard(e)
                        adj_rem[u] &= ~(1 << v)
                        adj_rem[v] &= ~(1 << u)
        
        return bicliques
    
    def try_merge(bicliques):
        """Try to merge bicliques whose union is a valid biclique."""
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(len(bicliques)):
                if changed:
                    break
                for j in range(i + 1, len(bicliques)):
                    L1, R1 = bicliques[i]
                    L2, R2 = bicliques[j]
                    
                    # Check if union [L1∪L2, R1∪R2] is a valid biclique
                    # Need: all L1∪L2 connected to all R1∪R2
                    L_new = set(L1) | set(L2)
                    R_new = set(R1) | set(R2)
                    
                    # Also need: the cross edges in the union must all be original edges
                    valid = True
                    for u in L_new:
                        for v in R_new:
                            if frozenset((u, v)) not in edge_set:
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid:
                        bicliques.pop(j)
                        bicliques[i] = [sorted(L_new), sorted(R_new)]
                        changed = True
                        break
        
        return bicliques
    
    def try_rebuild(bicliques):
        """Try to reduce biclique count by moving vertices between bicliques."""
        if time.time() > deadline:
            return bicliques
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(bicliques)):
                for j in range(len(bicliques)):
                    if i == j:
                        continue
                    L_i, R_i = bicliques[i]
                    L_j, R_j = bicliques[j]
                    
                    # Try moving a vertex from L_i to L_j
                    for u in L_i:
                        L_i_new = [x for x in L_i if x != u]
                        L_j_new = L_j + [u]
                        if not L_i_new:
                            continue
                        # Check if [L_j_new, R_j] is still valid
                        valid = True
                        for a in L_j_new:
                            for b in R_j:
                                if frozenset((a, b)) not in edge_set:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if valid:
                            # Check if [L_i_new, R_i] is still valid
                            valid2 = True
                            for a in L_i_new:
                                for b in R_i:
                                    if frozenset((a, b)) not in edge_set:
                                        valid2 = False
                                        break
                                if not valid2:
                                    break
                            if valid2:
                                bicliques[i] = [L_i_new, R_i]
                                bicliques[j] = [L_j_new, R_j]
                                improved = True
                                break
                    if improved:
                        break
                    # Try moving a vertex from R_i to R_j
                    for v in R_i:
                        R_i_new = [x for x in R_i if x != v]
                        R_j_new = R_j + [v]
                        if not R_i_new:
                            continue
                        valid = True
                        for a in L_j:
                            for b in R_j_new:
                                if frozenset((a, b)) not in edge_set:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if valid:
                            valid2 = True
                            for a in L_i:
                                for b in R_i_new:
                                    if frozenset((a, b)) not in edge_set:
                                        valid2 = False
                                        break
                                if not valid2:
                                    break
                            if valid2:
                                bicliques[i] = [L_i, R_i_new]
                                bicliques[j] = [L_j, R_j_new]
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
        
        return bicliques
    
    def validate(bicliques):
        """Check that bicliques form a valid partition."""
        covered = set()
        for L, R in bicliques:
            if not L or not R:
                return False
            for u in L:
                for v in R:
                    e = frozenset((u, v))
                    if e not in edge_set:
                        return False
                    if e in covered:
                        return False
                    covered.add(e)
        return covered == edge_set
    
    # Multi-restart strategy
    best_count = len(edges)  # Worst case: one biclique per edge
    best_result = None
    
    rng = random.Random(42)
    
    num_restarts = 20
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        # Vary the random seed for diversity
        rng2 = random.Random(r * 1337 + 7)
        
        bicliques = greedy_partition(rng2)
        
        if not bicliques:
            continue
        
        # Post-processing: merge
        bicliques = try_merge(bicliques)
        
        # Post-processing: rebuild (move vertices)
        bicliques = try_rebuild(bicliques)
        
        # Validate
        if validate(bicliques):
            if len(bicliques) < best_count:
                best_count = len(bicliques)
                best_result = bicliques
        
        # If we've already achieved a very good result, we can be more aggressive
        if best_count <= 8:
            break
    
    # If no valid result found, fall back to trivial (one biclique per edge)
    if best_result is None:
        best_result = []
        for u, v in edges:
            best_result.append([[u], [v]])
    
    # Final conversion to ensure proper types
    final_result = []
    for bic in best_result:
        l = [int(x) for x in bic[0]]
        r = [int(x) for x in bic[1]]
        final_result.append([l, r])
    
    return final_result
