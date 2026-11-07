import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    n = P['n']
    edges = P['edges']
    total_edges = len(edges)

    # Build adjacency mask
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    # Build edge set for quick lookup
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u, v), max(u, v)))

    # Build a list of edges for fast random access
    edge_list = list(edges)

    # Seed Solution (Baseline) - star decomposition
    best_bicliques = []
    for u in range(n):
        right_side = [v for a, v in edges if a == u]
        if right_side:
            best_bicliques.append([[u], right_side])
    best_count = len(best_bicliques)

    def get_edges_remaining(rem_adj):
        """Get list of remaining edges from rem_adj."""
        rem_edges = []
        for u in range(n):
            m = rem_adj[u]
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                if u < v:
                    rem_edges.append((u, v))
                m ^= lsb
        return rem_edges

    def expand_biclique(u, v, rem_adj, randomize=True):
        """Expand biclique starting from edge (u,v) in rem_adj."""
        L = {u}
        R = {v}
        
        # Use bitmasks for speed
        mask_L = (1 << u)
        mask_R = (1 << v)
        
        changed = True
        while changed:
            changed = False
            
            # Try to expand L: find w not in L such that w is adjacent to all in R
            # w must be adjacent to all vertices in R within rem_adj
            # For w to be added to L, rem_adj[w] must contain all bits in mask_R
            for w in range(n):
                if mask_L & (1 << w):
                    continue
                if (rem_adj[w] & mask_R) == mask_R:
                    # Check that w is not already in R
                    if mask_R & (1 << w):
                        continue
                    L.add(w)
                    mask_L |= (1 << w)
                    changed = True
            
            # Try to expand R: find w not in R such that w is adjacent to all in L
            for w in range(n):
                if mask_R & (1 << w):
                    continue
                if mask_L & (1 << w):
                    continue
                # Check that all vertices in L are adjacent to w
                valid = True
                m = mask_L
                while m:
                    lsb = m & -m
                    u2 = lsb.bit_length() - 1
                    if not (rem_adj[u2] & (1 << w)):
                        valid = False
                        break
                    m ^= lsb
                if valid:
                    R.add(w)
                    mask_R |= (1 << w)
                    changed = True
        
        return list(L), list(R)

    def greedy_partition(randomize_start=True, n_restarts=100):
        """Run greedy biclique partitioning."""
        results = []
        
        for r in range(n_restarts):
            if time.time() > deadline:
                break
            
            rem_adj = adj[:]
            bicliques = []
            edges_left = total_edges
            
            while edges_left > 0:
                if time.time() > deadline:
                    return None
                
                # Pick a starting edge
                if randomize_start:
                    # Random edge
                    found = False
                    for _ in range(20):
                        u = random.randint(0, n - 1)
                        mask = rem_adj[u]
                        if mask:
                            # Pick random v from mask
                            m = mask
                            candidates = []
                            while m:
                                lsb = m & -m
                                v_cand = lsb.bit_length() - 1
                                candidates.append(v_cand)
                                m ^= lsb
                            v = random.choice(candidates)
                            found = True
                            break
                    if not found:
                        # Fallback: find any edge
                        for i in range(n):
                            if rem_adj[i]:
                                u = i
                                m = rem_adj[i]
                                lsb = m & -m
                                v = lsb.bit_length() - 1
                                found = True
                                break
                        if not found:
                            break
                else:
                    # Deterministic: pick first available edge
                    found = False
                    for i in range(n):
                        if rem_adj[i]:
                            u = i
                            m = rem_adj[i]
                            lsb = m & -m
                            v = lsb.bit_length() - 1
                            found = True
                            break
                    if not found:
                        break
                
                # Expand biclique
                L, R = expand_biclique(u, v, rem_adj)
                
                if not L or not R:
                    # Shouldn't happen, but safety
                    bicliques.append([[u], [v]])
                    mask_u = 1 << u
                    mask_v = 1 << v
                    rem_adj[u] &= ~mask_v
                    rem_adj[v] &= ~mask_u
                    edges_left -= 1
                    continue
                
                bicliques.append([L, R])
                
                # Remove covered edges
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
                    edges_left = 0
            
            if edges_left == 0 and bicliques:
                results.append(bicliques)
        
        return results

    def local_search(bicliques, max_iters=200):
        """Try to merge bicliques to reduce count."""
        bicliques = [[list(b[0]), list(b[1])] for b in bicliques]
        
        def is_valid_biclique(L, R):
            """Check if L x R is a valid biclique in the original graph."""
            for u in L:
                for v in R:
                    if not (adj[u] & (1 << v)):
                        return False
            return True
        
        def get_covered_edges(bicliques):
            covered = set()
            for L, R in bicliques:
                for u in L:
                    for v in R:
                        covered.add((min(u, v), max(u, v)))
            return covered
        
        def try_merge_pair(i, j, bicliques):
            """Try to merge biclique i and j. Return new biclique or None."""
            Li, Ri = bicliques[i]
            Lj, Rj = bicliques[j]
            
            # The union of covered edges must be covered by the new biclique
            # Old covered edges
            old_covered = set()
            for u in Li:
                for v in Ri:
                    old_covered.add((min(u, v), max(u, v)))
            for u in Lj:
                for v in Rj:
                    old_covered.add((min(u, v), max(u, v)))
            
            # Try all 4 combinations
            options = [
                (set(Li) | set(Lj), set(Ri) & set(Rj)),
                (set(Li) | set(Rj), set(Ri) & set(Lj)),
                (set(Ri) | set(Lj), set(Li) & set(Rj)),
                (set(Ri) | set(Rj), set(Li) & set(Lj)),
            ]
            
            for L_cand, R_cand in options:
                if not L_cand or not R_cand:
                    continue
                # Check no overlap between L_cand and R_cand
                if L_cand & R_cand:
                    continue
                # Check validity
                if not is_valid_biclique(list(L_cand), list(R_cand)):
                    continue
                # Check coverage
                new_covered = set()
                for u in L_cand:
                    for v in R_cand:
                        new_covered.add((min(u, v), max(u, v)))
                if old_covered.issubset(new_covered):
                    return [list(L_cand), list(R_cand)]
            
            return None
        
        # Try merges iteratively
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            
            merged = False
            n_b = len(bicliques)
            
            for i in range(n_b):
                if time.time() > deadline:
                    break
                for j in range(i + 1, n_b):
                    if time.time() > deadline:
                        break
                    
                    result = try_merge_pair(i, j, bicliques)
                    if result is not None:
                        bicliques.pop(j)
                        bicliques.pop(i)
                        bicliques.append(result)
                        merged = True
                        break
                if merged:
                    break
            
            if not merged:
                break
        
        return bicliques

    # Run greedy with multiple restarts
    best = None
    best_count_local = best_count
    
    # First, try deterministic greedy
    det_results = greedy_partition(randomize_start=False, n_restarts=1)
    if det_results:
        for b in det_results:
            if len(b) < best_count_local:
                best_count_local = len(b)
                best = b
    
    # Then try randomized greedy with many restarts
    rand_results = greedy_partition(randomize_start=True, n_restarts=300)
    if rand_results:
        for b in rand_results:
            if len(b) < best_count_local:
                best_count_local = len(b)
                best = b
    
    if best is not None:
        # Apply local search
        improved = local_search(best)
        if len(improved) < best_count_local:
            best_count_local = len(improved)
            best = improved
        else:
            best = improved
    
    if best is None:
        # Fallback to star decomposition
        best = best_bicliques
        best_count_local = best_count

    # Verify the result
    def verify_partition(bicliques):
        """Verify that the partition is valid."""
        # Check all cross pairs are host edges
        for L, R in bicliques:
            for u in L:
                for v in R:
                    if (min(u, v), max(u, v)) not in edge_set:
                        return False
        
        # Check all edges are covered exactly once
        covered = set()
        for L, R in bicliques:
            for u in L:
                for v in R:
                    e = (min(u, v), max(u, v))
                    if e in covered:
                        return False  # Double covered
                    covered.add(e)
        
        if covered != edge_set:
            return False
        
        return True
    
    if not verify_partition(best):
        # If verification fails, fall back to star decomposition
        best = best_bicliques
        best_count_local = best_count

    # Convert to required format
    result = []
    for b in best:
        left = [int(x) for x in b[0]]
        right = [int(x) for x in b[1]]
        result.append([left, right])
    
    return result
