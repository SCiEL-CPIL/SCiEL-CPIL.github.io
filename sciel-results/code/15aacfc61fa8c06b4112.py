import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Helper to check validity
    def is_valid(adj, n):
        # Check K4
        for i in range(n):
            for j in range(i+1, n):
                if not (adj[i] >> j) & 1:
                    continue
                # Find common neighbors
                common = adj[i] & adj[j]
                # Remove i and j
                common &= ~(1 << i)
                common &= ~(1 << j)
                # Check if any edge exists among common neighbors
                # Iterate over bits in common
                k = common
                while k:
                    # Get lowest set bit
                    lsb = k & -k
                    idx = lsb.bit_length() - 1
                    k -= lsb
                    # Check if idx has a neighbor in common
                    if (adj[idx] & common) & ~(1 << idx):
                        return False
        # Check I6
        # We can use a recursive backtracking or just check all 6-subsets
        # For n=35, 1.6M subsets is borderline for pure Python loops in 115s if done repeatedly.
        # But we only need to check validity once per candidate in the SA loop? No, SA needs cost.
        # Let's stick to the delta update method for SA.
        pass

    # Let's refine the cost calculation for SA
    def count_k4(adj, n):
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                if (adj[i] >> j) & 1:
                    common = adj[i] & adj[j]
                    common &= ~(1 << i)
                    common &= ~(1 << j)
                    # Count edges in common
                    # For each k in common, count neighbors in common
                    # This is O(n^4) worst case if not careful, but common is small usually.
                    # Actually, just checking existence is enough for validity, but for cost we need count.
                    # Let's just count K4s by iterating 4-subsets if n is small, or use the property.
                    pass
        return count

    # Given the complexity, let's try a simpler approach:
    # 1. Try to find a known 35-vertex graph? No, we must generate it.
    # 2. Use a random restart hill climbing / SA on n=35, 34, ...
    
    def cost(adj, n):
        # Returns number of K4s + number of I6s
        k4_count = 0
        i6_count = 0
        
        # K4s
        # Iterate all 4-subsets
        # This is too slow for every SA step if done naively.
        # We need incremental update.
        
        pass

    # Let's use a different strategy: 
    # The problem is small enough (n<=35) that we can use a complete search with pruning for I6 and K4?
    # No, that's hard.
    
    # Let's go back to SA with incremental cost.
    
    def get_cost_delta(adj, n, u, v, flipping):
        # Returns delta_k4, delta_i6
        # flipping: True if adding edge, False if removing
        delta_k4 = 0
        delta_i6 = 0
        
        # Affected K4s: Those containing {u, v}
        # A K4 containing u,v requires two other vertices w,z such that
        # w,z are neighbors of u and v (if adding) or non-neighbors (if removing? No, K4 requires edges)
        # If adding edge (u,v):
        #   New K4s are formed if there exist w,z such that
        #   w~u, w~v, z~u, z~v, w~z
        #   And previously (u,v) was not an edge, so no K4 contained (u,v).
        #   So we count pairs (w,z) in (N(u) \cap N(v)) such that w~z.
        # If removing edge (u,v):
        #   We lose K4s that contained (u,v).
        #   Same condition: pairs (w,z) in (N(u) \cap N(v) \ {u,v}) such that w~z.
        
        if flipping:
            # Adding edge
            common = adj[u] & adj[v]
            common &= ~(1 << u)
            common &= ~(1 << v)
            # Count edges in common
            k = common
            while k:
                lsb = k & -k
                idx = lsb.bit_length() - 1
                k -= lsb
                # Check if idx is connected to any other in common
                # We just need to count edges.
                # To count edges in a subgraph defined by bitmask 'common':
                # Sum over bits in common of popcount(adj[idx] & common) / 2
                # But we are in a loop, let's just do it.
                pass
        else:
            # Removing edge
            pass

    # This is getting complicated to implement correctly and fast in Python.
    
    # Alternative:
    # Use the fact that for n=35, the number of edges is around 35*34/2 * p.
    # For R(4,6), density is likely around 0.5?
    
    # Let's try a simpler heuristic:
    # Start with a random graph.
    # While time remaining:
    #   Find a random K4 or I6.
    #   Flip a random edge in it to try to break it.
    #   This is a "Min-Conflicts" style.
    
    def find_violation(adj, n):
        # Returns a K4 (list of 4) or I6 (list of 6) or None
        # Check K4s first (fewer)
        for i in range(n):
            for j in range(i+1, n):
                if (adj[i] >> j) & 1:
                    common = adj[i] & adj[j]
                    common &= ~(1 << i)
                    common &= ~(1 << j)
                    # Get list of common neighbors
                    neighbors = []
                    k = common
                    while k:
                        lsb = k & -k
                        idx = lsb.bit_length() - 1
                        neighbors.append(idx)
                        k -= lsb
                    # Check for edge among neighbors
                    for a_idx in range(len(neighbors)):
                        w = neighbors[a_idx]
                        for b_idx in range(a_idx+1, len(neighbors)):
                            z = neighbors[b_idx]
                            if (adj[w] >> z) & 1:
                                return ('K4', [i, j, w, z])
        # Check I6s
        # This is expensive.
        # We can sample?
        # Or just check all 6-subsets?
        # For n=35, 1.6M subsets.
        # We can optimize by checking only if we haven't found K4.
        # But if graph is close to valid, we might have 0 K4s and some I6s.
        
        # Let's use a randomized check for I6 to save time in the loop,
        # and do a full check when we think we are close.
        
        return None

    # This is risky.
    
    # Let's try a different approach:
    # We know Exoo's 35-vertex graph exists.
    # Can we hardcode it? No, that's cheating/not general.
    # But we can try to search for it.
    
    # Let's implement a robust SA with full cost recalculation.
    # For n=35, full cost recalculation:
    # K4s: ~50k subsets.
    # I6s: ~1.6M subsets.
    # Total ~1.65M operations per cost eval.
    # In Python, 1.65M simple ops takes ~0.1s.
    # So we can do ~1000 steps in 100s.
    # This is not enough for SA to converge from random.
    
    # We need incremental cost.
    
    def popcount(x):
        return bin(x).count('1')

    def count_edges_in_mask(adj, mask):
        # Count edges within the set of vertices defined by mask
        # Sum popcount(adj[i] & mask) for i in mask, divide by 2
        total = 0
        k = mask
        while k:
            lsb = k & -k
            idx = lsb.bit_length() - 1
            k -= lsb
            total += popcount(adj[idx] & mask)
        return total // 2

    def calc_cost(adj, n):
        k4_count = 0
        i6_count = 0
        
        # K4s
        # Iterate pairs (u,v) that are edges
        for u in range(n):
            for v in range(u+1, n):
                if (adj[u] >> v) & 1:
                    common = adj[u] & adj[v]
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    # Count edges in common
                    k4_count += count_edges_in_mask(adj, common)
                    
        # I6s
        # Iterate 6-subsets? No, too slow.
        # We can count independent sets of size 6 using inclusion-exclusion or recursion?
        # Or just iterate all 6-subsets.
        # Let's try to optimize the I6 count.
        # An independent set of size 6 is a set of 6 vertices with no edges.
        # This is equivalent to a clique of size 6 in the complement graph.
        
        # Let's just iterate all 6-subsets.
        # We can prune?
        # If we have a set of 5 vertices that is independent, and we add a 6th,
        # it must not be connected to any of the 5.
        
        # Let's use a recursive backtracking to count I6s.
        # This might be faster if the graph is dense (few independent sets).
        # If the graph is sparse (many independent sets), it will be slow.
        # But we want to MINIMIZE I6s, so we expect few.
        
        def count_i6_rec(current_set, start_idx, adj, n):
            if len(current_set) == 6:
                return 1
            count = 0
            for i in range(start_idx, n):
                # Check if i can be added to current_set
                # i must not be connected to any in current_set
                ok = True
                for v in current_set:
                    if (adj[i] >> v) & 1:
                        ok = False
                        break
                if ok:
                    count += count_i6_rec(current_set + [i], i+1, adj, n)
            return count
        
        # This recursion is exponential. Worst case O(n^6).
        # For n=35, 35^6 is 1.8e9. Too slow.
        
        # So we must use the bitmask iteration for I6s, but it's 1.6M.
        # 1.6M iterations in Python is ~0.2s.
        # So full cost eval is ~0.3s.
        # We can do ~300 steps in 100s.
        # This is very slow for SA.
        
        # We MUST use incremental updates.
        
        pass

    # Given the constraints, maybe we should just try to find a graph with n=30 or 31?
    # Score 30/17 = 1.76. Score 31/17 = 1.82.
    # The inspiration said 1.176 (n=20) was found.
    # Let's aim for n=30.
    
    # Let's implement a simpler local search:
    # 1. Start with random graph on n=30.
    # 2. Repeat:
    #    a. Find a random K4 or I6.
    #    b. If none, return graph.
    #    c. Flip a random edge in the violation.
    #    d. Accept if cost decreases or with some probability.
    
    # To find a violation quickly:
    # - For K4: Iterate edges, check common neighbors for an edge.
    # - For I6: This is hard to find quickly without checking all.
    #   But if we are close to valid, we can just check all 6-subsets (1.6M for n=35, 500k for n=30).
    #   500k checks is ~0.05s.
    #   So we can afford to do a full scan for I6s every few steps.
    
    def find_i6(adj, n):
        # Returns a list of 6 vertices that are independent, or None
        # Use itertools.combinations
        import itertools
        for combo in itertools.combinations(range(n), 6):
            is_indep = True
            for i in range(6):
                for j in range(i+1, 6):
                    u = combo[i]
                    v = combo[j]
                    if (adj[u] >> v) & 1:
                        is_indep = False
                        break
                if not is_indep:
                    break
            if is_indep:
                return list(combo)
        return None

    def find_k4(adj, n):
        for i in range(n):
            for j in range(i+1, n):
                if (adj[i] >> j) & 1:
                    common = adj[i] & adj[j]
                    common &= ~(1 << i)
                    common &= ~(1 << j)
                    neighbors = []
                    k = common
                    while k:
                        lsb = k & -k
                        idx = lsb.bit_length() - 1
                        neighbors.append(idx)
                        k -= lsb
                    for a in range(len(neighbors)):
                        w = neighbors[a]
                        for b in range(a+1, len(neighbors)):
                            z = neighbors[b]
                            if (adj[w] >> z) & 1:
                                return [i, j, w, z]
        return None

    # Strategy:
    # Try n from 35 down to 17.
    # For each n, run a local search for a few seconds.
    # If found, return.
    
    best_n = 17
    best_adj = None
    # Initialize with Paley 17
    squares = {x * x % 17 for x in range(1, 17)}
    adj_17 = [0] * 17
    for i in range(17):
        for j in range(i+1, 17):
            if (j - i) % 17 in squares:
                adj_17[i] |= (1 << j)
                adj_17[j] |= (1 << i)
    best_adj = adj_17
    
    for n in range(35, 16, -1):
        if time.time() > deadline - 10:
            break
            
        # Run local search for this n
        # Use multiple restarts
        found = False
        for restart in range(50):
            if time.time() > deadline - 5:
                break
                
            # Random graph
            adj = [0] * n
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Local search
            for step in range(1000):
                if time.time() > deadline - 5:
                    break
                    
                # Check validity
                k4 = find_k4(adj, n)
                if k4:
                    # Flip a random edge in K4
                    u, v = random.sample(k4, 2)
                    if (adj[u] >> v) & 1:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                    else:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                    continue
                
                i6 = find_i6(adj, n)
                if i6:
                    # Flip a random edge in I6
                    u, v = random.sample(i6, 2)
                    if (adj[u] >> v) & 1:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                    else:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                    continue
                
                # No violations
                found = True
                best_n = n
                best_adj = adj
                break
                
            if found:
                break
                
    # Construct output
    edges = []
    for i in range(best_n):
        for j in range(i+1, best_n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
                
    return {'n': best_n, 'edges': edges}
