"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    def _count_k4(adj, n):
        """Count K4s in the graph. adj is list of bitmasks."""
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] >> j) & 1:
                    continue
                common = adj[i] & adj[j]
                common &= ~(1 << i)
                common &= ~(1 << j)
                # For each pair (k, l) in common, if k~l, it's a K4
                # Iterate through bits in common
                mask = common
                while mask:
                    k = (mask & -mask).bit_length() - 1
                    mask &= mask - 1
                    if k <= j: continue
                    # Check edges between k and other vertices in common
                    # We just need to know if there is ANY edge between two vertices in common
                    # Actually, we need to count distinct K4s.
                    # A K4 is {i, j, k, l} where i<j<k<l.
                    # If we iterate i, j, then find edges in common(i,j) where k,l > j,
                    # we might double count or miss.
                    # Better: Iterate all 4-tuples? Too slow.
                    # Efficient K4 count:
                    # For each edge (i,j), let C = N(i) \cap N(j).
                    # Number of K4s containing (i,j) is the number of edges in the subgraph induced by C.
                    # Sum over all edges (i,j) of E(C_{ij}) / 6? No, each K4 has 6 edges.
                    # So total K4s = (1/6) * sum_{(i,j) in E} E(N(i) \cap N(j)).
                    # Let's just count edges in C.
                    c_mask = common
                    edge_count_in_c = 0
                    while c_mask:
                        k = (c_mask & -c_mask).bit_length() - 1
                        c_mask &= c_mask - 1
                        if k <= j: continue
                        # Check edges from k to other vertices in C
                        # We only care about l > k to avoid double counting within C
                        l_mask = c_mask & adj[k]
                        # Remove vertices <= k from l_mask?
                        # We want l > k.
                        # Since we are iterating k, and l_mask contains neighbors of k in C,
                        # we need to ensure l > k.
                        # c_mask still contains vertices > k (since we removed k).
                        # So l_mask & c_mask gives neighbors of k in C that are > k.
                        l_mask &= c_mask
                        while l_mask:
                            edge_count_in_c += 1
                            l_mask &= l_mask - 1
                    count += edge_count_in_c
        return count // 6

    def _count_ind6(adj, n):
        """Count independent sets of size 6. Equivalent to cliques of size 6 in complement."""
        # This is expensive. We will use a heuristic check in the main loop,
        # but for the objective function, we need a count or a boolean.
        # For optimization, we can just check if ANY exists.
        # Let's implement a fast existence check.
        return _has_ind6(adj, n)

    def _has_ind6(adj, n):
        """Check if there is an independent set of size 6."""
        # Independent set in G is Clique in G_complement.
        # We can use a recursive search with pruning.
        # Or, for small n, combinations. For n=35, combinations(35,6) is ~1.6M.
        # 1.6M checks * 15 edge checks = 24M ops. Might be too slow for every step.
        # We will use this only for final verification or in a less frequent check.
        # For the local search objective, we will use a randomized check or a faster heuristic.
        
        # Fast check: 
        # If alpha(G) >= 6, there is a set of 6 vertices with no edges.
        # We can use a greedy coloring lower bound? No.
        # Let's just use combinations for n <= 25, and randomized for n > 25.
        if n <= 25:
            for combo in itertools.combinations(range(n), 6):
                valid = True
                for i in range(6):
                    for j in range(i+1, 6):
                        if (adj[combo[i]] >> combo[j]) & 1:
                            valid = False
                            break
                    if not valid: break
                if valid:
                    return True
            return False
        else:
            # Randomized check
            for _ in range(100):
                combo = random.sample(range(n), 6)
                valid = True
                for i in range(6):
                    for j in range(i+1, 6):
                        if (adj[combo[i]] >> combo[j]) & 1:
                            valid = False
                            break
                    if not valid: break
                if valid:
                    return True
            return False

    def _is_valid(adj, n):
        if _count_k4(adj, n) > 0:
            return False
        if _has_ind6(adj, n):
            return False
        return True

    def _build_adj(n, edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    # Known valid seeds
    def _clebsch_edges():
        edges = []
        for i in range(16):
            for j in range(i + 1, 16):
                d = bin(i ^ j).count('1')
                if d == 1 or d == 3:
                    edges.append([i, j])
        return edges

    def _paley_edges():
        n = 17
        squares = {x * x % n for x in range(1, n)}
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (j - i) % n in squares:
                    edges.append([i, j])
        return edges

    best_n = 16
    best_edges = _clebsch_edges()
    
    # Verify seeds
    adj_c = _build_adj(16, best_edges)
    if _is_valid(adj_c, 16):
        pass # It is valid
    
    adj_p = _build_adj(17, _paley_edges())
    if _is_valid(adj_p, 17):
        best_n = 17
        best_edges = _paley_edges()

    # Search for larger n
    # We will try to find a valid graph for n from 35 down to best_n + 1
    # Using random restarts and local search (Simulated Annealing style)
    
    for target_n in range(35, best_n, -1):
        if time.time() > deadline - 10.0:
            break
            
        found = False
        # Number of restarts depends on time available
        # We'll do a few restarts for each n
        for attempt in range(50):
            if time.time() > deadline - 5.0:
                break
            
            # Initialize random graph
            adj = [0] * target_n
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Local Search
            # Objective: Minimize violations.
            # We will flip edges to reduce violations.
            # Since counting violations is expensive, we will do a "repair" strategy:
            # 1. Find a K4 or Ind6.
            # 2. Flip an edge in it.
            
            current_violations = 0 # We don't track count, just existence
            
            for step in range(200):
                if time.time() > deadline - 2.0:
                    break
                
                # Check K4
                k4_found = None
                for a in range(target_n):
                    for b in range(a + 1, target_n):
                        if not (adj[a] >> b) & 1:
                            continue
                        common = adj[a] & adj[b]
                        common &= ~(1 << a)
                        common &= ~(1 << b)
                        c_mask = common
                        while c_mask:
                            c = (c_mask & -c_mask).bit_length() - 1
                            c_mask &= c_mask - 1
                            if c <= b: continue
                            common_abc = common & adj[c]
                            common_abc &= ~(1 << a)
                            common_abc &= ~(1 << b)
                            common_abc &= ~(1 << c)
                            d_mask = common_abc
                            while d_mask:
                                d = (d_mask & -d_mask).bit_length() - 1
                                d_mask &= d_mask - 1
                                if d > c:
                                    k4_found = [a, b, c, d]
                                    break
                            if k4_found: break
                        if k4_found: break
                    if k4_found: break
                
                if k4_found:
                    # Flip a random edge in the K4
                    random.shuffle(k4_found)
                    u, v = k4_found[0], k4_found[1]
                    if (adj[u] >> v) & 1:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                    else:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                    continue
                
                # Check Ind6
                ind6_found = None
                if target_n <= 25:
                    for combo in itertools.combinations(range(target_n), 6):
                        valid = True
                        for i in range(6):
                            for j in range(i+1, 6):
                                if (adj[combo[i]] >> combo[j]) & 1:
                                    valid = False
                                    break
                            if not valid: break
                        if valid:
                            ind6_found = list(combo)
                            break
                else:
                    # Randomized search for Ind6
                    for _ in range(50):
                        combo = random.sample(range(target_n), 6)
                        valid = True
                        for i in range(6):
                            for j in range(i+1, 6):
                                if (adj[combo[i]] >> combo[j]) & 1:
                                    valid = False
                                    break
                            if not valid: break
                        if valid:
                            ind6_found = list(combo)
                            break
                
                if ind6_found:
                    # Flip a random edge in the Ind6
                    random.shuffle(ind6_found)
                    u, v = ind6_found[0], ind6_found[1]
                    if (adj[u] >> v) & 1:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                    else:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                    continue
                
                # No violations found
                # Verify exactly
                if _is_valid(adj, target_n):
                    edges = []
                    for i in range(target_n):
                        for j in range(i + 1, target_n):
                            if (adj[i] >> j) & 1:
                                edges.append([i, j])
                    best_n = target_n
                    best_edges = edges
                    found = True
                    break
            
            if found:
                break
        
        if found:
            break

    return {'n': best_n, 'edges': best_edges}

# EVOLVE-BLOCK-END
