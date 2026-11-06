import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    def get_edges(adj, n):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    def exact_check(n, adj):
        """Exact check: no K4 and no independent 6-set."""
        # Check K4: for each edge (u,v), check common neighbors for triangles
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    common = adj[u] & adj[v]
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    # Count edges within common neighborhood
                    m = common
                    while m:
                        lsb = m & -m
                        a = lsb.bit_length() - 1
                        m ^= lsb
                        rest = common & ~(1 << a)
                        if adj[a] & rest:
                            return False
        # Check independent 6-set using bitset backtracking
        all_mask = (1 << n) - 1
        def find_indep_6(remaining, size):
            if size == 6:
                return True
            if remaining.bit_count() < size:
                return False
            m = remaining
            while m:
                lsb = m & -m
                u = lsb.bit_length() - 1
                m ^= lsb
                non_n = remaining & ~(adj[u] | (1 << u))
                if non_n.bit_count() >= size - 1:
                    if find_indep_6(non_n, size - 1):
                        return True
            return False
        if find_indep_6(all_mask, 6):
            return False
        return True

    def compute_energy(n, adj):
        """Compute energy: number of K4s + number of independent 6-sets."""
        k4_count = 0
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    common = adj[u] & adj[v]
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    m = common
                    while m:
                        lsb = m & -m
                        a = lsb.bit_length() - 1
                        m ^= lsb
                        rest = common & ~(1 << a)
                        k4_count += (adj[a] & rest).bit_count()
        k4_count //= 3

        i6_count = 0
        all_mask = (1 << n) - 1
        def count_indep_sets(remaining, size):
            if size == 0:
                return 1
            if remaining.bit_count() < size:
                return 0
            lsb = remaining & -remaining
            u = lsb.bit_length() - 1
            rest = remaining ^ lsb
            count = count_indep_sets(rest, size)
            non_n = rest & ~adj[u]
            count += count_indep_sets(non_n, size - 1)
            return count
        i6_count = count_indep_sets(all_mask, 6)
        return k4_count + i6_count

    def delta_energy_add_edge(n, adj, u, v):
        """Energy change when ADDING edge (u,v)."""
        common = adj[u] & adj[v]
        common &= ~(1 << u)
        common &= ~(1 << v)
        tri_count = 0
        m = common
        while m:
            lsb = m & -m
            a = lsb.bit_length() - 1
            m ^= lsb
            rest = common & ~(1 << a)
            tri_count += (adj[a] & rest).bit_count()
        tri_count //= 3

        all_mask = (1 << n) - 1
        non_n = all_mask & ~(adj[u] | adj[v] | (1 << u) | (1 << v))
        def count_indep_sets(remaining, size):
            if size == 0:
                return 1
            if remaining.bit_count() < size:
                return 0
            lsb = remaining & -remaining
            u2 = lsb.bit_length() - 1
            rest = remaining ^ lsb
            count = count_indep_sets(rest, size)
            non_n2 = rest & ~adj[u2]
            count += count_indep_sets(non_n2, size - 1)
            return count
        i5_count = count_indep_sets(non_n, 5)
        return tri_count - i5_count

    def delta_energy_remove_edge(n, adj, u, v):
        """Energy change when REMOVING edge (u,v)."""
        common = adj[u] & adj[v]
        common &= ~(1 << u)
        common &= ~(1 << v)
        tri_count = 0
        m = common
        while m:
            lsb = m & -m
            a = lsb.bit_length() - 1
            m ^= lsb
            rest = common & ~(1 << a)
            tri_count += (adj[a] & rest).bit_count()
        tri_count //= 3

        all_mask = (1 << n) - 1
        non_n = all_mask & ~(adj[u] | adj[v] | (1 << u) | (1 << v))
        def count_indep_sets(remaining, size):
            if size == 0:
                return 1
            if remaining.bit_count() < size:
                return 0
            lsb = remaining & -remaining
            u2 = lsb.bit_length() - 1
            rest = remaining ^ lsb
            count = count_indep_sets(rest, size)
            non_n2 = rest & ~adj[u2]
            count += count_indep_sets(non_n2, size - 1)
            return count
        i5_count = count_indep_sets(non_n, 5)
        return -tri_count + i5_count

    def simulated_annealing(n, adj, T_start, T_end, steps):
        """Run SA on the Max-SAT energy landscape."""
        current_energy = compute_energy(n, adj)
        best_energy = current_energy
        best_adj = adj[:]
        
        for s in range(steps):
            if time.time() > deadline - 10:
                break
            
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            
            is_edge = bool((adj[u] >> v) & 1)
            
            if is_edge:
                d = delta_energy_remove_edge(n, adj, u, v)
            else:
                d = delta_energy_add_edge(n, adj, u, v)
            
            T = T_start * (T_end / T_start) ** (s / max(steps - 1, 1))
            
            if d < 0 or (T > 0 and random.random() < math.exp(-d / T)):
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
                current_energy += d
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_adj = adj[:]
                    
                    if best_energy == 0:
                        return best_adj, 0, True
        
        return best_adj, best_energy, best_energy == 0

    def random_init(n, p=0.5):
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def structured_init(n):
        """Initialize with a quasi-random structure."""
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                d = (v - u) % n
                if d % 3 != 0 and d % 5 != 0:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def backtrack_construct(target_n, time_limit):
        """
        Constraint-programming-style backtracking: construct vertices sequentially.
        For each new vertex, determine which subsets of previous vertices it can be
        adjacent to without creating a K4 or an independent 6-set.
        
        We maintain:
        - adj: list of adjacency bitmasks for vertices 0..k-1
        - For vertex k, we need to choose a subset S of {0..k-1} such that:
          1. No K4 is formed: for any two vertices a,b in S that are adjacent,
             their common neighbors in {0..k-1} must not include both a and b
             (i.e., N(a) ∩ N(b) ∩ S has size < 2, or more precisely, no triangle
             in the common neighborhood that would form K4 with a,b)
          2. No independent 6-set is formed: the non-neighborhood of k within
             {0..k-1} must not contain an independent 5-set.
        
        For efficiency, we use the following pruning:
        - When adding vertex k, let S be its neighbor set among {0..k-1}.
        - K4 constraint: For any edge (a,b) in the existing graph where a,b ∈ S,
          we need that N(a) ∩ N(b) ∩ S has no edge (otherwise a,b and two common
          neighbors form a K4). More precisely: for any a,b ∈ S with a~b,
          the set C = N(a) ∩ N(b) ∩ S must be an independent set in the subgraph
          induced by S (no two vertices in C can be adjacent, since that would
          form a K4 with a,b).
        - I6 constraint: The set T = {0..k-1} \\ S (non-neighbors of k) must not
          contain an independent 5-set in the existing graph.
        
        We enumerate possible S values, preferring those that are "balanced".
        """
        adj = [0] * target_n
        k = 0  # number of vertices placed so far
        
        # Precompute helper: for a given existing graph on k vertices,
        # check if a candidate neighbor set S for vertex k is valid.
        
        def is_valid_K4(k, S, adj):
            """Check that adding vertex k with neighbor set S doesn't create K4."""
            # For any edge (a,b) in existing graph with a,b ∈ S:
            # N(a) ∩ N(b) ∩ S must be independent in G[S]
            for a in range(k):
                if not ((S >> a) & 1):
                    continue
                for b in range(a + 1, k):
                    if not ((S >> b) & 1):
                        continue
                    if not ((adj[a] >> b) & 1):
                        continue
                    # a and b are adjacent and both in S
                    # Common neighbors of a and b in S
                    common = adj[a] & adj[b] & S
                    common &= ~(1 << a)
                    common &= ~(1 << b)
                    # Check if common has any edge within S
                    m = common
                    while m:
                        lsb = m & -m
                        c = lsb.bit_length() - 1
                        m ^= lsb
                        rest = common & ~(1 << c)
                        if adj[c] & rest:
                            return False
            return True

        def has_indep_5_in_set(T, adj, k):
            """Check if the set T (subset of {0..k-1}) contains an independent 5-set."""
            if T.bit_count() < 5:
                return False
            
            def find_indep(remaining, size):
                if size == 5:
                    return True
                if remaining.bit_count() < size:
                    return False
                m = remaining
                while m:
                    lsb = m & -m
                    u = lsb.bit_length() - 1
                    m ^= lsb
                    non_n = remaining & ~(adj[u] | (1 << u))
                    if non_n.bit_count() >= size - 1:
                        if find_indep(non_n, size - 1):
                            return True
                return False
            
            return find_indep(T, 5)

        def is_valid_I6(k, S, adj):
            """Check that adding vertex k with neighbor set S doesn't create I6."""
            # Non-neighbors of k: T = {0..k-1} \\ S
            T = ((1 << k) - 1) & ~S
            return not has_indep_5_in_set(T, adj, k)

        def is_valid(k, S, adj):
            return is_valid_K4(k, S, adj) and is_valid_I6(k, S, adj)

        # Backtracking search
        # We'll use a randomized approach: try random S values, with bias toward
        # "balanced" densities.
        
        max_depth = target_n
        current_k = 0
        
        # State: adj[0..current_k-1] are fixed
        # We need to find S for vertex current_k
        
        # For efficiency, we'll do a limited search at each level
        # and use randomized restarts
        
        best_solution = None
        
        def search(current_k, adj, depth, max_attempts):
            nonlocal best_solution
            if time.time() > time_limit:
                return None
            
            if current_k == max_depth:
                # Found a complete solution
                return adj[:]
            
            # Try to find a valid S for vertex current_k
            # Generate candidate S values
            candidates = set()
            
            # Always try empty set and full set
            candidates.add(0)
            candidates.add((1 << current_k) - 1)
            
            # Try random subsets with various densities
            for _ in range(max_attempts):
                if time.time() > time_limit:
                    return None
                # Random density between 0.3 and 0.7
                p = 0.3 + 0.4 * random.random()
                S = 0
                for v in range(current_k):
                    if random.random() < p:
                        S |= (1 << v)
                candidates.add(S)
            
            # Also try "structured" candidates
            # Cyclic patterns
            for shift in range(min(current_k, 8)):
                S = 0
                for v in range(current_k):
                    if (v + shift) % 3 != 0:
                        S |= (1 << v)
                candidates.add(S)
            
            # Shuffle and try each candidate
            cand_list = list(candidates)
            random.shuffle(cand_list)
            
            for S in cand_list:
                if time.time() > time_limit:
                    return None
                
                if not is_valid(current_k, S, adj):
                    continue
                
                # Accept this S and recurse
                adj[current_k] = S
                # Update adj for previous vertices
                for v in range(current_k):
                    if (S >> v) & 1:
                        adj[v] |= (1 << current_k)
                    else:
                        adj[v] &= ~(1 << current_k)
                
                result = search(current_k + 1, adj, depth + 1, max_attempts)
                if result is not None:
                    return result
                
                # Backtrack
                adj[current_k] = 0
                for v in range(current_k):
                    if (S >> v) & 1:
                        adj[v] &= ~(1 << current_k)
                    else:
                        adj[v] |= (1 << current_k)
            
            return None
        
        # Multiple restarts with different random seeds
        for restart in range(20):
            if time.time() > time_limit - 5:
                break
            
            random.seed(restart * 7919 + 12345)
            adj = [0] * target_n
            result = search(0, adj, 0, 50)
            if result is not None:
                return result
        
        return None

    best_graph = None
    best_n = 0

    # Strategy 1: Try backtracking construction for smaller n first to validate
    # then scale up
    for target_n in [20, 25, 30, 33, 35]:
        if time.time() > deadline - 30:
            break
        
        # Give backtracking a time budget
        bt_time = min(30, deadline - time.time() - 20)
        if bt_time < 5:
            break
        
        result = backtrack_construct(target_n, start_time + bt_time)
        if result is not None:
            # Verify
            if exact_check(target_n, result):
                best_graph = result
                best_n = target_n
                break
    
    # Strategy 2: If backtracking didn't find a large graph, use SA
    if best_n < 30:
        for target_n in [35, 34, 33, 32, 31, 30]:
            if time.time() > deadline - 20:
                break
            
            found = False
            
            # Phase 1: Structured initializations with SA
            for restart in range(5):
                if time.time() > deadline - 15:
                    break
                if found:
                    break
                
                adj = structured_init(target_n)
                adj, energy, success = simulated_annealing(
                    target_n, adj,
                    T_start=8.0, T_end=0.01,
                    steps=8000
                )
                if success:
                    best_graph = adj
                    best_n = target_n
                    found = True
                    break
            
            # Phase 2: Random initializations with SA
            if not found:
                for restart in range(20):
                    if time.time() > deadline - 15:
                        break
                    if found:
                        break
                    
                    p = 0.3 + 0.4 * random.random()
                    adj = random_init(target_n, p)
                    adj, energy, success = simulated_annealing(
                        target_n, adj,
                        T_start=10.0, T_end=0.01,
                        steps=10000
                    )
                    if success:
                        best_graph = adj
                        best_n = target_n
                        found = True
                        break
            
            if found:
                break
    
    if best_graph is None:
        # Final fallback: Paley graph of order 17
        squares = {x * x % 17 for x in range(1, 17)}
        adj = [0] * 17
        for i in range(17):
            for j in range(i + 1, 17):
                if (j - i) % 17 in squares:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return {'n': 17, 'edges': get_edges(adj, 17)}
    else:
        return {'n': best_n, 'edges': get_edges(best_graph, best_n)}
