import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    def count_k4_ind6(adj, n):
        """Fast count of K4s and independent 6-sets using bitmasks."""
        # Build adjacency masks
        masks = [0] * n
        for u in range(n):
            m = 0
            for v in range(n):
                if adj[u, v]:
                    m |= (1 << v)
            masks[u] = m
        
        # Count K4s: for each edge (u,v), look at common neighbors, count edges among them
        k4_count = 0
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v]:
                    common = masks[u] & masks[v]
                    # Count edges in the subgraph induced by common neighbors
                    # Each edge (w,x) in common with w<x forms a K4 with u,v
                    c = common
                    while c:
                        lsb = c & -c
                        wi = lsb.bit_length() - 1
                        c ^= lsb
                        if wi > v:  # only count w > v to avoid double counting
                            rest = c & masks[wi]
                            # Count bits in rest
                            k4_count += bin(rest).count('1')
        
        # Count independent 6-sets: K6 in complement
        # Complement masks
        full_mask = (1 << n) - 1
        comp_masks = [0] * n
        for u in range(n):
            cm = full_mask & ~(masks[u] | (1 << u))
            comp_masks[u] = cm
        
        ind6_count = 0
        # Count K6 in complement graph
        # For each 4-clique in complement, check if it can be extended to 6
        # Simpler: iterate 6-combinations but with early termination
        # For n <= 35, C(35,6) = 1.6M which is too slow
        # Use recursive counting with pruning
        
        def count_cliques_k(mask, k, size):
            """Count k-cliques in graph defined by mask (bitmask of allowed vertices)."""
            if k == 0:
                return 1
            if size < k:
                return 0
            if k == 1:
                return size
            
            count = 0
            m = mask
            while m:
                lsb = m & -m
                vi = lsb.bit_length() - 1
                m ^= lsb
                # New mask: intersection of current mask (without vi) and neighbors of vi
                new_mask = (mask & ~lsb) & comp_masks[vi]
                new_size = bin(new_mask).count('1')
                if new_size >= k - 1:
                    count += count_cliques_k(new_mask, k - 1, new_size)
            return count
        
        # For n up to 35, count K6 in complement
        if n <= 30:
            ind6_count = count_cliques_k(full_mask, 6, n)
        else:
            # For larger n, use a faster check: just check if ANY independent 6 exists
            # Use backtracking
            def has_independent_6(adj, n):
                # Greedy + backtracking
                # Try to build independent set of size 6
                def backtrack(start, current_set, remaining):
                    if len(current_set) == 6:
                        return True
                    if remaining < 6 - len(current_set):
                        return False
                    for v in range(start, n):
                        if v not in current_set:
                            # Check if v is independent from all in current_set
                            ok = True
                            for u in current_set:
                                if adj[u, v]:
                                    ok = False
                                    break
                            if ok:
                                if backtrack(v + 1, current_set + [v], remaining - 1):
                                    return True
                    return False
                return backtrack(0, [], n)
            
            # For large n, we just need to know if there's ANY ind6
            # Use a different approach: check all 6-subsets with early exit
            # But this is O(n^6) worst case...
            # For n=35, let's use the clique counting but with early exit
            ind6_count = 0
            m = full_mask
            while m and ind6_count == 0:
                lsb = m & -m
                v0 = lsb.bit_length() - 1
                m ^= lsb
                m2 = (m & comp_masks[v0])
                while m2 and ind6_count == 0:
                    lsb2 = m2 & -m2
                    v1 = lsb2.bit_length() - 1
                    m2 ^= lsb2
                    m3 = (m2 & comp_masks[v1])
                    while m3 and ind6_count == 0:
                        lsb3 = m3 & -m3
                        v2 = lsb3.bit_length() - 1
                        m3 ^= lsb3
                        m4 = (m3 & comp_masks[v2])
                        while m4 and ind6_count == 0:
                            lsb4 = m4 & -m4
                            v3 = lsb4.bit_length() - 1
                            m4 ^= lsb4
                            m5 = (m4 & comp_masks[v3])
                            while m5 and ind6_count == 0:
                                lsb5 = m5 & -m5
                                v4 = lsb5.bit_length() - 1
                                m5 ^= lsb5
                                m6 = (m5 & comp_masks[v4])
                                if m6:
                                    ind6_count = 1  # Found at least one
                                    break
            # ind6_count is just 0 or 1 for large n, which is fine for cost minimization
        
        return k4_count, ind6_count

    def try_spectral(n, time_budget):
        """Try to find a valid graph using spectral construction + local search."""
        end_time = time.time() + time_budget
        best_adj = None
        best_cost = float('inf')
        
        # Generate several candidate initial graphs
        candidates = []
        
        # 1. Paley-type graph
        if n == 17:
            squares = {x * x % 17 for x in range(1, 17)}
            adj_paley = np.zeros((n, n), dtype=bool)
            for i in range(n):
                for j in range(i + 1, n):
                    if (j - i) % 17 in squares:
                        adj_paley[i, j] = True
                        adj_paley[j, i] = True
            candidates.append(adj_paley.copy())
        
        # 2. Random symmetric matrices with various densities
        for density in [0.3, 0.4, 0.5, 0.6, 0.7]:
            adj_rand = np.zeros((n, n), dtype=bool)
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < density:
                        adj_rand[i, j] = True
                        adj_rand[j, i] = True
            candidates.append(adj_rand)
        
        # 3. Spectral: top eigenvectors of random symmetric matrices
        for _ in range(3):
            M = np.random.randn(n, n)
            M = (M + M.T) / 2  # Symmetrize
            eigenvalues, eigenvectors = np.linalg.eigh(M)
            # Take top k eigenvectors, combine
            k = max(1, n // 5)
            top_eigs = eigenvectors[:, -k:]
            # Combine: sign of sum of top eigenvector components
            combined = np.sum(top_eigs, axis=1)
            adj_spec = np.zeros((n, n), dtype=bool)
            for i in range(n):
                for j in range(i + 1, n):
                    # If signs are the same, connect
                    if np.sign(combined[i]) == np.sign(combined[j]) and np.sign(combined[i]) != 0:
                        adj_spec[i, j] = True
                        adj_spec[j, i] = True
            candidates.append(adj_spec)
        
        # 4. Structured: block diagonal with Paley 17 + random connection
        if n >= 18:
            adj_block = np.zeros((n, n), dtype=bool)
            # First 17 vertices: Paley
            squares = {x * x % 17 for x in range(1, 17)}
            for i in range(17):
                for j in range(i + 1, 17):
                    if (j - i) % 17 in squares:
                        adj_block[i, j] = True
                        adj_block[j, i] = True
            # Remaining vertices: random
            for i in range(17, n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj_block[i, j] = True
                        adj_block[j, i] = True
            # Connection between blocks
            for i in range(17):
                for j in range(17, n):
                    if random.random() < 0.4:
                        adj_block[i, j] = True
                        adj_block[j, i] = True
            candidates.append(adj_block)
        
        # Evaluate and pick best candidate, then do local search
        best_candidate = None
        best_initial_cost = float('inf')
        
        for adj in candidates:
            k4, ind6 = count_k4_ind6(adj, n)
            cost = k4 + ind6 * 100  # Weight ind6 more heavily
            if cost < best_initial_cost:
                best_initial_cost = cost
                best_candidate = adj.copy()
        
        if best_candidate is None:
            return None
        
        adj = best_candidate
        current_cost = best_initial_cost
        
        # Local search with simulated annealing
        T = 5.0
        T_min = 0.1
        cooling = 0.99
        
        while time.time() < end_time:
            # Pick random edge to flip
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            
            old_state = adj[u, v]
            adj[u, v] = not old_state
            adj[v, u] = not old_state
            
            new_k4, new_ind6 = count_k4_ind6(adj, n)
            new_cost = new_k4 + new_ind6 * 100
            
            if new_cost < current_cost:
                current_cost = new_cost
                if current_cost == 0:
                    # Valid graph found
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if adj[i, j]:
                                edges.append([i, j])
                    return edges
            else:
                delta = new_cost - current_cost
                if random.random() < np.exp(-delta / T):
                    current_cost = new_cost
                else:
                    adj[u, v] = old_state
                    adj[v, u] = old_state
            
            T = max(T_min, T * cooling)
        
        return None

    # Start with known good graph
    n_start = 17
    squares = {x * x % 17 for x in range(1, 17)}
    best_edges = [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]
    best_n = 17
    
    # Try to improve from 18 up to 35
    for n in range(18, 36):
        if time.time() > deadline:
            break
        
        # Give more time to promising n values
        if n <= 22:
            budget = 3.0
        elif n <= 28:
            budget = 2.0
        else:
            budget = 1.5
        
        edges = try_spectral(n, budget)
        if edges is not None:
            best_n = n
            best_edges = edges
            if n >= 34:
                break
    
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
