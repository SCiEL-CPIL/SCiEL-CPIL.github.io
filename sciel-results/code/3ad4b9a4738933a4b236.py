import time

def construct():
    deadline = time.time() + 85
    N = 12
    M = 210  # 2 * 3 * 5 * 7
    max_i = 10000 // M  # 47 if M=210, but 10000//210 = 47. Wait.
    # 210 * 47 = 9870. 210 * 48 = 10080 > 10000.
    # So max_i is 47.
    # We need 12 distinct integers from 1..47.
    # Total pairs to check: 47*46/2 = 1081.
    # This is very small.
    
    # Let's re-evaluate max_i.
    # x_i <= 10000. x_i = M * i.
    # i <= 10000 / 210 = 47.61.
    # So i in 1..47.
    # We need 12 distinct i's.
    
    # Precompute 7-smooth numbers up to 2 * 47 = 94.
    # Actually, sum of two i's is at most 47+46=93.
    # Let's just check if i+j is 7-smooth for all pairs.
    
    def is_7_smooth(n):
        if n <= 1: return True
        while n % 2 == 0: n //= 2
        while n % 3 == 0: n //= 3
        while n % 5 == 0: n //= 5
        while n % 7 == 0: n //= 7
        return n == 1
    
    # Build compatibility graph for i in 1..47
    max_val = 10000 // M
    nodes = list(range(1, max_val + 1))
    n_nodes = len(nodes)
    
    # adj[i][j] = True if nodes[i] + nodes[j] is 7-smooth
    # We want a clique of size 12.
    
    # Since n_nodes is small (47), we can use a simple backtracking.
    # To speed up, order nodes by "compatibility count" descending.
    
    compat_count = [0] * n_nodes
    adj = [[False] * n_nodes for _ in range(n_nodes)]
    
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            s = nodes[i] + nodes[j]
            if is_7_smooth(s):
                adj[i][j] = True
                adj[j][i] = True
                compat_count[i] += 1
                compat_count[j] += 1
    
    # Sort nodes by degree descending to help backtracking
    # We'll work with indices into the original nodes list, but reorder the search.
    # Let's create a list of (degree, node_index) and sort.
    order = sorted(range(n_nodes), key=lambda x: -compat_count[x])
    
    # Map new index to old index
    # Actually, it's easier to just permute the adjacency matrix or use a visited set.
    # Let's use a recursive backtracking that picks candidates in order of `order`.
    
    best_clique = None
    
    def backtrack(start_idx, current_clique):
        nonlocal best_clique
        if len(current_clique) == N:
            best_clique = current_clique[:]
            return True
        
        # Heuristic: pick the next node from `order` starting at `start_idx`
        # This is not strictly correct for finding ANY clique, but good for finding A clique.
        # A better approach: iterate through candidates that are compatible with all in current_clique.
        
        remaining_needed = N - len(current_clique)
        
        # Candidates are nodes in `order` that are not in current_clique
        # and are compatible with all nodes in current_clique.
        # To be efficient, we can maintain a list of available candidates.
        
        # For simplicity and given small N, we just iterate through the sorted order.
        # We must ensure we don't pick duplicates.
        
        # Let's use a standard Bron-Kerbosch variant or just simple recursion.
        # Given the small size, simple recursion with pruning is fine.
        
        # Pruning: if the number of remaining candidates compatible with all current
        # is less than remaining_needed, return False.
        
        # We'll iterate through candidates in the order defined by `order`.
        # We need to track which indices we've considered to avoid re-checking.
        
        # Let's just try all subsets of size N? No, 47 choose 12 is large.
        # But the graph is likely dense enough that a greedy approach works.
        
        # Greedy approach:
        # Start with the node with highest degree.
        # Add the next node with highest degree that is compatible.
        # Repeat.
        
        # Let's try a greedy construction first.
        # If it fails, we can do a more exhaustive search.
        
        # Actually, let's just do a DFS with the sorted order.
        # `start_idx` in the `order` list? No, that prevents finding cliques that don't follow the order.
        
        # Let's use a set of available candidates.
        
        # Initialize available candidates as all nodes.
        # At each step, filter available candidates to those compatible with the last added node?
        # No, compatible with ALL nodes in current_clique.
        
        # Let's rewrite backtrack to take a set of available candidates.
        
        pass

    # Let's implement a cleaner backtracking with candidate set.
    
    # Pre-compute adjacency bitsets for speed? Not needed for 47 nodes.
    
    def find_clique():
        # Start with all nodes as candidates
        # We want to find a clique of size N.
        
        # Sort nodes by degree to guide search
        # We'll maintain a list of candidate indices.
        
        # To ensure we find a clique, we can use a recursive function that picks one node,
        # and then restricts candidates to its neighbors.
        
        # Let's try starting with each node in `order`.
        
        for start_node in order:
            if time.time() > deadline - 5:
                break
            clique = [start_node]
            # Candidates must be neighbors of start_node
            candidates = [n for n in order if n != start_node and adj[start_node][n]]
            
            if len(candidates) < N - 1:
                continue
                
            def dfs(clique, candidates):
                if len(clique) == N:
                    return clique[:]
                
                # Pruning
                if len(candidates) < N - len(clique):
                    return None
                
                # Try each candidate
                # We can iterate in order of degree (already sorted in `candidates` if we preserve order)
                # `candidates` is derived from `order` so it is sorted by degree.
                
                for i in range(len(candidates)):
                    node = candidates[i]
                    new_clique = clique + [node]
                    
                    # New candidates: neighbors of `node` that are also in `candidates`
                    # and have index > i in `candidates` to avoid duplicates?
                    # No, we just need them to be compatible with all in `new_clique`.
                    # Since `candidates` are all compatible with `clique`, we just need them
                    # to be compatible with `node`.
                    
                    # Also, to avoid permutations, we can require that new candidates
                    # come from the subset of `candidates` that appear after `node` in the `order`.
                    # This ensures we generate each clique only once (in sorted order).
                    
                    new_candidates = []
                    for j in range(i + 1, len(candidates)):
                        cand = candidates[j]
                        if adj[node][cand]:
                            new_candidates.append(cand)
                    
                    if len(new_candidates) >= N - len(new_clique):
                        res = dfs(new_clique, new_candidates)
                        if res is not None:
                            return res
                return None
            
            res = dfs(clique, candidates)
            if res is not None:
                # Convert indices back to values
                return [nodes[i] for i in res]
        
        return None

    result_indices = find_clique()
    
    if result_indices is not None:
        result = [M * x for x in result_indices]
        # Verify
        assert len(set(result)) == N
        assert all(1 <= x <= 10000 for x in result)
        
        # Verify prime factors
        def _pf(n):
            if n <= 1: return set()
            s = set()
            while n % 2 == 0: s.add(2); n //= 2
            while n % 3 == 0: s.add(3); n //= 3
            f = 5
            while f * f <= n:
                while n % f == 0: s.add(f); n //= f
                f += 2
            if n > 1: s.add(n)
            return s
        
        primes = set()
        for i in range(N):
            for j in range(i+1, N):
                primes |= _pf(result[i] + result[j])
        
        # If primes are subset of {2,3,5,7}, score is <= 4.
        # This is the optimal possible if we can't do 3 or 2.
        # 3-smooth sums would require primes {2,3,5}.
        # Let's check if we can find a clique with 5-smooth sums (primes 2,3,5).
        # If so, score is 3.
        
        # Let's try to find a clique with 5-smooth sums.
        # is_5_smooth(n): divisible only by 2,3,5.
        
        def is_5_smooth(n):
            if n <= 1: return True
            while n % 2 == 0: n //= 2
            while n % 3 == 0: n //= 3
            while n % 5 == 0: n //= 5
            return n == 1
            
        # Re-run search with 5-smooth constraint
        # We can just check the found clique. If it's 5-smooth, great.
        # But we might have missed a 5-smooth clique if the 7-smooth one was found first?
        # No, 5-smooth is a subset of 7-smooth.
        # So if a 5-smooth clique exists, it is also a 7-smooth clique.
        # Our search finds A 7-smooth clique. It might not be 5-smooth.
        # To guarantee minimum primes, we should search for 5-smooth first.
        
        # Let's check if the found clique is 5-smooth.
        is_5_smooth_clique = True
        for i in range(len(result_indices)):
            for j in range(i+1, len(result_indices)):
                if not is_5_smooth(result_indices[i] + result_indices[j]):
                    is_5_smooth_clique = False
                    break
            if not is_5_smooth_clique:
                break
        
        if is_5_smooth_clique:
            return result
            
        # If not, try to find a 5-smooth clique specifically.
        # We can modify the search to use is_5_smooth.
        
        # Let's just re-run the graph construction for 5-smooth.
        
        adj5 = [[False] * n_nodes for _ in range(n_nodes)]
        count5 = [0] * n_nodes
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                s = nodes[i] + nodes[j]
                if is_5_smooth(s):
                    adj5[i][j] = True
                    adj5[j][i] = True
                    count5[i] += 1
                    count5[j] += 1
                    
        order5 = sorted(range(n_nodes), key=lambda x: -count5[x])
        
        def find_clique_5():
            for start_node in order5:
                if time.time() > deadline - 5:
                    break
                clique = [start_node]
                candidates = [n for n in order5 if n != start_node and adj5[start_node][n]]
                
                if len(candidates) < N - 1:
                    continue
                    
                def dfs5(clique, candidates):
                    if len(clique) == N:
                        return clique[:]
                    if len(candidates) < N - len(clique):
                        return None
                    for i in range(len(candidates)):
                        node = candidates[i]
                        new_clique = clique + [node]
                        new_candidates = []
                        for j in range(i + 1, len(candidates)):
                            cand = candidates[j]
                            if adj5[node][cand]:
                                new_candidates.append(cand)
                        if len(new_candidates) >= N - len(new_clique):
                            res = dfs5(new_clique, new_candidates)
                            if res is not None:
                                return res
                    return None
                
                res = dfs5(clique, candidates)
                if res is not None:
                    return [nodes[i] for i in res]
            return None
            
        result_indices_5 = find_clique_5()
        if result_indices_5 is not None:
            result5 = [M * x for x in result_indices_5]
            assert len(set(result5)) == N
            assert all(1 <= x <= 10000 for x in result5)
            return result5

    # Fallback if no 7-smooth clique found (unlikely)
    # Use greedy on full range with allowed primes {2,3,5,7}
    allowed = {2, 3, 5, 7}
    
    def _has_only_primes(n, allowed):
        if n <= 1:
            return True
        for p in allowed:
            while n % p == 0:
                n //= p
        return n == 1

    best_result = None
    best_score = float('inf')
    
    # Try to build a set of 12 numbers
    # Start with small numbers
    for start in range(1, 20):
        if time.time() > deadline - 2:
            break
        current = [start]
        for c in range(start + 1, 10001):
            if time.time() > deadline - 2:
                break
            if c in current:
                continue
            ok = True
            for d in current:
                if not _has_only_primes(c + d, allowed):
                    ok = False
                    break
            if ok:
                current.append(c)
                if len(current) >= N:
                    break
        
        if len(current) >= N:
            # Calculate score
            def _pf(n):
                if n <= 1: return set()
                s = set()
                while n % 2 == 0: s.add(2); n //= 2
                while n % 3 == 0: s.add(3); n //= 3
                f = 5
                while f * f <= n:
                    while n % f == 0: s.add(f); n //= f
                    f += 2
                if n > 1: s.add(n)
                return s
                
            primes = set()
            for i in range(N):
                for j in range(i+1, N):
                    primes |= _pf(current[i] + current[j])
            sc = len(primes)
            if sc < best_score:
                best_score = sc
                best_result = current[:]
            if best_score <= 4:
                break
                
    if best_result:
        return best_result
    
    # Final fallback
    return [210 * i for i in range(1, N + 1)]

# EVOLVE-BLOCK-END
