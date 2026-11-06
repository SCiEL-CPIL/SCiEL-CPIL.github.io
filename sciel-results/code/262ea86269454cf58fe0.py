    # EVOLVE-BLOCK-START
    def construct():
        """
        Find a set of distinct pairwise coprime integers a in 1..511 that maximizes
        the sum of 1/(512 - a).
        
        Approach: Exact Maximum Weight Independent Set using Branch-and-Bound
        with bitset-based pruning and precomputed weight sums for efficiency.
        """
        from math import gcd
        import sys
        
        N = 512
        vertices = list(range(1, N))
        
        # 1. Sieve for primes
        is_prime = [True] * N
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(N**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, N, i):
                    is_prime[j] = False
        
        primes = [p for p in range(2, N) if is_prime[p]]
        
        # 2. Factorize numbers to build conflict graph
        # conflict[i] will be a bitmask of all j such that gcd(i,j) > 1
        conflict = [0] * N
        for i in range(1, N):
            mask = 0
            for j in range(1, N):
                if i != j and gcd(i, j) > 1:
                    mask |= (1 << j)
            conflict[i] = mask
            
        # 3. Weights
        weights = [0.0] * N
        for a in range(1, N):
            weights[a] = 1.0 / (N - a)
            
        # 4. Precompute weight sums for 8-bit chunks to speed up bound calculation
        byte_count = (N + 7) // 8
        weight_sum_table = [[0.0] * 256 for _ in range(byte_count)]
        for b in range(byte_count):
            for m in range(256):
                s = 0.0
                for k in range(8):
                    if m & (1 << k):
                        idx = b * 8 + k
                        if idx < N:
                            s += weights[idx]
                weight_sum_table[b][m] = s
                
        def sum_weights(mask):
            s = 0.0
            for b in range(byte_count):
                m = (mask >> (b * 8)) & 0xFF
                s += weight_sum_table[b][m]
            return s

        # 5. Greedy Initial Solution for better pruning
        # Sort by weight descending
        order = sorted(vertices, key=lambda x: weights[x], reverse=True)
        
        greedy_mask = 0
        greedy_score = 0.0
        greedy_set = []
        for v in order:
            if not (greedy_mask & (1 << v)):
                greedy_mask |= (1 << v)
                greedy_score += weights[v]
                greedy_set.append(v)
        
        best_score = greedy_score
        best_set = greedy_set[:]
        
        # 6. Branch and Bound
        sys.setrecursionlimit(10000)
        
        def dfs(available_mask, current_score, remaining_sum, chosen_mask):
            nonlocal best_score, best_set
            
            if available_mask == 0:
                if current_score > best_score:
                    best_score = current_score
                    best_set = [i for i in range(1, N) if chosen_mask & (1 << i)]
                return
            
            # Upper Bound Pruning
            if current_score + remaining_sum <= best_score:
                return
                
            # Select vertex with highest weight in available_mask
            v = -1
            for u in order:
                if available_mask & (1 << u):
                    v = u
                    break
            
            if v == -1:
                return
                
            # Branch 1: Include v
            neighbors = conflict[v]
            to_remove = available_mask & (neighbors | (1 << v))
            
            removed_weight = sum_weights(to_remove)
            new_available = available_mask & ~to_remove
            new_remaining = remaining_sum - removed_weight
            new_chosen = chosen_mask | (1 << v)
            
            dfs(new_available, current_score + weights[v], new_remaining, new_chosen)
            
            # Branch 2: Exclude v
            to_remove_ex = available_mask & (1 << v)
            removed_weight_ex = weights[v]
            
            new_available_ex = available_mask & ~(1 << v)
            new_remaining_ex = remaining_sum - removed_weight_ex
            
            dfs(new_available_ex, current_score, new_remaining_ex, chosen_mask)
            
        # Initial call
        initial_mask = 0
        for i in range(1, N):
            initial_mask |= (1 << i)
            
        initial_remaining = sum_weights(initial_mask)
        
        dfs(initial_mask, 0.0, initial_remaining, 0)
        
        return best_set
    # EVOLVE-BLOCK-END
    