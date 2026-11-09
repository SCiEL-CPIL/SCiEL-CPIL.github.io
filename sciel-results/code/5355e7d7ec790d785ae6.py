import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 150s limit, leave 35s for safety/overhead
    
    # Fallback: Andrasfai graph And(9) on 26 vertices
    # Triangle-free, alpha = 9
    best_result = {
        'n': 26,
        'edges': [[i, j] for i in range(26) for j in range(i + 1, 26) if (j - i) % 26 % 3 == 1]
    }
    
    # Try to find a larger graph
    # R(3, 10) <= 41, so max n is 40
    
    for n in range(40, 26, -1):
        if time.time() > deadline:
            break
        
        # Try multiple random restarts for this n
        restarts = 0
        max_restarts = 20000  # Increased restarts for structured generation
        
        while time.time() < deadline - 5 and restarts < max_restarts:
            restarts += 1
            
            # Generate a random sum-free set S in Z_n
            # S must be symmetric: if x in S, then -x in S.
            # S is sum-free: (S+S) \cap S = empty.
            
            # Possible differences are 1, ..., n//2
            # For each k, we can choose to include {k, n-k} or not.
            # We need to ensure sum-freeness.
            
            candidates = list(range(1, n // 2 + 1))
            random.shuffle(candidates)
            
            S = set()
            S_doubled = set() # 2*S elements, or rather, we check if x+y in S
            
            # We build S incrementally.
            # To be efficient, we can just pick a random subset and check, 
            # or build it greedily.
            
            # Greedy build of sum-free symmetric set
            for k in candidates:
                # Check if adding {k, -k} preserves sum-freeness
                # New elements in S: k, n-k
                # We need to ensure no sum of two elements in S is in S.
                # Specifically:
                # 1. k + k = 2k. If 2k in S, fail.
                # 2. k + (-k) = 0. 0 is not in S (usually).
                # 3. k + s for s in S. If (k+s) % n in S, fail.
                # 4. (-k) + s for s in S. If (-k+s) % n in S, fail.
                # 5. k + (-k) ... handled.
                
                # Efficient check:
                # Let S_old be the current set.
                # We add k and nk = n-k.
                # Check if (S_old + {k, nk}) intersects (S_old + {k, nk}).
                # Actually, simpler:
                # For each s in S_old:
                #   if (s + k) % n in S_old: fail
                #   if (s + nk) % n in S_old: fail
                #   if (s + k) % n == k: fail (implies s=0, not in S)
                #   if (s + nk) % n == nk: fail
                # Also check internal sums of new elements:
                #   (k + k) % n in S_old?
                #   (k + nk) % n = 0 in S_old?
                #   (nk + nk) % n in S_old?
                
                valid = True
                s_old_list = list(S)
                for s in s_old_list:
                    if ((s + k) % n) in S:
                        valid = False
                        break
                    if ((s + n - k) % n) in S:
                        valid = False
                        break
                
                if valid:
                    # Check sums involving the new elements themselves
                    if ((k + k) % n) in S or (k + k) % n == k or (k + k) % n == n - k:
                        valid = False
                    if ((n - k + n - k) % n) in S or (n - k + n - k) % n == k or (n - k + n - k) % n == n - k:
                        valid = False
                    # Sum of k and n-k is 0. 0 is not in S.
                    
                if valid:
                    S.add(k)
                    S.add(n - k)
            
            # If S is empty or too small, the graph is sparse, alpha will be large.
            # We want dense enough to keep alpha < 10.
            # For n=40, we need fairly large S.
            # Random greedy sum-free sets might be too small.
            # Let's try to force a larger set by picking from a known structure?
            # Or just try many random ones.
            
            if len(S) < n // 3: # Heuristic: need decent density
                continue

            # Construct adjacency list
            adj = [0] * n
            for i in range(n):
                for k in S:
                    j = (i + k) % n
                    if i < j:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Check if alpha < 10 using a more efficient branch and bound
            all_mask = (1 << n) - 1
            non_adj = [0] * n
            for i in range(n):
                non_adj[i] = (all_mask ^ (1 << i)) & ~adj[i]
            
            # Quick greedy check: try to build a large independent set greedily
            def greedy_independent_set():
                remaining = all_mask
                count = 0
                while remaining:
                    # Pick vertex with minimum degree in remaining
                    min_deg = n + 1
                    best_v = -1
                    temp = remaining
                    while temp:
                        lsb = temp & -temp
                        v = lsb.bit_length() - 1
                        deg = (adj[v] & remaining).bit_count()
                        if deg < min_deg:
                            min_deg = deg
                            best_v = v
                        temp ^= lsb
                    count += 1
                    remaining &= non_adj[best_v]
                    if count >= 10:
                        return count
                return count
            
            if greedy_independent_set() >= 10:
                continue  # This graph definitely has alpha >= 10
            
            # Now do exact check with branch and bound
            def has_independent_set_10():
                def dfs(mask, size):
                    if size == 10:
                        return True
                    if mask == 0:
                        return False
                    
                    # Pruning: if remaining vertices + current size < 10, fail
                    if mask.bit_count() + size < 10:
                        return False
                    
                    # Pick vertex with maximum degree in induced subgraph
                    max_deg = -1
                    best_v = -1
                    temp = mask
                    while temp:
                        lsb = temp & -temp
                        v = lsb.bit_length() - 1
                        deg = (adj[v] & mask).bit_count()
                        if deg > max_deg:
                            max_deg = deg
                            best_v = v
                        temp ^= lsb
                    
                    if best_v == -1:
                        return False
                    
                    v_bit = 1 << best_v
                    
                    # Try including best_v
                    include_mask = non_adj[best_v] & mask
                    if dfs(include_mask, size + 1):
                        return True
                    
                    # Try excluding best_v
                    exclude_mask = mask & ~v_bit
                    if dfs(exclude_mask, size):
                        return True
                    
                    return False
                
                return dfs(all_mask, 0)
            
            if not has_independent_set_10():
                # Valid graph found
                edges = []
                for i in range(n):
                    mask = adj[i]
                    while mask:
                        lsb = mask & -mask
                        j = lsb.bit_length() - 1
                        if i < j:
                            edges.append([i, j])
                        mask ^= lsb
                return {'n': n, 'edges': edges}
        
    return best_result
