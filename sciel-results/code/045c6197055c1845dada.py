import math
import time

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    max_val = 1000000
    
    # Calculate LCM of divisors to limit search space for L
    M = 1
    for d in divisors:
        M = math.lcm(M, d)
        
    # Precompute multiples for each divisor? No, depends on L.
    # But we can precompute the residues?
    
    def find_matching(W, L):
        # Returns a list of 10 distinct integers in [L, L+W] 
        # such that res[i] is divisible by divisors[i]
        # or None if no such assignment exists.
        
        # Find all candidate values in [L, L+W] that are multiples of any divisor
        candidates = set()
        for d in divisors:
            # First multiple >= L
            k = (L + d - 1) // d
            v = k * d
            while v <= L + W:
                candidates.add(v)
                v += d
                
        if len(candidates) < n:
            return None
            
        # Map value to index in candidate list for efficiency
        # Actually, we just need to build the bipartite graph.
        # Left side: indices 0..n-1
        # Right side: candidates
        
        # For each i, find which candidates are divisible by divisors[i]
        graph = [[] for _ in range(n)]
        for i, d in enumerate(divisors):
            for v in candidates:
                if v % d == 0:
                    graph[i].append(v)
            
            # If any i has no candidates, matching is impossible
            if not graph[i]:
                return None

        # Bipartite Matching using DFS (Hungarian-like augmenting path)
        # match_to_val[i] = value assigned to i, or -1 if unassigned
        # match_from_val[v] = index assigned to v, or -1 if unassigned
        # Since candidates are dynamic, we map values to indices 0..len(candidates)-1
        val_list = list(candidates)
        val_to_idx = {v: idx for idx, v in enumerate(val_list)}
        
        # Rebuild graph with indices
        graph_idx = [[] for _ in range(n)]
        for i in range(n):
            for v in graph[i]:
                graph_idx[i].append(val_to_idx[v])
        
        match_i = [-1] * n  # match_i[i] = index in val_list
        match_v = [-1] * len(val_list) # match_v[j] = i
        
        def dfs(i, seen):
            for j in graph_idx[i]:
                if seen[j]:
                    continue
                seen[j] = True
                if match_v[j] == -1 or dfs(match_v[j], seen):
                    match_v[j] = i
                    match_i[i] = j
                    return True
            return False
        
        for i in range(n):
            seen = [False] * len(val_list)
            if not dfs(i, seen):
                return None
                
        # Construct result
        res = [val_list[match_i[i]] for i in range(n)]
        return res

    # Binary search for minimum W
    # Lower bound: 9 (10 distinct integers)
    # Upper bound: Let's start with a reasonable guess, e.g., 1000
    low = 9
    high = 1000
    
    # Ensure high is feasible. If not, expand.
    while True:
        feasible = False
        for L in range(M):
            res = find_matching(high, L)
            if res is not None:
                feasible = True
                # Check bounds
                if all(1 <= x <= max_val for x in res):
                    break
        if feasible:
            break
        low = high
        high *= 2
        if high > max_val:
             # Fallback if something goes wrong, though LCM periodicity helps
             # Just return divisors if we can't find a better one
             return divisors[:]

    best_res = None
    while low <= high:
        mid = (low + high) // 2
        found = False
        for L in range(M):
            res = find_matching(mid, L)
            if res is not None:
                # Check bounds 1..1000000
                if all(1 <= x <= max_val for x in res):
                    found = True
                    best_res = res
                    break
        if found:
            high = mid - 1
        else:
            low = mid + 1
            
    return best_res

# EVOLVE-BLOCK-END
