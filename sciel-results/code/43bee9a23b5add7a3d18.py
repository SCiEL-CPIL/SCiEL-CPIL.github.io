import time
from math import gcd

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    start_time = time.time()
    deadline = start_time + 85  # safety margin
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    period = 1
    for d in divisors:
        period = lcm(period, d)
    # period = 30030

    # Bipartite matching using DFS (Kuhn's algorithm)
    # Left side: divisors (0..n-1)
    # Right side: candidate values
    # We want to assign distinct values to divisors.
    
    def find_matching(candidates):
        # candidates[i] is a list of possible values for divisor i
        # We map values to indices on the right side
        # Since values can be large, we use a dict or set for matching
        # But for Kuhn's, we usually use an array matchR[j] = i
        # Here j is the value itself. Since values are <= 1000000, we can use a dict for matchR.
        
        matchR = {}  # value -> divisor index
        
        def dfs(u, visited):
            for v in candidates[u]:
                if v in visited:
                    continue
                visited.add(v)
                if v not in matchR or dfs(matchR[v], visited):
                    matchR[v] = u
                    return True
            return False
        
        for u in range(n):
            visited = set()
            if not dfs(u, visited):
                return None
        
        # Reconstruct assignment
        assignment = [None] * n
        for v, u in matchR.items():
            assignment[u] = v
        return assignment

    # Iterate span S
    # The span is max - min.
    # If we fix a window [L, L+S], and find a matching, the actual min and max of the assigned values
    # might be strictly inside [L, L+S].
    # However, if we find a matching in [L, L+S], the span of the result is <= S.
    # We want to minimize the actual span.
    # If we iterate S from 1 upwards, the first S for which a matching exists in *some* window [L, L+S]
    # gives a solution with span <= S.
    # Is it guaranteed to be exactly S? No.
    # But if a solution with span S' < S exists, it would have been found in a previous iteration
    # (specifically, if the values are in [L', L'+S'], then they are also in [L', L'+S] for S > S').
    # Wait, if we find a matching in [L, L+S], the values used are a subset of [L, L+S].
    # The span of this subset is <= S.
    # If we iterate S=1, 2, ..., the first S where *any* window has a matching will yield a solution
    # with span <= S.
    # Could there be a solution with span S' < S that was missed?
    # If a solution with span S' exists, it fits in a window of length S'.
    # So it would have been found when we iterated S=S'.
    # Thus, the first S that yields a valid matching gives the minimum possible span.
    
    # We need to be careful: The "window" [L, L+S] must contain the values.
    # The values are distinct.
    
    max_S = 200 # Heuristic upper bound, can increase if needed
    
    for S in range(1, max_S + 1):
        if time.time() > deadline:
            break
            
        for L in range(1, period + 1): # L must be >= 1 for positive integers
            if time.time() > deadline:
                break
                
            candidates = []
            feasible = True
            for d in divisors:
                k_min = (L + d - 1) // d
                k_max = (L + S) // d
                if k_max < k_min:
                    feasible = False
                    break
                # Generate multiples
                # k_min * d >= L
                # k_max * d <= L + S
                # Ensure k_min * d >= 1 (guaranteed if L>=1 and d>=1)
                opts = list(range(k_min * d, k_max * d + 1, d))
                candidates.append(opts)
            
            if not feasible:
                continue
                
            # Check if any candidate list is empty
            if any(len(c) == 0 for c in candidates):
                continue
                
            # Solve matching
            # Optimization: Sort divisors by number of options (MRV) before matching?
            # Kuhn's algorithm works on the order. We can reorder rows.
            # Let's just run it. n=10 is small.
            
            # We need to modify find_matching to handle the specific structure or just use it as is.
            # The current find_matching uses a global matchR.
            # It's better to make it local or pass state.
            
            matchR = {}
            
            def dfs(u, visited):
                for v in candidates[u]:
                    if v in visited:
                        continue
                    visited.add(v)
                    if v not in matchR or dfs(matchR[v], visited):
                        matchR[v] = u
                        return True
                return False
            
            success = True
            for u in range(n):
                visited = set()
                if not dfs(u, visited):
                    success = False
                    break
            
            if success:
                # Reconstruct
                assignment = [None] * n
                for v, u in matchR.items():
                    assignment[u] = v
                
                # Verify distinctness and bounds
                # The matching ensures distinct values.
                # Values are in [L, L+S] so max-min <= S.
                # We can return this.
                # To be safe, we can check if the actual span is smaller?
                # No, we iterate S. If this is the first S, the span is <= S.
                # If the actual span is < S, it would have been found earlier?
                # Yes, because if values are in [L, L+S] and span is S' < S,
                # they are contained in [min_val, max_val] which has length S'.
                # This window [min_val, max_val] would have been checked when S_iter = S'.
                # So the first success is the optimal span.
                
                return [int(x) for x in assignment]
                
    # Fallback if no solution found in max_S
    # This should not happen for small S
    return divisors[:]
