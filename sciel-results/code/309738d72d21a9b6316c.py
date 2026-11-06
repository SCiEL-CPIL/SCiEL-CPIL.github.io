import time

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    n = 30
    L = 1
    for i in range(1, n + 1):
        L = L * i // __import__('math').gcd(L, i)
        
    w = [L // i for i in range(1, n + 1)]
    
    mid = n // 2
    w1 = w[:mid]
    w2 = w[mid:]
    
    def generate_sums(weights):
        sums = [0]
        for wi in weights:
            new_sums = []
            for s in sums:
                new_sums.append(s - wi)
                new_sums.append(s)
                new_sums.append(s + wi)
            sums = new_sums
        return sums

    def generate_coeffs_for_sum(weights, target_sum):
        # Recursively or iteratively find coeffs
        # Since 3^15 is large, we can use a set of reachable sums at each step?
        # No, we can just DFS with pruning?
        # Or simple recursion with memoization?
        # Actually, we can just re-run the generation and store the path for the first occurrence.
        # But that requires storing paths for all sums, which is memory heavy.
        
        # Better: DFS with backtracking.
        # We want to find c_i such that sum c_i w_i = target_sum.
        # We can do this by iterating through all 3^15 possibilities?
        # 14 million iterations is fast in Python if optimized?
        # Maybe.
        
        # Let's try a simple recursive search with memoization of (index, remaining_sum) -> bool?
        # No, we need the coeffs.
        
        # Let's just iterate through all 3^15 combos?
        # 3^15 = 14,348,907.
        # This is a one-time cost.
        
        import itertools
        # Generate all combos
        # This is slow in Python.
        
        # Better: Re-run the DP set approach for this half, but store the parent pointer?
        # We can store a dict: sum -> (prev_sum, c)
        # Size of dict is number of distinct sums in this half.
        # For 15 terms, distinct sums are much fewer than 3^15.
        
        parent = {0: (None, None)}
        curr = {0}
        for i, wi in enumerate(weights):
            next_parent = {}
            next_curr = set()
            for s in curr:
                for c in [-1, 0, 1]:
                    ns = s + c * wi
                    if ns not in next_parent:
                        next_parent[ns] = (s, c)
                        next_curr.add(ns)
            curr = next_curr
            # We need to keep parent pointers for backtracking
            # But we only have one level. We need the full history.
            
        # This is getting complicated.
        # Let's stick to the full history DP for the whole problem if memory allows.
        # Or, for the half, we can just store the history of sets.
        # 15 steps. Sets are small.
        
        history = [set([0])]
        curr = {0}
        for i, wi in enumerate(weights):
            next_set = set()
            for s in curr:
                next_set.add(s - wi)
                next_set.add(s)
                next_set.add(s + wi)
            curr = next_set
            history.append(curr)
            
        # Backtrack
        coeffs = [0] * len(weights)
        current_sum = target_sum
        for i in range(len(weights) - 1, -1, -1):
            wi = weights[i]
            if current_sum - wi in history[i]:
                coeffs[i] = 1
                current_sum -= wi
            elif current_sum in history[i]:
                coeffs[i] = 0
            elif current_sum + wi in history[i]:
                coeffs[i] = -1
                current_sum += wi
            else:
                coeffs[i] = 0
        return coeffs

    # Generate sums
    sums1 = generate_sums(w1)
    sums2 = generate_sums(w2)
    
    if time.time() > deadline:
        return [0] * 28 + [1, -1]
        
    sums1.sort()
    
    best_val = float('inf')
    best_s1 = 0
    best_s2 = 0
    
    for s2 in sums2:
        # Binary search for -s2 in sums1
        import bisect
        idx = bisect.bisect_left(sums1, -s2)
        
        # Check idx and idx-1
        candidates = []
        if idx < len(sums1):
            candidates.append(sums1[idx])
        if idx > 0:
            candidates.append(sums1[idx - 1])
            
        for s1 in candidates:
            total = s1 + s2
            if total != 0:
                if abs(total) < best_val:
                    best_val = abs(total)
                    best_s1 = s1
                    best_s2 = s2
                    
    if time.time() > deadline:
        return [0] * 28 + [1, -1]
        
    if best_s1 == 0 and best_s2 == 0:
        # Fallback
        return [0] * 28 + [1, -1]
        
    coeffs1 = generate_coeffs_for_sum(w1, best_s1)
    coeffs2 = generate_coeffs_for_sum(w2, best_s2)
    
    return coeffs1 + coeffs2
