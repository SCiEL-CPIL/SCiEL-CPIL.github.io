import time
import random
import numpy as np

def _find_minimal_conflict(S, target=200):
    """
    Finds the subset of S with the minimum number of elements that sums to target.
    Returns the subset (list) or None.
    """
    if not S:
        return None
    
    # dp[s] = min number of elements to reach sum s
    # -1 means unreachable
    dp = [-1] * (target + 1)
    dp[0] = 0
    
    # parent[s] = (prev_sum, element)
    parent = [None] * (target + 1)
    
    # We iterate through elements. 
    # To find the MINIMAL subset, we should be careful with order?
    # Actually, standard 0/1 knapsack DP for min items works regardless of order 
    # if we update correctly.
    # However, standard iteration `for x in S: for s in target..x: ...` 
    # finds a solution. Does it find the MINIMUM cardinality?
    # Yes, if we take the min.
    
    # But we must ensure we don't use the same element twice.
    # The standard 1D DP array `dp` updated in reverse `s` from `target` down to `x`
    # ensures each element is used at most once.
    
    # We need to store the path.
    # Since we are minimizing count, `dp[s]` holds the count.
    # `parent[s]` needs to hold the choice.
    # But `dp` is 1D, so `parent` must be overwritten carefully.
    # This approach (1D DP with parent) is tricky for reconstruction because 
    # `parent[s]` might be overwritten by a later element that achieves the same min count,
    # or a different path.
    
    # Alternative: 2D DP `dp[i][s]` is too slow/memory heavy for 100 elements?
    # 100 * 200 = 20,000 states. Very fast.
    # Let's use 2D DP for safety and clarity in reconstruction.
    
    n = len(S)
    # dp[i][s] = min elements from S[0..i-1] to sum to s
    # -1 if unreachable
    # We can optimize memory to 2 rows, but we need full path, so 2D or store parents in 1D carefully.
    # Given N=100, T=200, 2D is fine.
    
    # dp[i][s]
    # i from 0 to n
    # s from 0 to target
    
    # Initialize
    # dp[0][0] = 0, dp[0][s>0] = -1
    dp = [[-1] * (target + 1) for _ in range(n + 1)]
    dp[0][0] = 0
    
    # prev_choice[i][s] = 1 if S[i-1] used, 0 if not
    # We only need to store this for reconstruction.
    choice = [[0] * (target + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        x = S[i - 1]
        for s in range(target + 1):
            # Option 1: Don't use x
            if dp[i - 1][s] != -1:
                dp[i][s] = dp[i - 1][s]
                choice[i][s] = 0
            else:
                dp[i][s] = -1
                choice[i][s] = 0 # default
                
            # Option 2: Use x
            if x <= s:
                if dp[i - 1][s - x] != -1:
                    candidate_count = dp[i - 1][s - x] + 1
                    if dp[i][s] == -1 or candidate_count < dp[i][s]:
                        dp[i][s] = candidate_count
                        choice[i][s] = 1
                    elif candidate_count == dp[i][s]:
                        # Tie-break: prefer using x? Or not? 
                        # Doesn't matter for existence, but might matter for which element we pick to remove.
                        # We want a "good" conflict to remove from.
                        # Let's stick to first found or arbitrary.
                        pass
    
    if dp[n][target] == -1:
        return None
    
    # Reconstruct
    subset = []
    s = target
    for i in range(n, 0, -1):
        if choice[i][s] == 1:
            subset.append(S[i - 1])
            s -= S[i - 1]
    
    return subset

def construct():
    start_time = time.time()
    deadline = 50.0  # Safety margin
    
    universe = set(range(1, 101))
    best_set = None
    best_size = 0
    
    # Strategy 1: Greedy Set Cover (Blocking Set)
    # Find min B such that no subset of U\B sums to 200.
    B = set()
    
    # We will run the greedy loop.
    # To make it robust, we can try different tie-breaking heuristics for the element to remove.
    # Heuristic 1: Max element in minimal conflict.
    # Heuristic 2: Min element in minimal conflict.
    # Heuristic 3: Element in minimal conflict with highest "degree" (precomputed? no).
    
    # Let's try Heuristic 1 (Max) first.
    B_curr = set()
    while time.time() - start_time < deadline - 10:
        S_curr = list(universe - B_curr)
        conflict = _find_minimal_conflict(S_curr)
        if conflict is None:
            if len(S_curr) > best_size:
                best_size = len(S_curr)
                best_set = S_curr[:]
            break
        
        # Pick element to add to B
        # max(conflict)
        elem_to_add = max(conflict)
        B_curr.add(elem_to_add)
        
        # Check if we improved? No, we are building B.
        # We only check validity when B is "done" (no conflict).
        
    # Strategy 2: Polish the result from Strategy 1
    if best_set is not None:
        # Try to remove elements from B (add to S) to see if we can keep them.
        # We iterate through elements in B_curr (the one that produced best_set)
        # Actually, best_set corresponds to a specific B. Let's reconstruct B from best_set.
        B_best = universe - set(best_set)
        
        # Try to remove each element x from B_best (i.e., add to S)
        # If S remains valid, keep x out of B.
        # Order of removal: Try removing elements that are "least critical".
        # Large elements are usually more critical. Try removing small elements first?
        # Or just try all.
        
        S_polished = set(best_set)
        # Sort B_best by value, maybe ascending to try removing small ones first?
        # Or descending?
        # Let's try ascending (remove small blockers first, as they are more "replaceable" in sums?)
        for x in sorted(list(B_best)):
            # Tentatively add x to S
            S_test = S_polished | {x}
            # Check if S_test has a subset summing to 200
            # Fast check:
            # We can reuse _find_minimal_conflict, but it's slow.
            # Let's use a fast bitset check.
            
            bits = np.zeros(201, dtype=np.uint8)
            bits[0] = 1
            valid = True
            for val in S_test:
                if val > 200:
                    continue
                shifted = np.zeros(201, dtype=np.uint8)
                shifted[val:] = bits[:201 - val]
                bits = bits | shifted
                # Early exit? No, must finish all.
            
            if bits[200] == 0:
                # Valid, keep x in S
                S_polished = S_test
            else:
                # Invalid, x must stay in B (not in S)
                pass
        
        if len(S_polished) > best_size:
            best_size = len(S_polished)
            best_set = list(S_polished)

    # Strategy 3: Randomized restarts with Greedy Set Cover
    # Maybe max(conflict) is not always best.
    # Try random element from conflict?
    # Or element with max "impact"?
    
    while time.time() - start_time < deadline - 2:
        B_rand = set()
        while time.time() - start_time < deadline - 2:
            S_rand = list(universe - B_rand)
            conflict = _find_minimal_conflict(S_rand)
            if conflict is None:
                if len(S_rand) > best_size:
                    best_size = len(S_rand)
                    best_set = S_rand[:]
                break
            
            # Heuristic: Random choice from conflict?
            # Or Max?
            # Let's try Max again, but with random tie-breaking if multiple max? No, max is unique.
            # Let's try: Pick element that appears in the most "potential" sums?
            # Too complex.
            # Let's just try removing the LARGEST element in the conflict.
            elem_to_add = max(conflict)
            B_rand.add(elem_to_add)
            
            # Sometimes, removing a smaller element allows a larger set?
            # Let's try a different heuristic in subsequent iterations:
            # Remove the element with the highest value / (value + 1)? No.
            # Let's try: Remove the element that is part of the "densest" part of the sum.
            # For now, stick to Max.
            
            # To diversify, let's try removing the MIN element sometimes?
            if random.random() < 0.2:
                elem_to_add = min(conflict)
            
            B_rand.add(elem_to_add)

    return best_set
