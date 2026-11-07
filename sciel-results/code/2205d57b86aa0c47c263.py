"""Construct maximum-size set with cardinality-separated subset sums."""
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    max_val = 256
    max_size = 60
    
    # Greedy construction with bitset DP
    # dp[k] = set of achievable sums with exactly k elements
    # We use Python ints as bitsets for efficiency
    
    def try_add(elements, x, dp, current_size):
        """Try to add x to elements, check if cardinality separation holds."""
        if x in elements:
            return None
        
        new_elements = elements + [x]
        new_dp = [dp[k] for k in range(len(dp))]
        # Extend dp if needed
        while len(new_dp) <= current_size + 1:
            new_dp.append(0)
        
        # Update: for k from current_size down to 0
        for k in range(current_size, -1, -1):
            new_dp[k+1] |= (dp[k] << x)
        
        # Check cardinality separation: for each pair (i,j) with i≠j, 
        # dp[i] & dp[j] must be 0 (no common sums)
        n_k = len(new_dp)
        for i in range(n_k):
            for j in range(i+1, n_k):
                if new_dp[i] & new_dp[j]:
                    return None
        
        return new_elements, new_dp
    
    # Start with empty set
    elements = []
    dp = [1]  # dp[0] = bit 0 set (empty subset sum = 0)
    
    # Greedy: try to add numbers, prefer smaller ones to leave room
    # But we need to be strategic. Let's try multiple strategies.
    
    # Strategy 1: increasing order
    best_elements = []
    best_dp = [1]
    
    for x in range(1, max_val + 1):
        if time.time() > deadline:
            break
        result = try_add(elements, x, dp, len(elements))
        if result:
            elements, dp = result
            best_elements = list(elements)
            best_dp = dp
            if len(elements) >= max_size:
                break
    
    # Try to improve: remove some elements and try different combinations
    # Use a more systematic approach: try all subsets of a candidate pool
    
    # Let's try a different approach: use exponential spacing
    # a_i = 2^(i-1) * base, but capped at 256
    
    # Actually, let's try a smarter greedy with backtracking for small sets
    
    # Try building with larger gaps
    elements2 = []
    dp2 = [1]
    
    # Use values that grow fast enough
    # For cardinality k, max sum < min sum for k+1
    # If we pick a_i such that a_i > sum of all previous, it works
    # But that gives at most log2(256) ≈ 8 elements
    
    # Better: we don't need superincreasing, just separation
    # Let's try: pick elements, then verify
    
    # Multi-start greedy with different orderings
    import random
    random.seed(42)
    
    candidates = list(range(1, max_val + 1))
    
    # Try increasing
    for ordering_name, ordering in [
        ("inc", list(range(1, max_val + 1))),
        ("dec", list(range(max_val, 0, -1))),
        ("sparse", [1, 3, 7, 15, 31, 63, 127, 255, 256] + list(range(2, 128))),
    ]:
        if time.time() > deadline:
            break
        
        elems = []
        dpp = [1]
        
        for x in ordering:
            if time.time() > deadline:
                break
            if len(elems) >= max_size:
                break
            result = try_add(elems, x, dpp, len(elems))
            if result:
                elems, dpp = result
        
        if len(elems) > len(best_elements):
            best_elements = list(elems)
            best_dp = dpp
    
    # Now try to extend best_elements by trying to add more elements
    # with a more careful search
    if len(best_elements) < max_size and time.time() < deadline:
        # Try adding elements not in best_elements
        in_set = set(best_elements)
        for x in range(1, max_val + 1):
            if time.time() > deadline:
                break
            if x in in_set:
                continue
            result = try_add(best_elements, x, best_dp, len(best_elements))
            if result:
                best_elements, best_dp = result
                in_set.add(x)
                if len(best_elements) >= max_size:
                    break
    
    # Final verification
    if not best_elements:
        return [1]
    
    # Verify: check all subset sums by cardinality
    n = len(best_elements)
    # dp[k] = bitset of sums achievable with exactly k elements
    verify_dp = [0] * (n + 1)
    verify_dp[0] = 1
    for x in best_elements:
        for k in range(n, 0, -1):
            verify_dp[k] |= (verify_dp[k-1] << x)
    
    # Check disjointness
    for i in range(n + 1):
        for j in range(i + 1, n + 1):
            if verify_dp[i] & verify_dp[j]:
                # Invalid! Fall back to a known-valid small set
                return [1, 3, 7]
    
    # Ensure all are Python ints
    return [int(x) for x in best_elements]
