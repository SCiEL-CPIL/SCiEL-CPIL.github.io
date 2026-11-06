"""Program using modular arithmetic sieve + greedy construction with fast mean-check."""

def construct():
    n = 512
    max_size = 48
    
    # Step 1: Modular arithmetic sieve to pre-filter candidates.
    # We want to prioritize x such that it is "harder" for x to be the mean of other elements.
    # The most common violation is a 3-term arithmetic progression: x = (y+z)/2 => 2x = y+z.
    # We score each candidate x by counting how many pairs (y, z) in [1, 512] satisfy
    # y + z = 2x (exact). Candidates with fewer such pairs are less likely to be part of a mean-violating subset.
    # Additionally, we can use a modular heuristic to further refine the ordering.
    
    candidates = list(range(1, n + 1))
    
    # Precompute exact 3-AP counts
    # For a fixed x, y must be in [1, 512] and z = 2x - y must be in [1, 512] and z != y, z != x.
    # y ranges from max(1, 2x - 512) to min(512, 2x - 1).
    
    scores = {x: 0 for x in candidates}
    
    for x in candidates:
        count = 0
        low = max(1, 2*x - n)
        high = min(n, 2*x - 1)
        for y in range(low, high + 1):
            z = 2*x - y
            if z != y and z != x and 1 <= z <= n:
                count += 1
        scores[x] = count
        
    # Sort candidates by score (ascending). Low score = fewer exact 3-APs = better candidate.
    candidates.sort(key=lambda x: scores[x])
    
    # Step 2: Greedy construction with bitset DP validation
    
    def is_valid_mean_check(selected, candidate):
        """Check if adding candidate to selected keeps the mean-free property."""
        new_set = selected + [candidate]
        n_elems = len(new_set)
        
        # For each element a in new_set, check if a is the mean of any subset
        # of at least 2 OTHER elements.
        for i, a in enumerate(new_set):
            others = [x for j, x in enumerate(new_set) if j != i]
            m = len(others)
            
            # We need to check: does there exist a subset T of others with |T| >= 2
            # such that sum(T) = |T| * a?
            
            others_set = set(others)
            found = False
            
            # Check k=2
            for b in others:
                if b != a and (2*a - b) in others_set and (2*a - b) != b:
                    found = True
                    break
            if found:
                return False
            
            # For k >= 3: use DP
            # dp[k] is a bitset where bit s is set if sum s is achievable with k elements
            # Initialize: dp[0] has bit 0 set
            dp = [0] * (m + 1)
            dp[0] = 1  # bit 0 set
            
            for x in others:
                # Process in reverse to avoid using x multiple times
                for k in range(m, 0, -1):
                    # dp[k] |= dp[k-1] << x
                    dp[k] |= (dp[k-1] << x)
            
            # Now check for each k >= 3 if bit (k*a) is set in dp[k]
            for k in range(3, m + 1):
                target_sum = k * a
                if target_sum <= sum(others) and (dp[k] >> target_sum) & 1:
                    return False
        
        return True
    
    # Greedy: try candidates in the sorted order
    selected = []
    for x in candidates:
        if len(selected) >= max_size:
            break
        if is_valid_mean_check(selected, x):
            selected.append(x)
    
    return selected
