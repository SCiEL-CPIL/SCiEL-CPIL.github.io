"""Program using modular arithmetic sieve + greedy construction with fast mean-check."""

def construct():
    n = 512
    max_size = 48
    
    # Step 1: Modular arithmetic sieve to pre-filter candidates
    # We want to keep x such that it's "harder" for x to be the mean of other elements.
    # Key insight: if x is the mean of k other elements y_1..y_k, then sum(y_i) = k*x.
    # This implies sum(y_i) ≡ 0 (mod something related to x).
    # 
    # A simpler pre-filter: prioritize numbers that are "spread out" and avoid
    # obvious arithmetic progressions. We can use a heuristic: keep numbers where
    # 2*x mod p is "large" for small primes p, making it less likely that
    # 2*x equals the sum of two other elements modulo p.
    #
    # Actually, let's think differently. The constraint is: no element a in S 
    # satisfies a = mean(T) for any subset T of S\{a} with |T| >= 2.
    # This means sum(T) != |T| * a for all such T.
    #
    # For pairs: a = (b+c)/2 means 2a = b+c. So we want to avoid 3-term APs.
    # For triples: a = (b+c+d)/3 means 3a = b+c+d. Etc.
    #
    # The most restrictive case is pairs (2-element subsets), which gives us
    # the condition that no three elements form an arithmetic progression.
    # A set with no 3-term AP is a "sum-free" type set. The largest such set
    # in [1, n] has size roughly n / log(n) or so, but actually for no 3-AP,
    # the max size is about n * 2^{-log2(n)} which is very small... 
    # Wait, no. The largest subset of [1,n] with no 3-term AP is known to be
    # approximately n / (log n)^{1/2} or similar. For n=512, that's maybe ~100?
    # But we need to also avoid higher-order means.
    #
    # Let's use a greedy approach with efficient checking.
    # 
    # Strategy: 
    # 1. Start with a candidate pool from [1, 512].
    # 2. Use modular sieve: for each small prime p, remove x if there exist
    #    two other candidates b, c such that b + c ≡ 2x (mod p). But this is
    #    too aggressive. Instead, let's just use a good ordering heuristic.
    #
    # Better approach: Greedy with bitset-based mean checking.
    # For each candidate x (in some order), check if adding x to the current set
    # violates the mean condition. The check involves: for each a in the new set,
    # verify that a is not the mean of any subset of the others.
    #
    # To speed up: precompute for the current set S, for each possible target mean m,
    # whether there exists a subset of S\{m} whose sum equals |T|*m.
    #
    # Let's implement a faster check using dynamic programming on subset sums.
    
    import numpy as np
    
    # Greedy construction with efficient validation
    selected = []
    # Try candidates in an order that maximizes the chance of fitting 48 elements.
    # Use a heuristic: try numbers in a "spread out" order to avoid clustering.
    # One good heuristic: try numbers in order of their 2-adic valuation (powers of 2 first? or last?)
    # Actually, for avoiding 3-term APs, it's known that taking numbers with all bits set
    # (like 1, 3, 7, 15, ...) gives a set with no 3-term AP. But that's only 9 numbers up to 512.
    #
    # Let's try: sort candidates by a hash or by their value in a way that spreads them out.
    # A common trick: use the order from the "Stanley sequence" or similar.
    #
    # Simplest effective heuristic: try all numbers from 1 to 512, but in an order
    # that prioritizes numbers far from existing selected numbers.
    #
    # For efficiency, let's implement the mean check using subset-sum DP.
    
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
            # This is equivalent to: does there exist k in {2, ..., m} and a subset
            # of size k with sum = k*a?
            
            # Use DP: dp[k][s] = True if we can pick k elements from others with sum s.
            # But this is expensive. Let's optimize.
            
            # For k=2: check if there exist b, c in others with b+c = 2a.
            # This is a simple pair-sum check.
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
            # The maximum possible sum for k elements is bounded.
            # Let's use a bitset DP for subset sums.
            # dp[s] = bitmask of subset sizes that can achieve sum s
            # Or: dp[k] = set of achievable sums with k elements.
            
            # Given the small size (max 48 elements), we can use a simpler approach.
            # For each k from 3 to m, check if k*a is achievable as a sum of k elements from others.
            
            # Use numpy for bitset operations
            max_sum = sum(others)
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
                if target_sum <= max_sum and (dp[k] >> target_sum) & 1:
                    return False
        
        return True
    
    # Greedy: try candidates in an order that maximizes success
    # Heuristic: try numbers in descending order of "isolation" or use a random-ish order
    # Let's try: sort by (x % 7, x % 5, x) to spread out modulo small primes
    candidates = list(range(1, 513))
    # Sort by a composite key to spread out
    candidates.sort(key=lambda x: (x % 7, x % 5, x % 3, x))
    
    for x in candidates:
        if len(selected) >= max_size:
            break
        if is_valid_mean_check(selected, x):
            selected.append(x)
    
    return selected
