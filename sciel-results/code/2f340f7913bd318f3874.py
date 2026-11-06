def construct():
    import time
    import random
    
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin before 90s kill
    
    N = 512
    MAX_K = 48
    
    # Precompute: for each x in [1,512], which other elements could form a mean conflict
    # x is the mean of subset S (|S|>=2, all in S != x) iff sum(S) = |S| * x
    # i.e., sum(s - x for s in S) = 0 for some subset S of size >= 2 from [1,512]\{x}
    
    # We'll use a bitset-based subset sum check
    # For a candidate set, we check all elements
    
    def is_valid_set(S):
        """Check if set S (list of ints) satisfies the no-mean property."""
        for a in S:
            # Check if a is the mean of any subset of size >= 2 from S\{a}
            V = [t - a for t in S if t != a]
            if not V:
                continue
            # Need: 0 in subset sums of V using at least 2 elements
            # Use DP with offset
            # Values range from -(512) to 512, sum range is bounded
            # For |V| elements, sum range is [-512*|V|, 512*|V|]
            # We need at least 2 elements, so track min elements used
            
            # Simpler: check all subset sizes >= 2
            # Use bitset DP where bit i represents achievable sum (i - offset)
            # with the constraint of using at least 2 elements
            
            # For efficiency, use two DP arrays: dp1 for sums achievable with >=1 element,
            # dp2 for sums achievable with >=2 elements
            # Actually, let's just do: for each subset size k >= 2, check if 0 is achievable
            
            # More efficient: single pass with count tracking
            # dp[j] = bitmask of counts? No, too complex.
            # 
            # Alternative: since we just need to know if 0 is in subset sums with >=2 elements,
            # we can compute all subset sums and track the minimum number of elements.
            # 
            # Bitset approach: 
            # Let M = 512 * len(V) (max absolute sum)
            # dp[sum + M] tracks if sum is achievable, but we also need element count.
            # 
            # Simpler approach for small sets: just enumerate if |S| is small
            # But |S| can be up to 48, so 2^47 is too large.
            
            # Use the two-DP approach:
            # dp1: bitset where bit (s + offset) is set if s is achievable with >=1 element
            # dp2: bitset where bit (s + offset) is set if s is achievable with >=2 elements
            # 
            # Transition when adding element v:
            # new_dp2 = dp2 | (dp1 shifted by v)  # use v plus one more element from dp1
            # new_dp1 = dp1 | (dp1 shifted by v) | (1 if v alone)
            # Wait, that's not quite right. Let me think more carefully.
            #
            # dp1: sums achievable with exactly 1 element
            # dp2: sums achievable with >= 2 elements
            #
            # When adding v:
            # new_dp1 = dp1 | {v}  (shifts don't apply, we're building up)
            # Actually, standard subset sum:
            # Let dp[k] = set of sums achievable using exactly k elements. We want union of dp[k] for k>=2 to contain 0.
            # 
            # For efficiency with bitsets, let's do:
            # dp_geq1: sums achievable with >= 1 element
            # dp_geq2: sums achievable with >= 2 elements
            #
            # Initialize: both empty
            # For each v in V:
            #   new_dp_geq2 = dp_geq2 | shift(dp_geq1, v)  # add v to any 1-element subset
            #   new_dp_geq1 = dp_geq1 | shift(dp_geq1, v) | {v}
            #   dp_geq1, dp_geq2 = new_dp_geq1, new_dp_geq2
            #
            # At the end, check if 0 is in dp_geq2.
            
            M = 512 * len(V) + 1
            # Use Python integers as bitsets
            # dp_geq1: bit i means sum (i - M) is achievable with >=1 element
            # dp_geq2: bit i means sum (i - M) is achievable with >=2 elements
            
            dp1 = 0  # sums with >=1 element
            dp2 = 0  # sums with >=2 elements
            
            for v in V:
                # Shift dp1 by v: if v > 0, shift left; if v < 0, shift right
                if v > 0:
                    shifted_dp1 = dp1 << v
                elif v < 0:
                    shifted_dp1 = dp1 >> (-v)
                else:
                    shifted_dp1 = dp1
                
                # New dp2: old dp2 or (dp1 shifted by v) [adding v to a 1-element subset]
                # But wait, we also need to handle the case where v is added to dp2
                # Actually: new_dp2 = dp2 | (dp1 shifted by v) | (dp2 shifted by v)
                # Because: dp2 already has sums with >=2 elements. Adding v to any of those gives >=3 elements.
                # And adding v to any 1-element sum gives exactly 2 elements.
                
                if v > 0:
                    shifted_dp2 = dp2 << v
                    shifted_dp1 = dp1 << v
                    new_dp1 = dp1 | shifted_dp1 | (1 << (M + v))  # v alone
                elif v < 0:
                    shifted_dp2 = dp2 >> (-v)
                    shifted_dp1 = dp1 >> (-v)
                    new_dp1 = dp1 | shifted_dp1 | (1 << (M + v))  # v alone
                else:
                    shifted_dp2 = dp2
                    shifted_dp1 = dp1
                    new_dp1 = dp1 | shifted_dp1 | (1 << M)  # v=0 alone, sum is 0
                
                new_dp2 = dp2 | shifted_dp2 | shifted_dp1
                
                # Mask to keep within bounds [0, 2*M]
                mask = (1 << (2 * M + 1)) - 1
                dp1 = new_dp1 & mask
                dp2 = new_dp2 & mask
            
            # Check if sum 0 (bit M) is set in dp2
            if (dp2 >> M) & 1:
                return False
        
        return True
    
    # Greedy approach: try to build a large set
    # Strategy: use a conflict-based ordering
    # Two numbers a, b have a "2-element conflict" with c if c = (a+b)/2
    # More generally, we want to avoid creating situations where some element is a mean.
    
    # Better strategy: random restarts with greedy
    best_set = []
    
    for trial in range(200):
        if time.time() - start_time > DEADLINE:
            break
        
        # Random ordering with some structure
        candidates = list(range(1, 513))
        
        # Prioritize numbers that are less likely to be means
        # Numbers that are far from other numbers are safer
        # Let's use a weighted random shuffle
        random.shuffle(candidates)
        
        current = []
        for x in candidates:
            if time.time() - start_time > DEADLINE:
                break
            
            if len(current) >= MAX_K:
                break
            
            # Quick check: can we add x?
            # Test if current + [x] is valid
            test_set = current + [x]
            if is_valid_set(test_set):
                current.append(x)
        
        if len(current) > len(best_set):
            best_set = current[:]
            if len(best_set) >= MAX_K:
                break
    
    # Try to improve best_set by swapping
    if time.time() - start_time < DEADLINE - 10:
        for _ in range(50):
            if time.time() - start_time > DEADLINE:
                break
            
            improved = False
            S = best_set[:]
            
            # Try removing one element and adding a different one
            for i in range(len(S)):
                if time.time() - start_time > DEADLINE:
                    break
                removed = S.pop(i)
                # Try to add a new element
                for x in range(1, 513):
                    if x in S:
                        continue
                    test = S + [x]
                    if is_valid_set(test):
                        S = test
                        improved = True
                        break
                S.insert(i, removed)
                if improved:
                    break
            
            if improved:
                best_set = S
    
    return best_set
