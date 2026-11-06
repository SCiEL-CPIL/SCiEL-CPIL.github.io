def construct():
    selected = []
    
    def is_valid(S):
        n = len(S)
        if n < 2:
            return True
        
        # Precompute sorted list for potential optimizations, though not strictly needed
        # We will check each element a
        for a in S:
            others = [y for y in S if y != a]
            m = len(others)
            if m < 2:
                continue
            
            # Bitset DP for subset sums of 'others'
            # dp[k] is a bitset where bit s is 1 if sum s is achievable with k elements
            dp = [0] * (m + 1)
            dp[0] = 1  # Bit 0 is set
            
            for y in others:
                # Update from high k to low k to avoid using y multiple times
                for k in range(m, 0, -1):
                    dp[k] |= (dp[k-1] << y)
            
            # Check if any k >= 2 has sum k*a
            for k in range(2, m + 1):
                target_sum = k * a
                # Check if bit target_sum is set in dp[k]
                if (dp[k] >> target_sum) & 1:
                    return False
        return True

    for x in range(512, 0, -1):
        candidate = selected + [x]
        if is_valid(candidate):
            selected = candidate
    
    return selected
