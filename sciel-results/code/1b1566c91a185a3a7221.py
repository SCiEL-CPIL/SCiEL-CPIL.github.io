"""Program using optimized greedy construction with incremental mean-check."""

def construct():
    n = 512
    max_size = 48
    
    # Strategy: Pick numbers in increasing order.
    # When adding x, only check if any existing b is the mean of a subset containing x.
    # This is valid because:
    # 1. x is the largest element, so it cannot be the mean of smaller elements.
    # 2. Any subset not containing x was already validated in previous steps.
    
    candidates = list(range(1, n + 1))
    
    def is_valid_incremental(selected, x):
        """
        Check if adding x to selected (where x > all elements in selected)
        violates the mean-free property.
        """
        for b in selected:
            # Check if b is the mean of a subset U of (selected \ {b}) U {x}
            # with |U| >= 2.
            # Let U = {x} U T, where T is a subset of (selected \ {b}).
            # |U| >= 2 implies 1 + |T| >= 2 => |T| >= 1.
            # Mean condition: b = (x + sum(T)) / (1 + |T|)
            # => sum(T) = (1 + |T|) * b - x
            
            others = [y for y in selected if y != b]
            m = len(others)
            
            if m == 0:
                # T must be non-empty, but others is empty. No valid T.
                continue
                
            # DP for subset sums of 'others'
            # dp[k] is a bitset of achievable sums with k elements
            dp = [0] * (m + 1)
            dp[0] = 1  # sum 0 with 0 elements
            
            for y in others:
                for k in range(m, 0, -1):
                    dp[k] |= (dp[k-1] << y)
            
            # Check for each k >= 1 if target sum is achievable
            for k in range(1, m + 1):
                target_sum = (1 + k) * b - x
                if target_sum < 0:
                    continue
                if target_sum > 512 * k: # Max possible sum bound
                    continue
                if (dp[k] >> target_sum) & 1:
                    return False
        
        return True

    selected = []
    for x in candidates:
        if len(selected) >= max_size:
            break
        if is_valid_incremental(selected, x):
            selected.append(x)
    
    return selected
