<<<<<<< SEARCH
def _can_form_sum(target, vals):
    """
    Check if target can be formed as a sum of a subset of vals with size >= 1.
    vals are integers (can be negative).
    Returns True if yes.
    """
    if not vals:
        return False
        
    n = len(vals)
    # Sum of absolute values for bounds
    max_sum = sum(abs(v) for v in vals)
    if abs(target) > max_sum:
        return False
        
    offset = max_sum
    # We need subset size >= 1
    # dp[k] bitset
    dp = [0] * (n + 1)
    dp[0] = 1 << offset
    
    for v in vals:
        # Iterate k downwards
        for k in range(n, 0, -1):
            if dp[k-1] == 0:
                continue
            if v >= 0:
                shifted = dp[k-1] << v
            else:
                shifted = dp[k-1] >> (-v)
            dp[k] |= shifted
            
    # Check if any k >= 1 has bit (offset + target) set
    bit_idx = offset + target
    for k in range(1, n + 1):
        if dp[k] & (1 << bit_idx):
            return True
    return False
=======
def _can_form_sum(target, vals):
    """
    Check if target can be formed as a sum of a subset of vals with size >= 1.
    vals are integers (can be negative).
    Returns True if yes.
    """
    if not vals:
        return False
        
    n = len(vals)
    # Sum of absolute values for bounds
    max_sum = sum(abs(v) for v in vals)
    if abs(target) > max_sum:
        return False
        
    if target == 0:
        # Need subset of size >= 1 summing to 0
        # Check if any single element is 0
        if 0 in vals:
            return True
        # Otherwise need subset of size >= 2
        return _zero_sum_size_ge_2(vals)
        
    offset = max_sum
    # We need subset size >= 1
    # dp[k] bitset
    dp = [0] * (n + 1)
    dp[0] = 1 << offset
    
    for v in vals:
        # Iterate k downwards
        for k in range(n, 0, -1):
            if dp[k-1] == 0:
                continue
            if v >= 0:
                shifted = dp[k-1] << v
            else:
                shifted = dp[k-1] >> (-v)
            dp[k] |= shifted
            
    # Check if any k >= 1 has bit (offset + target) set
    bit_idx = offset + target
    for k in range(1, n + 1):
        if dp[k] & (1 << bit_idx):
            return True
    return False
>>>>>>> REPLACE

