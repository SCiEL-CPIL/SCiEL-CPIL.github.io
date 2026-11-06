import time

def construct():
    """
    Finds the longest contiguous interval [start, start + length - 1] within [1, 1008]
    such that for all x, y in the interval, x * y % 1009 != 1.
    """
    p = 1009
    n = 1008
    
    # Precompute modular inverses
    # inv[x] is the unique y in [1, p-1] such that x*y % p == 1
    inv = [0] * (n + 1)
    for x in range(1, n + 1):
        inv[x] = pow(x, p - 2, p)
        
    best_start = 1
    best_length = 0
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50.0  # 50s out of 90s allowed
    
    # Iterate over all possible start positions
    for s in range(1, n + 1):
        # Check time limit
        if time.time() > deadline:
            break
            
        # For a fixed start s, find the maximum valid end e.
        # limit represents the first position that CANNOT be included in the interval.
        # Initially, we can go up to n.
        limit = n + 1
        
        # Scan x from s upwards
        for x in range(s, n + 1):
            # If x has reached the limit, we cannot include x.
            # The interval [s, x-1] is the max valid interval for this s.
            if x >= limit:
                break
            
            ix = inv[x]
            # If the inverse of x is also >= s, then if we include x,
            # we must NOT include ix.
            # Since ix >= s, ix is to the right of (or at) s.
            # So the interval end must be < ix.
            if ix >= s:
                if ix < limit:
                    limit = ix
                    # If limit becomes <= x, the loop will break in the next iteration
                    # or immediately if we check x >= limit.
                    # Actually, if ix == x, limit becomes x.
                    # Next check x >= limit will be true, so we break.
                    # This correctly handles self-inverse elements.
        
        max_valid_e = limit - 1
        if max_valid_e >= s:
            length = max_valid_e - s + 1
            if length > best_length:
                best_length = length
                best_start = s
        else:
            # No valid interval of length >= 1 starting at s
            pass

    # Fallback if no valid interval found (should not happen, e.g., [2,2] is valid)
    if best_length == 0:
        best_start = 2
        best_length = 1

    return [best_start, best_length]
