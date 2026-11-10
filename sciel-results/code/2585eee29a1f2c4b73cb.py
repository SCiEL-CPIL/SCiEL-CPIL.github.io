def construct():
    import time
    import bisect
    from math import isqrt

    start_time = time.time()
    DEADLINE = 85.0

    def sieve(n):
        is_prime = bytearray(b'\x01') * (n + 1)
        if n >= 0: is_prime[0] = 0
        if n >= 1: is_prime[1] = 0
        for i in range(2, isqrt(n) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = 0
        return [i for i in range(n+1) if is_prime[i]]
    
    primes = sieve(10000)
    n = len(primes)
    
    # dp[i] is a list of (gap, length) tuples, sorted by gap.
    # Lengths are strictly increasing.
    dp = [None] * n
    # parent[i] is a dict mapping gap to (prev_idx, prev_gap)
    # We only store the parent for the states that are in the Pareto frontier of dp[i]
    parent = [None] * n
    
    best_length = 1
    best_i = 0
    best_gap = 0
    
    # Initialize dp[0]
    dp[0] = [(0, 1)]
    parent[0] = {0: (-1, 0)}
    
    for j in range(1, n):
        candidates = []
        
        for i in range(j):
            gap_to_i = primes[j] - primes[i]
            
            # Find the best state in dp[i] that can be extended to j
            # We need state (g, l) in dp[i] such that g <= gap_to_i
            # Since dp[i] is sorted by g, and l increases with g,
            # the best state is the one with the largest g <= gap_to_i.
            
            frontier_i = dp[i]
            if not frontier_i:
                continue
                
            # Binary search for rightmost index with gap <= gap_to_i
            # frontier_i is list of (gap, length)
            # We search on the gap part.
            
            # Create a list of just gaps for bisect? Or use a key function?
            # bisect doesn't support key in older versions, but we can extract gaps.
            # Since frontier_i is small, extracting is fast.
            gaps_i = [x[0] for x in frontier_i]
            
            idx = bisect.bisect_right(gaps_i, gap_to_i) - 1
            
            if idx >= 0:
                prev_gap, prev_len = frontier_i[idx]
                new_len = prev_len + 1
                # Candidate for dp[j]: (gap_to_i, new_len)
                # Source: (i, prev_gap)
                candidates.append((gap_to_i, new_len, i, prev_gap))
        
        if not candidates:
            # Should not happen as i=j-1 always works? 
            # If primes[j] - primes[j-1] is the gap, and dp[j-1] has (0,1), then 0 <= gap, so it works.
            dp[j] = [(0, 1)]
            parent[j] = {0: (-1, 0)}
            continue

        # Sort candidates by gap, then by length descending (to handle ties)
        candidates.sort(key=lambda x: (x[0], -x[1]))
        
        frontier_j = []
        parent_j = {}
        max_len_so_far = 0
        
        for gap, length, prev_i, prev_gap in candidates:
            if length > max_len_so_far:
                frontier_j.append((gap, length))
                parent_j[gap] = (prev_i, prev_gap)
                max_len_so_far = length
                if length > best_length:
                    best_length = length
                    best_i = j
                    best_gap = gap
        
        dp[j] = frontier_j
        parent[j] = parent_j
        
        if time.time() - start_time > DEADLINE:
            break
    
    # Reconstruct
    if best_length <= 1:
        return [primes[0]]
    
    seq = []
    i = best_i
    gap = best_gap
    
    while True:
        seq.append(primes[i])
        p = parent[i].get(gap)
        if p is None:
            break
        prev_i, prev_gap = p
        if prev_i == -1:
            break
        i = prev_i
        gap = prev_gap
        
    seq.reverse()
    
    # Verify
    if len(seq) >= 2:
        gaps = [seq[k+1] - seq[k] for k in range(len(seq)-1)]
        for k in range(len(gaps)-1):
            if gaps[k+1] < gaps[k]:
                # Fallback
                return [primes[0]]
                
    return seq
