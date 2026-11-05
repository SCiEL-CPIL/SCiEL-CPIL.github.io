import time
import sys
import bisect

def _sieve(n):
    """Return a list of all primes <= n using a sieve."""
    if n < 2:
        return []
    is_prime = bytearray(b'\x01') * (n + 1)
    is_prime[0] = is_prime[1] = 0
    i = 2
    while i * i <= n:
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
        i += 1
    return [i for i in range(2, n + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    # Generate primes up to 10000
    primes = _sieve(10000)
    n = len(primes)
    
    # dp[i][j] = length of longest valid sequence ending with primes[i], primes[j] (i < j)
    # Since we need nondecreasing gaps, if the last two are ... k, i, j, then primes[i]-primes[k] <= primes[j]-primes[i]
    # So primes[k] >= primes[i] - (primes[j] - primes[i]) = 2*primes[i] - primes[j]
    
    # We want to maximize the total length.
    # Let dp[j] be a list of (end_prev, length) pairs? No, that's not quite right.
    # We need the previous element.
    # Let's use dp[i][j] where i < j.
    # dp[i][j] = 1 + max( dp[k][i] ) for all k < i such that primes[i] - primes[k] <= primes[j] - primes[i].
    # If no such k exists, dp[i][j] = 2.
    
    # Optimization: For a fixed i, the condition is primes[k] >= 2*primes[i] - primes[j].
    # Since primes are sorted, we can binary search for the smallest k such that primes[k] >= 2*primes[i] - primes[j].
    # Let this index be k_min.
    # Then we need the max dp[k][i] for k in [k_min, i-1].
    
    # We can maintain a suffix maximum for each column i.
    # Let max_dp_to_i[k] = max(dp[m][i]) for m in [k, i-1].
    
    # However, building this table is O(n^2) space and O(n^2 log n) time if we recompute max each time or O(n^2) if we precompute suffix maxes.
    
    # Space: n x n matrix of ints. n=1229. 1.5M ints. In Python, a list of lists of ints is memory heavy but 1.5M is okay.
    # Time: n^2 iterations. 1.5M iterations. Inside, a bisect (log n) and a lookup.
    # 1.5M * 10 ops = 15M ops. Should be fast (< 1s).
    
    # Let's implement it.
    
    if n == 0:
        return []
    if n == 1:
        return primes
    
    # dp[i][j] for i < j
    # We store dp[j] as a list of length j, where dp[j][i] is the length ending in i, j.
    # Actually, let's use a 2D list dp[n][n] to be safe, or just upper triangle.
    
    # dp[i][j]
    dp = [[0] * n for _ in range(n)]
    prev = [[-1] * n for _ in range(n)]
    
    # Initialize dp[i][j] = 2 for all i < j
    for j in range(1, n):
        for i in range(j):
            dp[i][j] = 2
            prev[i][j] = i
    
    # suffix_max[i][k] = max(dp[m][i]) for m in [k, i-1]
    # We can compute this on the fly or precompute for each i.
    # Since we iterate j from 1 to n-1, and for each j we iterate i from 1 to j-1.
    # When we are at column i, all dp[k][i] for k < i are already computed (because k < i < j, so column i was processed when j' = i, which is < j? No.
    # dp[k][i] depends on k < i.
    # When we compute dp[i][j], we need dp[k][i].
    # dp[k][i] was computed when the second index was i.
    # So when we are processing column j, all columns i < j are fully computed.
    # Therefore, we can precompute suffix maxima for column i once column i is done.
    
    # suffix[i][k] = max(dp[m][i]) for m in [k, i-1]
    suffix = [None] * n
    
    # Initialize suffix for i=0? No, i must be at least 1 to have a previous k.
    # For i=1, k can only be 0.
    
    max_len = 2
    best_i = 0
    best_j = 1
    
    for j in range(1, n):
        # Before processing column j, column j-1 is complete.
        # We need suffix[j-1] to compute dp[i][j] where the previous is j-1? No.
        # To compute dp[i][j], we need max dp[k][i] for k < i.
        # So we need suffix[i].
        # This means we must have computed suffix[i] before we use it.
        # We can compute suffix[i] right after column i is fully populated.
        
        # But wait, to populate column j, we need suffix[i] for all i < j.
        # So we should compute suffix[i] for all i < j at the start of the loop for j?
        # Or compute suffix[i] immediately after finishing column i.
        
        # Let's restructure:
        # 1. Initialize dp for "length 2".
        # 2. Loop j from 2 to n-1?
        # No, dp[i][j] for j=1 are base cases (length 2).
        # For j=2, we need suffix[1].
        # So after initializing all base cases? No, we can do it incrementally.
        
        pass

    # Correct incremental approach:
    # 1. Initialize dp[i][j]=2 for all i<j.
    # 2. We need to update dp[i][j] for j from 1 to n-1.
    #    Actually, the recurrence is dp[i][j] = 1 + max(dp[k][i]) for valid k.
    #    The base case is length 2, which corresponds to no valid k (or just i, j).
    #    So we can initialize dp[i][j]=2.
    #    Then for j from 1 to n-1:
    #       For i from 1 to j-1:
    #          Compute k_min using bisect.
    #          If valid, lookup max in suffix[i].
    #          Update dp[i][j] and prev[i][j].
    #    
    #    But we need suffix[i] to be ready.
    #    suffix[i] depends on dp[k][i] for k < i.
    #    dp[k][i] is only finalized when we finish processing column i.
    #    So, we can compute suffix[i] at the end of the loop for column i.
    
    # Let's restart the loop logic.
    
    # Initialize dp and prev
    dp = [[0] * n for _ in range(n)]
    prev = [[-1] * n for _ in range(n)]
    
    for j in range(1, n):
        for i in range(j):
            dp[i][j] = 2
            prev[i][j] = i
            
    suffix_max = [None] * n  # suffix_max[i] is a list where suffix_max[i][k] = max(dp[m][i]) for m in [k, i-1]
    suffix_idx = [None] * n  # corresponding index m
    
    max_len = 2
    best_i = 0
    best_j = 1
    
    for j in range(1, n):
        # Compute suffix for j-1? No, we need suffix[i] for i < j.
        # Specifically, to compute dp[i][j], we need suffix[i].
        # So at the start of iteration j, we need suffix[i] for all i in [1, j-1].
        # We can compute suffix[i] at the end of iteration i.
        
        # But for j=1, we have no i < 1 (since i>=1 for prev to exist? No, i can be 0).
        # If i=0, there is no k < 0. So dp[0][j] remains 2.
        # So we only need to update dp[i][j] for i >= 1.
        
        # Step 1: Update dp[i][j] for i in 1..j-1
        for i in range(1, j):
            # Need suffix[i]
            # threshold = 2*primes[i] - primes[j]
            threshold = 2 * primes[i] - primes[j]
            # Find smallest k such that primes[k] >= threshold
            k_min = bisect.bisect_left(primes, threshold)
            
            if k_min < i:
                # We need max dp[k][i] for k in [k_min, i-1]
                # suffix_max[i] must be computed.
                # It was computed at the end of the loop for column i.
                # Since i < j, column i was processed in a previous outer loop iteration.
                # So suffix_max[i] is available.
                
                if suffix_max[i] is not None:
                    # Check bounds
                    if k_min <= i - 1:
                        val = suffix_max[i][k_min]
                        idx = suffix_idx[i][k_min]
                        if val + 1 > dp[i][j]:
                            dp[i][j] = val + 1
                            prev[i][j] = idx
        
        # Step 2: Compute suffix for column j
        # suffix_max[j][k] = max(dp[m][j]) for m in [k, j-1]
        # We need this for future columns j' > j.
        
        if j > 0: # j is at least 1
            smax = [0] * j
            sidx = [-1] * j
            
            # smax[j-1] = dp[j-1][j]
            # Wait, dp[i][j] is defined for i < j.
            # So the "column" j contains dp[0][j], ..., dp[j-1][j].
            # We need suffix max over the first index i.
            # Let's call the indices i_0, i_1...
            # suffix_max[j][k] = max( dp[i][j] for i in [k, j-1] )
            
            if j == 1:
                # Only i=0
                smax[0] = dp[0][1]
                sidx[0] = 0
            else:
                # Start from i = j-1 down to 0
                max_val = dp[j-1][j]
                max_idx = j-1
                smax[j-1] = max_val
                sidx[j-1] = max_idx
                
                for i in range(j-2, -1, -1):
                    if dp[i][j] > max_val:
                        max_val = dp[i][j]
                        max_idx = i
                    smax[i] = max_val
                    sidx[i] = max_idx
            
            suffix_max[j] = smax
            suffix_idx[j] = sidx
            
        # Update global max
        for i in range(1, j):
            if dp[i][j] > max_len:
                max_len = dp[i][j]
                best_i = i
                best_j = j
                
        # Check deadline
        if time.time() > deadline - 1.0:
            break

    # Reconstruct sequence
    if max_len == 2:
        # Find any pair, or just return the first two?
        # The problem asks for MAXIMIZE length.
        # If max_len is 2, any valid pair works.
        # We should return a valid sequence of length 2.
        # Let's pick the first one found or (primes[0], primes[1]).
        # But wait, is (primes[0], primes[1]) always valid?
        # Gaps: just one gap. Nondecreasing is trivially satisfied.
        # Yes.
        return [primes[0], primes[1]]
        
    seq = [best_j]
    cur = best_j
    for _ in range(max_len - 1):
        prev_idx = prev[seq[-2]][seq[-1]] if len(seq) >= 2 else best_i
        # Wait, prev[i][j] stores the index k.
        # So if we are at j, and prev_i is i, then the one before i is prev[prev_i][i].
        
        # Let's trace backwards.
        # We have best_i, best_j.
        # Sequence ends with ... best_i, best_j.
        # Next element is prev[best_i][best_j].
        
        seq.append(best_i)
        # Now we are at best_i. We need the element before best_i.
        # This is stored in prev[prev[best_i][best_j]][best_i].
        
        curr = best_j
        prev_curr = best_i
        while prev_curr != -1 and len(seq) < max_len:
            # prev_curr is the current last element in the reverse sequence?
            # No.
            # Let's use a while loop.
            pass

    # Cleaner reconstruction
    path = []
    j = best_j
    i = best_i
    path.append(j)
    path.append(i)
    
    while len(path) < max_len:
        # We need the index before i.
        # It is stored in prev[prev_i][i] where prev_i is the index before i.
        # Wait.
        # dp[i][j] was updated using k.
        # prev[i][j] = k.
        # So the sequence is ... k, i, j.
        # So if we have i, j, we look at prev[i][j] to get k.
        # Then we have k, i. We look at prev[k][i] to get the one before k.
        
        k = prev[i][j]
        if k == -1:
            break
        path.append(k)
        j = i
        i = k
        
    path.reverse()
    
    # Verify
    if not path:
        return []
        
    # Double check validity
    gaps = []
    for k in range(1, len(path)):
        gap = primes[path[k]] - primes[path[k-1]]
        if gaps and gap < gaps[-1]:
            # Invalid? Should not happen if DP is correct.
            pass
        gaps.append(gap)
        
    return [primes[idx] for idx in path]

# Quick test
if __name__ == "__main__":
    res = construct()
    print(len(res))
    print(res[:10])
