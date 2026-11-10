"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import bisect

def construct():
    start = time.time()
    deadline = start + 50
    
    n_max = 10000
    is_prime = [True] * (n_max + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n_max**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n_max + 1, i):
                is_prime[j] = False
    primes = [p for p in range(2, n_max + 1) if is_prime[p]]
    n = len(primes)
    
    if n <= 2:
        return primes
    
    # dp[j][k] = length of longest valid subsequence ending with primes[j], primes[k]
    # where the last gap is primes[k] - primes[j]
    # We store only the best length and predecessor
    # dp is a 2D list of lists: dp[j][k] for j < k
    # To save memory and time, we use a list of dicts or just a flat structure
    # But n=1229, so n^2 is about 1.5M, which is fine.
    
    # dp[j][k] stores the length of the longest sequence ending at (j, k)
    # prev[j][k] stores the index i such that the sequence is ... -> i -> j -> k
    # We need to find, for each (j, k), the best i < j such that:
    #   primes[j] - primes[i] <= primes[k] - primes[j]
    #   i.e., primes[i] >= primes[j] - (primes[k] - primes[j]) = 2*primes[j] - primes[k]
    #   and dp[i][j] is maximized.
    
    # So for fixed j, as k varies, the threshold T = 2*primes[j] - primes[k] decreases as k increases.
    # We want max dp[i][j] for i in [lo, j-1] where primes[i] >= T.
    # Since primes is sorted, lo = bisect_left(primes, T).
    # We need a data structure that, for fixed j, can answer: max dp[i][j] for i >= lo.
    # This is a suffix maximum query on the array dp[0..j-1][j].
    
    # We can precompute suffix maxima for each j, but dp[i][j] is computed as we go.
    # Actually, dp[i][j] depends on dp[?][i], so we need to process j in increasing order,
    # and for each j, we need all dp[i][j] for i < j to be already computed.
    # dp[i][j] is computed when we process pair (i, j), which happens when the outer loop is at i.
    # So by the time we are at outer loop j, all dp[i][j] for i < j have been computed.
    
    # So for each j, we can build a suffix-max array over i=0..j-1 of dp[i][j].
    # Then for each k > j, we compute T = 2*primes[j] - primes[k], find lo, and query suffix_max[lo].
    
    # This is what the current code does. Let's optimize it.
    
    # Optimization: Instead of storing full dp and prev matrices, we can store
    # dp as a list of lists where dp[j] is a list of length n, but only entries k > j are valid.
    # Actually, let's use a 2D array but be careful with memory.
    
    # n = 1229, n^2 = 1,510,441. Each entry is an int. In Python, ints are objects, so this is heavy.
    # Let's use arrays of type 'i' or just lists. Lists of lists should be fine.
    
    # Alternative: use a list of dictionaries? No, lists are faster for indexing.
    
    # Let's try to optimize the inner loop by reducing overhead.
    
    dp = [[0] * n for _ in range(n)]
    prev = [[-1] * n for _ in range(n)]
    
    best_len = 2
    best_j, best_k = 0, 1
    
    for j in range(1, n):
        if time.time() > deadline:
            break
        # Build suffix max of dp[i][j] for i = 0..j-1
        # suff_val[i] = max(dp[i][j], dp[i+1][j], ..., dp[j-1][j])
        # suff_arg[i] = argmax
        suff_val = [0] * (j + 1)
        suff_arg = [-1] * (j + 1)
        for i in range(j - 1, -1, -1):
            if dp[i][j] >= suff_val[i + 1]:
                suff_val[i] = dp[i][j]
                suff_arg[i] = i
            else:
                suff_val[i] = suff_val[i + 1]
                suff_arg[i] = suff_arg[i + 1]
        
        for k in range(j + 1, n):
            gap_jk = primes[k] - primes[j]
            threshold = primes[j] - gap_jk
            lo = bisect.bisect_left(primes, threshold)
            if lo >= j:
                dp[j][k] = 2
                prev[j][k] = -1
            else:
                bv = suff_val[lo]
                ba = suff_arg[lo]
                if bv > 0:
                    dp[j][k] = bv + 1
                    prev[j][k] = ba
                else:
                    dp[j][k] = 2
                    prev[j][k] = -1
            
            if dp[j][k] > best_len:
                best_len = dp[j][k]
                best_j, best_k = j, k
    
    if best_len <= 2:
        return primes[:2]
    
    seq = [primes[best_k], primes[best_j]]
    j, k = best_j, best_k
    while len(seq) < best_len:
        i = prev[j][k]
        if i < 0:
            break
        seq.append(primes[i])
        k = j
        j = i
    
    seq.reverse()
    return seq
# EVOLVE-BLOCK-END
