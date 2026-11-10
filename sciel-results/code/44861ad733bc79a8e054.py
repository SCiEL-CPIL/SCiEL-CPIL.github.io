"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from math import isqrt
    import bisect

    # Generate all primes <= 10000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, isqrt(n) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    primes = sieve(10000)
    n = len(primes)
    
    if n <= 2:
        return primes

    # dp[i][j] = length of longest valid subsequence ending with primes[i], primes[j]
    # parent[i][j] = index h that achieved dp[i][j]
    dp = [[0] * n for _ in range(n)]
    parent = [[-1] * n for _ in range(n)]
    
    # Initialize base cases
    for i in range(n):
        for j in range(i + 1, n):
            dp[i][j] = 2
            parent[i][j] = -1

    best_len = 2
    best_pair = (0, 1)
    
    # Precompute logs for sparse table
    log_table = [0] * (n + 1)
    for i in range(2, n + 1):
        log_table[i] = log_table[i // 2] + 1
        
    # Process each i as the second-to-last element
    for i in range(1, n - 1):
        # Build Sparse Table for the array A = [dp[0][i], ..., dp[i-1][i]]
        # We need to query max in range [L, i-1]
        
        # If i is small, we can just use a simple loop or a small array
        # But let's build the ST for efficiency
        
        # The array we are building ST on is V = [dp[h][i] for h in range(i)]
        # We also need to track the index h that gives the max value.
        
        # ST[k][x] will store (value, index) for the range [x, x + 2^k - 1]
        # We only need to build it for the current i.
        
        if i == 0:
            continue
            
        # Initialize ST for level 0
        # We can use a list of lists, or just a 2D array
        # Max length is i, so max k is log2(i)
        max_k = log_table[i]
        # st[k][x]
        st_val = [[0] * i for _ in range(max_k + 1)]
        st_idx = [[0] * i for _ in range(max_k + 1)]
        
        for h in range(i):
            st_val[0][h] = dp[h][i]
            st_idx[0][h] = h
            
        for k in range(1, max_k + 1):
            length = 1 << k
            half = length // 2
            for x in range(i - length + 1):
                # Compare [x, x+half-1] and [x+half, x+length-1]
                v1, i1 = st_val[k-1][x], st_idx[k-1][x]
                v2, i2 = st_val[k-1][x+half], st_idx[k-1][x+half]
                if v1 >= v2:
                    st_val[k][x] = v1
                    st_idx[k][x] = i1
                else:
                    st_val[k][x] = v2
                    st_idx[k][x] = i2

        # Now process j > i
        for j in range(i + 1, n):
            # Find L such that primes[L] >= 2*primes[i] - primes[j]
            threshold = 2 * primes[i] - primes[j]
            # L is the first index >= threshold
            L = bisect.bisect_left(primes, threshold)
            
            if L < i:
                # Query max in [L, i-1]
                # Length of range is i - L
                length = i - L
                k = log_table[length]
                
                # Compare [L, L+2^k-1] and [i-2^k+1, i-1]
                # Note: i-1 is the right end. The second interval starts at i - 2^k + 1?
                # No, the range is [L, i-1].
                # Interval 1: [L, L + 2^k - 1]
                # Interval 2: [i - 2^k, i - 1]
                
                start1 = L
                start2 = i - (1 << k)
                
                v1, i1 = st_val[k][start1], st_idx[k][start1]
                v2, i2 = st_val[k][start2], st_idx[k][start2]
                
                if v1 >= v2:
                    best_h = i1
                    best_val = v1
                else:
                    best_h = i2
                    best_val = v2
                    
                dp[i][j] = 2 + best_val
                parent[i][j] = best_h
            else:
                dp[i][j] = 2
                parent[i][j] = -1
                
            if dp[i][j] > best_len:
                best_len = dp[i][j]
                best_pair = (i, j)

    # Reconstruct the sequence
    if best_len == 2:
        return [primes[0], primes[1]]
    
    seq = []
    i, j = best_pair
    seq.append(primes[j])
    
    while parent[i][j] != -1:
        h = parent[i][j]
        seq.append(primes[i])
        j = i
        i = h
        
    seq.append(primes[i])
    seq.reverse()
    
    return seq
# EVOLVE-BLOCK-END
