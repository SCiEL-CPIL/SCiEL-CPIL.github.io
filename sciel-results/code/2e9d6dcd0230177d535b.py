import time
import numpy as np

def construct():
    """
    Finds the longest subsequence of primes <= 10000 with non-decreasing gaps.
    Uses Dynamic Programming with vectorized NumPy operations.
    Optimized memory usage with smaller dtypes.
    """
    deadline = time.time() + 85.0
    LIMIT = 10000
    
    # 1. Generate primes using Sieve of Eratosthenes
    sieve = np.ones(LIMIT + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for i in range(2, int(LIMIT**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    primes = np.array([i for i, is_p in enumerate(sieve) if is_p], dtype=np.int32)
    N = len(primes)
    
    G = LIMIT + 1
    
    # dp[i][g]: max length of valid sequence ending at primes[i] with last gap exactly g.
    # Lengths are small, use uint8.
    dp = np.zeros((N, G), dtype=np.uint8)
    dp[:, 0] = 1
    
    # pred[i][g]: index j of previous prime in optimal sequence ending at i with gap g.
    # Indices < 2000, use int16.
    pred = np.full((N, G), -1, dtype=np.int16)
    
    # pref[i][g]: max(dp[i][0], ..., dp[i][g])
    # Use uint8.
    pref = np.zeros((N, G), dtype=np.uint8)
    
    best_len = 1
    best_i = 0
    best_g = 0
    
    # 2. Fill DP table
    for i in range(1, N):
        if time.time() > deadline:
            break
            
        pi = primes[i]
        
        # Vectorized approach
        gaps = pi - primes[:i]  # shape (i,)
        
        indices_j = np.arange(i, dtype=np.int32)
        # pref is uint8, so vals will be uint8.
        # Max length is small, so no overflow.
        vals = pref[indices_j, gaps] + 1
        
        current_dp_row = dp[i, gaps]
        mask = vals > current_dp_row
        if np.any(mask):
            dp[i, gaps[mask]] = vals[mask]
            pred[i, gaps[mask]] = indices_j[mask].astype(np.int16)
        
        # Compute prefix max for row i
        pref[i] = np.maximum.accumulate(dp[i])
        
        # Update global best
        max_len_i = np.max(dp[i])
        if max_len_i > best_len:
            best_len = max_len_i
            best_g = int(np.argmax(dp[i]))
            best_i = i
            
    # 3. Reconstruct sequence
    seq = []
    if best_len > 0:
        i = best_i
        g = best_g
        while i >= 0:
            seq.append(int(primes[i]))
            if i == 0:
                break
            j = int(pred[i][g])
            if j == -1:
                break
            
            target_val = int(pref[j][g])
            # Find h such that dp[j][h] == target_val and h <= g
            # We want to find the specific gap h used in the optimal path to j with max gap <= g.
            # Any h satisfying dp[j][h] == target_val and h <= g is valid for the chain.
            # We pick the largest h to be consistent with the "last gap" concept?
            # Actually, it doesn't matter for validity, as long as h <= g.
            # But we need to update the state to (j, h).
            
            # Optimization: Instead of np.where on the whole row, we can search in [0, g]
            # But np.where is implemented in C and is fast.
            h_candidates = np.where(dp[j, :g+1] == target_val)[0]
            if len(h_candidates) > 0:
                h = int(h_candidates[-1])
            else:
                h = 0
                
            i = j
            g = h
        seq.reverse()
        
    return seq
