import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50 seconds safety margin
    
    # 1. Sieve of Eratosthenes for primes <= 10000
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    
    primes = [i for i, flag in enumerate(is_prime) if flag]
    m = len(primes)
    
    # 2. DP Table
    # dp[i][j] stores length of longest valid sequence ending with primes[j], primes[i]
    # where j < i
    # We only need dp[i][j] for i > j
    # Initialize with 0
    dp = [[0] * m for _ in range(m)]
    prev = [[-1] * m for _ in range(m)]
    
    max_len = 0
    best_i = 0
    best_j = 0
    
    # Iterate j from 0 to m-2
    for j in range(m - 1):
        # Precompute suffix max and argmax for dp[j][0..j-1]
        # suf_max[k] = max(dp[j][k], ..., dp[j][j-1])
        # suf_argmax[k] = index h in [k, j-1] achieving suf_max[k]
        suf_max = [0] * j
        suf_argmax = [0] * j
        if j > 0:
            suf_max[j-1] = dp[j][j-1]
            suf_argmax[j-1] = j-1
            for k in range(j-2, -1, -1):
                if dp[j][k] >= suf_max[k+1]:
                    suf_max[k] = dp[j][k]
                    suf_argmax[k] = k
                else:
                    suf_max[k] = suf_max[k+1]
                    suf_argmax[k] = suf_argmax[k+1]
        
        if j == 0:
            for i in range(j + 1, m):
                dp[i][j] = 2
                prev[i][j] = -1
                if dp[i][j] > max_len:
                    max_len = dp[i][j]
                    best_i = i
                    best_j = j
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
            continue
        
        # Initialize h_ptr for i = j+1 using binary search
        threshold_init = 2 * primes[j] - primes[j+1]
        lo, hi = 0, j - 1
        h_ptr = -1
        while lo <= hi:
            mid = (lo + hi) // 2
            if primes[mid] >= threshold_init:
                h_ptr = mid
                hi = mid - 1
            else:
                lo = mid + 1
        
        # Iterate i from j+1 to m-1, maintaining h_ptr (non-increasing)
        for i in range(j + 1, m):
            threshold = 2 * primes[j] - primes[i]
            
            # Update h_ptr: threshold decreases as i increases, so h_ptr moves left
            if h_ptr >= 0:
                while h_ptr > 0 and primes[h_ptr - 1] >= threshold:
                    h_ptr -= 1
            else:
                # No valid h for previous threshold; with smaller threshold, re-search
                lo2, hi2 = 0, j - 1
                while lo2 <= hi2:
                    mid = (lo2 + hi2) // 2
                    if primes[mid] >= threshold:
                        h_ptr = mid
                        hi2 = mid - 1
                    else:
                        lo2 = mid + 1
            
            if h_ptr != -1:
                best_h = suf_argmax[h_ptr]
                best_val = suf_max[h_ptr]
                dp[i][j] = 1 + best_val
                prev[i][j] = best_h
            else:
                dp[i][j] = 2
                prev[i][j] = -1
            
            if dp[i][j] > max_len:
                max_len = dp[i][j]
                best_i = i
                best_j = j
                
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # 3. Reconstruct path
    if max_len < 2:
        return primes[:max_len] # Fallback, though max_len should be at least 2 if m>=2
    
    path = []
    i, j = best_i, best_j
    path.append(primes[i])
    path.append(primes[j])
    
    while prev[i][j] != -1:
        h = prev[i][j]
        i, j = j, h
        path.append(primes[j])
        
    path.reverse()
    return path

# Test locally
if __name__ == "__main__":
    res = construct()
    print(f"Length: {len(res)}")
    print(f"Sequence: {res}")
    # Verify
    valid = True
    for k in range(len(res)-2):
        gap1 = res[k+1] - res[k]
        gap2 = res[k+2] - res[k+1]
        if gap1 > gap2:
            valid = False
            break
    print(f"Valid: {valid}")
    print(f"All primes <= 10000: {all(p <= 10000 for p in res)}")
    print(f"Increasing: {all(res[i] < res[i+1] for i in range(len(res)-1))}")
