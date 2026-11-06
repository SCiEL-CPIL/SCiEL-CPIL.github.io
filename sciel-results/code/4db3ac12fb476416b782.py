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
        # Precompute suffix max for dp[j][0..j-1]
        # suf_max[k] = max(dp[j][k], ..., dp[j][j-1])
        suf_max = [0] * j
        if j > 0:
            suf_max[j-1] = dp[j][j-1]
            for k in range(j-2, -1, -1):
                suf_max[k] = max(dp[j][k], suf_max[k+1])
        
        # Iterate i from j+1 to m-1
        for i in range(j + 1, m):
            # Condition: P[h] >= 2*P[j] - P[i]
            threshold = 2 * primes[j] - primes[i]
            
            # Find smallest h < j such that primes[h] >= threshold
            # Binary search in primes[0..j-1]
            lo, hi = 0, j - 1
            h_min = -1
            while lo <= hi:
                mid = (lo + hi) // 2
                if primes[mid] >= threshold:
                    h_min = mid
                    hi = mid - 1
                else:
                    lo = mid + 1
            
            if h_min != -1:
                # Max length ending with P[h], P[j] for h in [h_min, j-1]
                best_h = h_min
                best_val = suf_max[h_min]
                # We need to find which h gave this max value to store in prev
                # Since suf_max is non-decreasing as index decreases? No, suf_max[k] is max from k to end.
                # So suf_max[h_min] is the max value. We need to find an h that achieves it.
                # We can just scan from h_min to j-1 to find the first one with dp[j][h] == best_val?
                # Or we can store the argmax in a separate array.
                # For simplicity and speed, since j is small on average, we can just find it.
                # But wait, suf_max[h_min] is the max. Any h in [h_min, j-1] with dp[j][h] == suf_max[h_min] is valid.
                # Let's just pick the one that maximizes dp[j][h].
                # Actually, we can just use h_min if we don't care which specific predecessor, 
                # but we need to reconstruct the path.
                # Let's find the h in [h_min, j-1] that maximizes dp[j][h].
                # Since we have suf_max, we know the value.
                # We can iterate h from h_min to j-1 and pick the first one with dp[j][h] == best_val.
                # This is O(j) in worst case, but j is small.
                for h in range(h_min, j):
                    if dp[j][h] == best_val:
                        best_h = h
                        break
                
                dp[i][j] = 1 + best_val
                prev[i][j] = best_h
            else:
                dp[i][j] = 2
                prev[i][j] = -1
            
            if dp[i][j] > max_len:
                max_len = dp[i][j]
                best_i = i
                best_j = j
                
            # Time check
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
