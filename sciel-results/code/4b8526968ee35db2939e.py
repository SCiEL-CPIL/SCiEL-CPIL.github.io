import time
import math
from bisect import bisect_left

def construct():
    start_time = time.time()
    DEADLINE = 55.0  # 55 seconds safety margin
    
    N = 10000
    # Sieve of Eratosthenes
    is_prime = [True] * (N + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(N)) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
    
    primes = [i for i, val in enumerate(is_prime) if val]
    M = len(primes)
    
    if M < 2:
        return primes
    
    # dp[i][j] = max length of sequence ending with primes[j], primes[i]
    # bp[i][j] = the k such that dp[i][j] = dp[j][k] + 1 (or -1 if length is 2)
    dp = [None] * M
    bp = [None] * M
    dp[0] = []
    bp[0] = []
    dp[1] = [2]
    bp[1] = [-1]  # length 2, no predecessor
    
    global_max = 2
    best_i, best_j = 1, 0
    
    for j in range(0, M - 1):
        # Build suffix max array for row j
        # dp[j] has length j
        # suf_max[k] = max(dp[j][k], dp[j][k+1], ..., dp[j][j-1])
        # suf_arg[k] = argmax index in [k, j-1]
        if j == 0:
            # No predecessors for j=0
            for i in range(1, M):
                if dp[i] is None:
                    dp[i] = [0] * i
                    bp[i] = [-1] * i
                dp[i][0] = 2
                bp[i][0] = -1
                if 2 > global_max:
                    global_max = 2
                    best_i, best_j = i, 0
            continue
        
        suf_max = [0] * (j + 1)
        suf_arg = [-1] * (j + 1)
        # suf_max[j] and suf_arg[j] remain 0 and -1 (unused)
        suf_max[j-1] = dp[j][j-1]
        suf_arg[j-1] = j-1
        for k in range(j-2, -1, -1):
            if dp[j][k] > suf_max[k+1]:
                suf_max[k] = dp[j][k]
                suf_arg[k] = k
            else:
                suf_max[k] = suf_max[k+1]
                suf_arg[k] = suf_arg[k+1]
        
        # Initialize k_curr
        T = 2 * primes[j] - primes[j+1]
        k_curr = bisect_left(primes, T)
        if k_curr > j:
            k_curr = j
        
        for i in range(j + 1, M):
            if time.time() - start_time > DEADLINE:
                break
            
            T = 2 * primes[j] - primes[i]
            # T decreases as i increases, so k_curr can only decrease
            while k_curr > 0 and primes[k_curr - 1] >= T:
                k_curr -= 1
            
            if k_curr < j:
                val = suf_max[k_curr] + 1
                pred = suf_arg[k_curr]
            else:
                val = 2
                pred = -1
            
            # Store dp[i][j] and bp[i][j]
            if dp[i] is None:
                dp[i] = [0] * i
                bp[i] = [-1] * i
            dp[i][j] = val
            bp[i][j] = pred
            
            if val > global_max:
                global_max = val
                best_i, best_j = i, j
    
    # Reconstruct the sequence using backpointers
    seq = [primes[best_i], primes[best_j]]
    cur_i, cur_j = best_i, best_j
    while cur_j >= 0 and bp[cur_i][cur_j] != -1:
        k = bp[cur_i][cur_j]
        seq.append(primes[k])
        cur_i, cur_j = cur_j, k
    
    seq.reverse()
    return seq

# Test locally
if __name__ == "__main__":
    res = construct()
    print(len(res))
    print(res[:10])
    # Verify
    gaps = [res[i+1] - res[i] for i in range(len(res)-1)]
    for i in range(len(gaps)-1):
        assert gaps[i] <= gaps[i+1], f"Gap violation at {i}: {gaps[i]} > {gaps[i+1]}"
    assert all(res[i] < res[i+1] for i in range(len(res)-1))
    assert all(p <= 10000 for p in res)
    # Check primality
    def is_prime_check(n):
        if n < 2: return False
        for i in range(2, int(math.isqrt(n)) + 1):
            if n % i == 0: return False
        return True
    assert all(is_prime_check(p) for p in res)
