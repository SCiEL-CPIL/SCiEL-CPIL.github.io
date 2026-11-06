import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    N_MIN = 10000
    N_MAX = 1000000
    
    # Precompute smallest prime factor up to N_MAX + 1
    max_val = N_MAX + 1
    spf = list(range(max_val + 1))
    for i in range(2, int(math.isqrt(max_val)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, max_val + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute Largest Prime Factor (LPF) for all numbers up to N_MAX + 1
    # lpf[i] = max(spf[i], lpf[i // spf[i]])
    lpf = [0] * (max_val + 1)
    lpf[0] = 0
    lpf[1] = 1
    for i in range(2, max_val + 1):
        p = spf[i]
        # Since p is the smallest prime factor, i // p < i (for i > 1)
        # So lpf[i // p] is already computed
        lpf[i] = p if p > lpf[i // p] else lpf[i // p]
    
    best_n = N_MIN
    best_val = float('inf')
    
    # Iterate through all valid n
    # The loop is O(N) with simple integer comparisons, very fast in Python
    for n in range(N_MIN, N_MAX + 1):
        elapsed = time.time() - start_time
        if elapsed > deadline:
            break
        
        # val = max(lpf[n], lpf[n+1])
        ln = lpf[n]
        ln1 = lpf[n + 1]
        val = ln if ln > ln1 else ln1
        
        if val < best_val:
            best_val = val
            best_n = n
            # Early termination if we hit the theoretical lower bound?
            # The smallest prime factor for any number > 1 is 2.
            # Can max(lpf(n), lpf(n+1)) be 2?
            # This would require n and n+1 to be powers of 2.
            # Consecutive powers of 2 are only 1 and 2.
            # Since n >= 10000, this is impossible.
            # Next smallest prime is 3.
            # Can max(lpf(n), lpf(n+1)) be 3?
            # n and n+1 must be of form 2^a * 3^b.
            # Let's see if such a pair exists in range.
            # If best_val becomes very small, we might be close to optimal.
            # However, without a proven lower bound for this specific range,
            # we continue scanning. The scan is fast.
            
    return best_n
