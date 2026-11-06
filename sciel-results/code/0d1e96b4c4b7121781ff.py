import time

def construct():
    N = 123457
    start_time = time.time()
    deadline = start_time + 80  # 80s out of 90s limit
    
    # 1. Precompute largest prime factor (lpf) and list of primes
    lpf = [1] * (N + 1)
    primes = []
    
    for i in range(2, N + 1):
        if lpf[i] == 1:
            primes.append(i)
            # Mark multiples - standard sieve for largest prime factor
            for j in range(i, N + 1, i):
                lpf[j] = i
                
    # 2. Precompute for each possible B, whether there exists a valid pair
    # valid[B] = True if there exists a in [1, N//2] with lpf[a] <= B and lpf[N-a] <= B
    # We can do this more efficiently by checking each a once and recording the 
    # minimum B needed for that a to be valid
    
    # For each a in [1, N//2], the minimum B needed is max(lpf[a], lpf[N-a])
    # So we want to find the minimum over all a of max(lpf[a], lpf[N-a])
    
    # Direct approach: compute min over a of max(lpf[a], lpf[N-a])
    # This is O(N) which is fine for N=123457
    
    min_max_lpf = N + 1
    best_a = 1
    
    for a in range(1, N // 2 + 1):
        b = N - a
        m = max(lpf[a], lpf[b])
        if m < min_max_lpf:
            min_max_lpf = m
            best_a = a
    
    # The answer is min_max_lpf, and the pair is (best_a, N - best_a)
    # But we need to verify that min_max_lpf is achievable
    # Since we're taking the minimum over all valid a, this is guaranteed
    
    # However, the problem asks to minimize max(P(a), P(b)) where P is the largest prime factor
    # Our computation directly gives this minimum
    
    if time.time() > deadline:
        # Emergency fallback
        return [1, 123456]
    
    return [best_a, N - best_a]
