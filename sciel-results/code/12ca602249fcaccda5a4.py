import math
import time

def construct():
    N = 123457
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s limit, though this runs in <0.1s
    
    # Sieve of Eratosthenes to find smallest prime factor
    spf = list(range(N + 1))
    for i in range(2, int(math.isqrt(N)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i

    # Precompute Largest Prime Factor (LPF) for all numbers up to N
    # LPF[1] = 1 by definition
    lpf = [1] * (N + 1)
    if N >= 2:
        lpf[2] = 2
        for i in range(3, N + 1):
            p = spf[i]
            j = i // p
            # lpf[i] is max of p and lpf[j]
            # Since p is the smallest prime factor, p <= any other prime factor of j?
            # Not necessarily. Example: i=6, spf=2, j=3, lpf[3]=3. max(2,3)=3. Correct.
            # Example: i=12, spf=2, j=6, lpf[6]=3. max(2,3)=3. Correct.
            # Example: i=45, spf=3, j=15, lpf[15]=5. max(3,5)=5. Correct.
            # The largest prime factor of i is either p or the largest prime factor of i/p.
            # Wait, is p always <= lpf[i/p]?
            # No. i=10, p=2, j=5, lpf[5]=5. max(2,5)=5.
            # i=9, p=3, j=3, lpf[3]=3. max(3,3)=3.
            # i=25, p=5, j=5, lpf[5]=5. max(5,5)=5.
            # The logic holds: LPF(i) = max(spf(i), LPF(i/spf(i)))
            lpf[i] = p if p > lpf[j] else lpf[j]

    best_a = 1
    best_b = N - 1
    best_val = max(lpf[best_a], lpf[best_b])
    
    # We only need to check up to N // 2 because a and b are symmetric
    # a ranges from 1 to N//2
    for a in range(1, N // 2 + 1):
        if time.time() > deadline:
            break
        b = N - a
        current_max = lpf[a] if lpf[a] > lpf[b] else lpf[b]
        if current_max < best_val:
            best_val = current_max
            best_a = a
            best_b = b
            # Optional: Early exit if we find a very small value?
            # The minimum possible is 1 (if N=2, 1+1). Here N is odd, so one is even, one is odd.
            # P(even) >= 2. P(odd) >= 3 (unless 1).
            # So min max is at least 2?
            # If a=1, P(a)=1. b=N-1. P(b) is large.
            # If a=2, P(a)=2.
            # If we find a solution with max prime factor 2, it's impossible for odd N > 2.
            # If we find 3, it means one is power of 2, other power of 3?
            # 2^k + 3^m = 123457.
            # We can just let it run, it's fast.
            
    return [best_a, best_b]
