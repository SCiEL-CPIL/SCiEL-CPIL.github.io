import math
import time

def construct():
    N = 123457
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 60s limit
    
    # Pre-compute largest prime factor for all numbers up to N
    # Using a segmented-like sieve approach for LPF
    # lpf[i] will hold the largest prime factor of i
    lpf = [0] * (N + 1)
    lpf[1] = 1
    
    # Sieve of Eratosthenes variant to find LPF
    # For each prime p, update lpf for its multiples
    # We only need to consider primes up to N
    # Optimization: we can stop when p*p > N for the inner loop if we were just factoring,
    # but for LPF we need to mark all multiples.
    # Actually, a standard way is:
    # for i in range(2, N+1):
    #   if lpf[i] == 0: # i is prime
    #     for j in range(i, N+1, i):
    #       lpf[j] = i # Since we iterate i in increasing order, the last prime divisor is the largest
    
    # This is O(N log log N) roughly, which is very fast for N=123457.
    
    # Let's implement it efficiently
    for i in range(2, N + 1):
        if lpf[i] == 0:  # i is prime
            for j in range(i, N + 1, i):
                lpf[j] = i
    
    # Now lpf[n] contains the largest prime factor of n for all n in [1, N]
    # P(1) = 1, lpf[1] = 1. Correct.
    
    # We want to minimize max(lpf[a], lpf[N-a]) for a in [1, N-1]
    # Since lpf is precomputed, this is a single linear scan.
    # To optimize, we can iterate a from 1 to N//2 and check both a and N-a.
    # Due to symmetry, max(lpf[a], lpf[N-a]) is the same for (a, N-a) and (N-a, a).
    # So we just need to find the minimum value of max(lpf[a], lpf[N-a]).
    
    best_val = float('inf')
    best_a = 1
    best_b = N - 1
    
    # We can potentially prune if lpf[a] is already >= best_val, but lpf[N-a] might be small.
    # However, max(lpf[a], lpf[N-a]) >= lpf[a]. So if lpf[a] >= best_val, this pair cannot improve.
    # Same for lpf[N-a].
    # So if min(lpf[a], lpf[N-a]) >= best_val, we can skip? No, max could be small if both are small.
    # Wait, max(x, y) >= x and max(x, y) >= y.
    # So if lpf[a] >= best_val, then max(lpf[a], lpf[N-a]) >= lpf[a] >= best_val. No improvement.
    # Similarly if lpf[N-a] >= best_val.
    # So we only need to check if lpf[a] < best_val AND lpf[N-a] < best_val.
    # If either is >= best_val, the max will be >= best_val.
    
    for a in range(1, N // 2 + 1):
        b = N - a
        pa = lpf[a]
        pb = lpf[b]
        
        # If either factor is already >= current best, the max will be >= current best
        if pa >= best_val or pb >= best_val:
            continue
            
        current_max = pa if pa > pb else pb
        if current_max < best_val:
            best_val = current_max
            best_a = a
            best_b = b
            
        # Early termination: if best_val == 2, that's the minimum possible for any composite number > 1.
        # P(1)=1, P(2)=2.
        # Can we get max=1? Only if a=1 and b=1, but a+b=123457. No.
        # Can we get max=2? This requires both a and b to be powers of 2 (or 1).
        # Since N is odd, one of a,b is even, one is odd.
        # If max=2, both must be 2-smooth.
        # 2-smooth numbers are powers of 2.
        # So a=2^x, b=2^y. 2^x + 2^y = 123457.
        # Since 123457 is odd, one term must be 1 (2^0) and the other must be 123456.
        # Is 123456 a power of 2?
        # 2^16 = 65536, 2^17 = 131072. No.
        # So max=2 is impossible.
        # Next is 3.
        # If best_val becomes 3, we can stop?
        # Min possible P(n) for n>1 is 2. But we established max=2 is impossible.
        # So 3 is the next candidate.
        # If we find a pair with max=3, it is optimal.
        if best_val == 3:
            break

    return [best_a, best_b]
