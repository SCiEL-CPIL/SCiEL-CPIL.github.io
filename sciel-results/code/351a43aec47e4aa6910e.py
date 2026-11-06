import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55 seconds to be safe

    N = 123457
    
    # Strategy: Precompute Largest Prime Factor (LPF) for all numbers up to N.
    # This allows O(1) lookup for P(a) and P(b).
    # Then iterate a from 1 to N//2, compute score = max(lpf[a], lpf[N-a]).
    # This is an exhaustive search over all valid pairs (a, N-a) with a <= N-a.
    
    lpf = [0] * (N + 1)
    
    # Sieve to compute LPF
    # lpf[i] will be the largest prime factor of i.
    # For i from 2 to N:
    #   If lpf[i] == 0, i is prime, so lpf[i] = i.
    #   For each multiple j of i (starting from i*i? or i?), if lpf[j] is not set to a larger prime, set it.
    # Actually, a more efficient way:
    # Iterate p from 2 to N.
    # If lpf[p] == 0, p is prime.
    # For j in range(p, N+1, p):
    #     lpf[j] = p
    # Since we iterate p in increasing order, the last assignment to lpf[j] will be from the largest prime factor.
    # Wait, no. If p is a prime factor of j, we set lpf[j] = p.
    # If we iterate p from 2 to N, and for each prime p, we update all multiples,
    # then for a number j with prime factors p1 < p2 < ... < pk,
    # lpf[j] will first be set to p1, then p2, ..., finally pk.
    # So lpf[j] will correctly be the largest prime factor.
    
    # Complexity: Sum_{p prime} N/p ~ N * sum_{p} 1/p ~ N log log N.
    # For N=123457, this is fast.
    
    for p in range(2, N + 1):
        if lpf[p] == 0: # p is prime
            for j in range(p, N + 1, p):
                lpf[j] = p
                
    # Now lpf[i] holds the largest prime factor of i for all i >= 2.
    # lpf[1] = 1 by definition (P(1)=1).
    lpf[1] = 1
    
    best_val = float('inf')
    best_pair = [1, N]
    
    # Iterate a from 1 to N//2
    # We want to minimize max(lpf[a], lpf[N-a])
    
    for a in range(1, N // 2 + 1):
        b = N - a
        score = max(lpf[a], lpf[b])
        
        if score < best_val:
            best_val = score
            best_pair = [a, b]
            
        # Early termination if we found a score of 2?
        # The smallest possible score is 2 (if both a and b are powers of 2, but they sum to an odd number, so one must be odd.
        # If a is a power of 2, P(a)=2. If b is odd, P(b)>=3. So min score is at least 3?
        # If a=1, P(a)=1. b=N. P(N)=123457 (if prime).
        # Let's just run the loop. It's O(N) which is ~120k iterations. Very fast.
        
        if time.time() > deadline:
            break
            
    return best_pair
