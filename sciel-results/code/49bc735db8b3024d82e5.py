"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct():
    N = 123457
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin before 90s limit
    
    # Precompute largest prime factor for all numbers up to N
    lpf = [0] * (N + 1)
    lpf[1] = 1
    
    # Sieve of Eratosthenes variant to find largest prime factor
    for i in range(2, N + 1):
        if lpf[i] == 0:  # i is prime
            for j in range(i, N + 1, i):
                lpf[j] = i
    
    best_a = 1
    best_b = N - 1
    best_cost = max(lpf[1], lpf[best_b])
    
    # Search for the split a+b=N that minimizes max(P(a), P(b))
    # We only need to check a up to N//2 because the problem is symmetric
    half_n = N // 2
    
    # Optimization: Check a=1 first, then iterate. 
    # If we find cost=2, it's the minimum possible (since P(n)>=2 for n>1, and one of a,b must be >1).
    # Actually, if a=1, P(a)=1. P(b) must be minimized.
    # If we find a split where max(P(a), P(b)) == 2, we can stop.
    # This happens if both a and b are powers of 2? No, a+b is odd.
    # If a is power of 2, P(a)=2. If b is power of 2, P(b)=2.
    # Sum of two powers of 2 is even (if both > 0). But N is odd.
    # So one must be odd. P(odd) >= 3.
    # So min cost is at least 3.
    # If we find cost=3, we can stop?
    # P(a)=3 implies a is 3^k * 2^j? No, P(a)=3 means all prime factors are 2 or 3.
    # P(b)=3 means all prime factors are 2 or 3.
    # If we find cost=3, it's likely optimal or near optimal.
    
    for a in range(1, half_n + 1):
        if time.time() > deadline:
            break
            
        b = N - a
        cost = max(lpf[a], lpf[b])
        if cost < best_cost:
            best_cost = cost
            best_a = a
            best_b = b
            if best_cost == 3:
                break
            if best_cost == 2:
                break
                
    return [best_a, best_b]
# EVOLVE-BLOCK-END
