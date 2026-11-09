"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    # We need to find the smallest n in [1, 1000000] such that phi(n) % 120 == 0.
    # The search space is small enough to compute phi(n) for all n up to a limit
    # using a sieve. We can stop early if we find the minimum.
    # However, a full sieve up to 1000000 is fast (< 0.1s).
    # Let's just compute the answer directly.
    
    # Based on analysis, the smallest such n is 143.
    # phi(143) = phi(11) * phi(13) = 10 * 12 = 120.
    # 120 is divisible by 120.
    
    # To be absolutely safe and "stronger algorithm" as requested, 
    # we implement the sieve search. It is deterministic and fast.
    
    limit = 1000000
    # Optimization: We know the answer is small. We can search up to a smaller limit
    # first, but 10^6 is fine.
    
    # Sieve for Euler's Totient function
    phi = list(range(limit + 1))
    for i in range(2, limit + 1):
        if phi[i] == i:  # i is prime
            for j in range(i, limit + 1, i):
                phi[j] -= phi[j] // i
                
    # Find the smallest n
    for n in range(1, limit + 1):
        if phi[n] % 120 == 0:
            return n
            
    # Fallback (should not be reached)
    return 143
# EVOLVE-BLOCK-END
