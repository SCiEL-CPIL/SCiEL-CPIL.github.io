"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def construct():
    # We need the smallest n in [1, 1000000] such that phi(n) % 120 == 0.
    # 120 = 2^3 * 3 * 5.
    # phi(n) is divisible by 120 iff it is divisible by 8, 3, and 5.
    
    # Let's compute phi(n) for n from 1 upwards until we find one.
    # To do this efficiently, we can precompute the smallest prime factor (SPF)
    # or simply use a sieve-like approach to compute phi(n) for all n up to a limit.
    # Since we want the *smallest* n, we can just iterate n=1,2,3,... and compute phi(n).
    # However, computing phi(n) from scratch for each n is O(sqrt(n)).
    # A better way: Precompute phi(n) for all n up to some limit using a sieve.
    # The limit doesn't need to be huge. Let's guess the answer is small.
    # phi(n) grows roughly as n. So if phi(n) ~ 120, n ~ 120/0.6 ~ 200.
    # Let's just precompute phi up to 10000 to be safe, then scan.
    
    # Actually, we can just compute phi(n) on the fly for n=1,2,...
    # But let's use a linear sieve or simple sieve to precompute phi up to N.
    # Let N = 10000.
    
    N = 10000
    phi = list(range(N + 1))
    for i in range(2, N + 1):
        if phi[i] == i:  # i is prime
            for j in range(i, N + 1, i):
                phi[j] -= phi[j] // i
    
    for n in range(1, N + 1):
        if phi[n] % 120 == 0:
            return n
    
    # If not found in 10000, extend search (though unlikely)
    # For robustness, we could continue, but the task says minimize n,
    # and the first one is likely small.
    # Let's just return a known valid number if we somehow miss it,
    # but the loop above should find it.
    # Just in case, return 120*120 as a fallback (which is valid but not minimal).
    return 14400

# EVOLVE-BLOCK-END
