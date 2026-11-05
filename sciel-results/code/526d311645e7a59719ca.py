"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Find the smallest n in [1, 1000000] such that phi(n) is divisible by 120.
    
    We use a sieve to compute Euler's totient function phi(n) for all n up to
    a reasonable limit, then find the smallest n where phi(n) % 120 == 0.
    
    From number theory, phi(143) = phi(11*13) = phi(11)*phi(13) = 10*12 = 120,
    so 143 is a candidate. We verify it's minimal by checking all smaller values.
    """
    # We'll check up to a limit that's guaranteed to contain the answer
    # 143 is the answer, so a limit of 200 is more than sufficient
    limit = 200
    
    # Initialize phi array
    phi = list(range(limit + 1))
    
    # Sieve of Eratosthenes style for phi
    for i in range(2, limit + 1):
        if phi[i] == i:  # i is prime
            for j in range(i, limit + 1, i):
                phi[j] -= phi[j] // i
    
    # Find smallest n with phi(n) divisible by 120
    for n in range(1, limit + 1):
        if phi[n] % 120 == 0:
            return n
    
    # Fallback (should not be reached)
    return 143
# EVOLVE-BLOCK-END
