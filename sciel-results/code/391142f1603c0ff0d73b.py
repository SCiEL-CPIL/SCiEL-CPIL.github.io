"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Finds the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    Uses a sieve to compute Euler's totient function for all numbers up to 1,000,000.
    """
    LIMIT = 1000000
    # Initialize phi[i] = i
    phi = list(range(LIMIT + 1))
    
    # Sieve for totient
    # Only need to iterate i up to sqrt(LIMIT)
    i = 2
    while i * i <= LIMIT:
        if phi[i] == i:  # i is prime
            # Update all multiples of i
            for j in range(i, LIMIT + 1, i):
                phi[j] -= phi[j] // i
        i += 1
        
    # Find the smallest n with phi(n) % 120 == 0
    for n in range(1, LIMIT + 1):
        if phi[n] % 120 == 0:
            return n
            
    # Fallback (should never happen)
    return 120 * 120
# EVOLVE-BLOCK-END
