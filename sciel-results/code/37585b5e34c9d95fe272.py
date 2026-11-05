def construct():
    """
    Return the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    Uses a sieve to compute Euler's totient function for a range of integers.
    """
    # The minimal solution is known to be small (143).
    # We search up to a safe limit to ensure we find it.
    limit = 10000
    
    # Initialize phi array with phi[i] = i
    phi = list(range(limit + 1))
    
    # Sieve for phi
    for i in range(2, limit + 1):
        if phi[i] == i:  # i is prime
            for j in range(i, limit + 1, i):
                phi[j] -= phi[j] // i
                
    # Find the smallest n with phi[n] % 120 == 0
    for n in range(1, limit + 1):
        if phi[n] % 120 == 0:
            return n
            
    # Fallback if not found in the small range (should not happen)
    # Expand search if necessary, but 143 is well within 10000
    return -1
