def construct():
    """
    Return the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    We compute phi(n) for all n up to a sufficient bound and return the smallest valid n.
    The smallest such n is known to be 240 (phi(240) = 64, wait no).
    Let's actually compute it properly.
    
    phi(n) divisible by 120 = 2^3 * 3 * 5.
    
    Let's just compute phi for all n up to 1000000 using a sieve and find the min.
    """
    N = 1000000
    phi = list(range(N + 1))
    
    # Sieve of Eratosthenes style for totient
    for i in range(2, N + 1):
        if phi[i] == i:  # i is prime
            for j in range(i, N + 1, i):
                phi[j] -= phi[j] // i
    
    # Find the smallest n with phi[n] % 120 == 0
    for n in range(1, N + 1):
        if phi[n] % 120 == 0:
            return n
    
    # Should not reach here for N=1000000
    return N
