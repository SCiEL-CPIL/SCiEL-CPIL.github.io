def construct():
    """
    Returns the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    Uses a linear sieve to compute Euler's totient function efficiently.
    """
    limit = 1000000
    # phi[i] will hold Euler's totient of i
    phi = list(range(limit + 1))
    primes = []
    is_composite = [False] * (limit + 1)
    
    for i in range(2, limit + 1):
        if not is_composite[i]:
            primes.append(i)
            phi[i] = i - 1
        for p in primes:
            if i * p > limit:
                break
            is_composite[i * p] = True
            if i % p == 0:
                phi[i * p] = phi[i] * p
                break
            else:
                phi[i * p] = phi[i] * (p - 1)
    
    # Find the smallest n with phi[n] divisible by 120
    for n in range(1, limit + 1):
        if phi[n] % 120 == 0:
            return n
    
    # This should never be reached for the given constraints
    return 1
