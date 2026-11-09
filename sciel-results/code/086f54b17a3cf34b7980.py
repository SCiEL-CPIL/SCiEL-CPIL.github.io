def construct():
    import time
    from math import gcd

    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit

    N = 256

    # Build list of candidate numbers: 1 and all primes <= N
    # Any composite number c with prime factorization p1^a1 * ... * pk^ak
    # can be replaced by the individual prime powers, each of which is coprime
    # to all other prime powers and sums to strictly more than c (for k >= 2)
    # or equals c (for k=1, i.e., prime powers).
    # Thus the optimal set consists of 1 and the highest power of each prime <= N.

    # Sieve of Eratosthenes for primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = False
    if N >= 1:
        is_prime[1] = False
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False

    primes = [i for i in range(2, N + 1) if is_prime[i]]

    # For each prime, compute its highest power <= N
    prime_powers = []
    for p in primes:
        power = p
        while power * p <= N:
            power *= p
        prime_powers.append(power)

    # The solution: 1 plus all highest prime powers
    # Each element is a power of a distinct prime, so any two are coprime.
    # gcd(1, anything) = 1, so 1 is coprime to all others.
    solution = [1] + prime_powers

    # Verify pairwise coprimality (sanity check)
    n = len(solution)
    for i in range(n):
        for j in range(i + 1, n):
            if gcd(solution[i], solution[j]) != 1:
                # This should never happen; fall back to just primes
                return [1] + primes

    # Convert to Python ints
    return [int(x) for x in solution]
