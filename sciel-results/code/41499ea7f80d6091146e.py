def construct():
    n = 256
    candidates = list(range(2, n + 1))
    
    # Generate primes
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
    
    primes = sieve(n)
    
    def is_compatible(a, S):
        # Check if a can be added to S
        # For all distinct b, c in S:
        # 1. a does not divide b*c
        # 2. b does not divide a*c  (Note: b is in S, a is new, c is in S)
        # 3. c does not divide a*b
        # We can optimize by precomputing or just iterating.
        for i in range(len(S)):
            b = S[i]
            for j in range(i + 1, len(S)):
                c = S[j]
                if (b * c) % a == 0:
                    return False
                if (a * c) % b == 0:
                    return False
                if (a * b) % c == 0:
                    return False
        return True

    def is_valid_set(S):
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                for k in range(j + 1, len(S)):
                    a, b, c = S[i], S[j], S[k]
                    if (b * c) % a == 0:
                        return False
        return True

    import random
    best = []
    
    # Candidate 1: All primes
    if is_valid_set(primes):
        best = primes[:]
        
    # Candidate 2: Greedy with various orderings
    orderings = [
        list(range(2, n + 1)),
        list(range(n, 1, -1)),
        sorted(range(2, n + 1), key=lambda x: -x.bit_count()),
        sorted(range(2, n + 1), key=lambda x: x.bit_count()),
        primes, # Ensure primes are tried in greedy too (will result in primes)
    ]
    
    # Add some random orderings
    for _ in range(50):
        rand_order = candidates[:]
        random.shuffle(rand_order)
        orderings.append(rand_order)
        
    for order in orderings:
        S = []
        for a in order:
            if is_compatible(a, S):
                S.append(a)
        if len(S) > len(best):
            best = S
    
    # Local search / Hill Climbing to try to improve best
    # Try to swap elements to see if we can enable adding more elements
    # This is complex, so we'll just do a few random restarts with greedy + local repair
    # If best is 54, we are likely optimal.
    
    # One more check: Can we add any number to best?
    added = True
    while added:
        added = False
        for a in candidates:
            if a not in best:
                if is_compatible(a, best):
                    best.append(a)
                    added = True
    
    # Verify final result
    if not is_valid_set(best):
        raise ValueError("Final set is invalid")
    
    return list(best)
