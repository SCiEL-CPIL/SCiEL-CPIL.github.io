def construct():
    import time
    from math import gcd

    start_time = time.time()
    deadline = start_time + 55

    N = 256

    # Sieve of Eratosthenes
    is_prime = [True] * (N + 1)
    is_prime[0] = False
    if N >= 1:
        is_prime[1] = False
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False

    primes = [i for i in range(2, N + 1) if is_prime[i]]

    # Calculate max power for each prime
    prime_powers = {}
    for p in primes:
        power = p
        while power * p <= N:
            power *= p
        prime_powers[p] = power

    # Base solution: 1 + all prime powers
    # We will try to improve by combining primes
    
    # Identify primes that are "small" enough to potentially combine
    # A prime p can combine with q if P_max(p) * q <= N (roughly)
    # Or more generally, if there exists a multiple of p and q <= N that is better
    
    # Build a graph of potential combinations
    # Node: Prime index
    # Edge: Weight = Gain
    
    n_primes = len(primes)
    # We only care about matching small primes with others
    # Primes > sqrt(N) cannot be combined with any other prime power > 1 
    # because p * q > N if p,q > sqrt(N).
    # Actually, if p > sqrt(N), p^2 > N.
    # So p can only be combined with q if p*q <= N.
    # This implies q <= N/p < sqrt(N).
    # So only "small" primes (<= sqrt(N)) can be in a combination with >1 primes.
    # Primes > sqrt(N) are always alone (as P_max(p)=p).
    
    sqrt_N = int(N**0.5)
    small_primes = [p for p in primes if p <= sqrt_N]
    large_primes = [p for p in primes if p > sqrt_N]
    
    # For each pair of small primes, or small and large, check if combining is beneficial
    # We can combine any two primes p, q if max_multiple(p, q) > P_max(p) + P_max(q)
    
    def max_multiple(p, q):
        # Find max k such that k is multiple of p and q, k <= N
        # k is a multiple of lcm(p, q) = p*q (since p,q distinct primes)
        # So k = m * p * q
        # Max m such that m * p * q <= N
        m = N // (p * q)
        if m == 0:
            return 0
        return m * p * q

    # We want to find a matching on the set of primes involved in beneficial combinations
    # But we can also combine 3 primes?
    # p*q*r <= N.
    # This is more complex.
    # Given N=256, max product of 3 distinct primes:
    # 2*3*5 = 30. 2*3*7 = 42.
    # Max value: 2^a 3^b 5^c.
    # We checked this earlier. 140 was max for 2,5,7.
    # Gain for 2,5,7: 140 - (128+25+49) = -62.
    # So 3-prime combinations are likely not beneficial for N=256.
    # We will stick to pairs.
    
    # Build list of beneficial pairs (i, j, gain)
    # i, j are indices in 'primes'
    # We only consider pairs where at least one prime is small?
    # If both are large, p*q > N, so gain is negative (0 - (p+q)).
    # So at least one must be small.
    
    beneficial_pairs = []
    for i in range(n_primes):
        p = primes[i]
        if p > sqrt_N:
            continue
        for j in range(i + 1, n_primes):
            q = primes[j]
            # If q is large, p*q might be <= N
            # If q is small, p*q might be <= N
            
            mult = max_multiple(p, q)
            if mult == 0:
                continue
            
            gain = mult - (prime_powers[p] + prime_powers[q])
            if gain > 0:
                beneficial_pairs.append((i, j, gain))

    # Now we have a set of edges. We want Max Weight Matching.
    # The graph is small (primes <= 256 is ~54 primes).
    # We can use a simple greedy or DP if the graph is sparse.
    # Given the structure, the graph is likely sparse.
    
    # Greedy approach: Sort by gain descending, pick if disjoint.
    # This is not optimal for general graphs, but might be good enough.
    # Or use a simple backtracking if the number of candidates is small.
    
    # Let's filter to only primes that are part of at least one beneficial pair
    involved_indices = set()
    for i, j, g in beneficial_pairs:
        involved_indices.add(i)
        involved_indices.add(j)
        
    # If no pairs, return base
    if not involved_indices:
        solution = [1] + [prime_powers[p] for p in primes]
        return [int(x) for x in solution]

    # Sort pairs by gain
    beneficial_pairs.sort(key=lambda x: x[2], reverse=True)
    
    # Greedy matching
    matched = set()
    selected_pairs = []
    for i, j, g in beneficial_pairs:
        if i not in matched and j not in matched:
            selected_pairs.append((i, j))
            matched.add(i)
            matched.add(j)
            
    # Construct solution
    solution = [1]
    for idx in range(n_primes):
        if idx in matched:
            continue # Handled by pair
        solution.append(prime_powers[primes[idx]])
        
    for i, j in selected_pairs:
        p = primes[i]
        q = primes[j]
        mult = max_multiple(p, q)
        solution.append(mult)
        
    # Verify pairwise coprimality
    n = len(solution)
    for i in range(n):
        for j in range(i + 1, n):
            if gcd(solution[i], solution[j]) != 1:
                # Fallback
                return [1] + [prime_powers[p] for p in primes]
                
    return [int(x) for x in solution]
