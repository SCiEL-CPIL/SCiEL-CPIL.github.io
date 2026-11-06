def _search():
    """Optimized search using pre-computed shifted prime power sets."""
    deadline = time.time() + 55
    
    # Generate prime powers <= 256
    is_prime = [True] * 257
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(256)) + 1):
        if is_prime[i]:
            for j in range(i*i, 257, i):
                is_prime[j] = False
    primes = [i for i in range(2, 257) if is_prime[i]]
    
    prime_powers = set()
    for p in primes:
        val = p
        while val <= 256:
            prime_powers.add(val)
            val *= p
            
    # Pre-compute shifted sets
    shifted_pp = {}
    for b in range(1, 129):
        shifted_pp[b] = {p - b for p in prime_powers if p > b}
        
    best_score = 0
    best_A = [1, 2, 3]
    best_B = [1, 2, 3]
    
    # Grid search over B
    for b1 in range(1, 20):
        for b2 in range(b1+1, 40):
            for b3 in range(b2+1, 60):
                B = [b1, b2, b3]
                
                valid_as = list(shifted_pp[b1] & shifted_pp[b2] & shifted_pp[b3])
                valid_as = [a for a in valid_as if 1 <= a <= 128]
                
                if len(valid_as) < 3:
                    continue
                    
                valid_as.sort()
                
                for combo in combinations(valid_as, 3):
                    A = list(combo)
                    sums = set()
                    for a in A:
                        for b in B:
                            sums.add(a + b)
                    
                    if len(sums) == 9:
                        if _pairwise_coprime(list(sums)):
                            return A, B
                            
    # Fallback random search
    rng = random.Random(42)
    for _ in range(10000):
        if time.time() > deadline:
            break
        A = list(rng.sample(range(1, 129), 3))
        B = list(rng.sample(range(1, 129), 3))
        score = _eval(A, B)
        if score > best_score:
            best_score = score
            best_A = A[:]
            best_B = B[:]
            if best_score == 9:
                return best_A, best_B
                
    return best_A, best_B
