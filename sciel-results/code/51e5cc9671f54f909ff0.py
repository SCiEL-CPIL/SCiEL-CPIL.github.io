def construct():
    MAX_DENOM = 10**12
    answers = []
    
    # Precompute primes up to 500
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    primes = get_primes(500)
    
    def factorize(num, primes):
        factors = {}
        temp = num
        for p in primes:
            if p * p > temp:
                break
            while temp % p == 0:
                factors[p] = factors.get(p, 0) + 1
                temp //= p
        if temp > 1:
            factors[temp] = factors.get(temp, 0) + 1
        return factors

    def get_divisors_from_factors(factors):
        if not factors:
            return [1]
        primes = list(factors.keys())
        divisors = [1]
        for p in primes:
            e = factors[p]
            new_divs = []
            for d in divisors:
                for k in range(e + 1):
                    new_divs.append(d * (p ** k))
            divisors = new_divs
        return divisors

    for n in range(3, 257):
        found = None
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        for x in range(x_min, x_max + 1):
            A = 4 * x - n
            if A <= 0:
                continue
            
            M = n * x
            M_sq = M * M
            
            # Factor M to get divisors of M^2
            fac_M = factorize(M, primes)
            # Exponents in M^2 are 2 * exponents in M
            fac_M_sq = {p: 2*e for p, e in fac_M.items()}
            
            divisors = get_divisors_from_factors(fac_M_sq)
            
            for U in divisors:
                # Check y integrality: A | (U + M)
                if (U + M) % A != 0:
                    continue
                
                V = M_sq // U
                # Check z integrality: A | (V + M)
                if (V + M) % A != 0:
                    continue
                
                y = (U + M) // A
                z = (V + M) // A
                
                if y <= 0 or z <= 0:
                    continue
                
                if y > MAX_DENOM or z > MAX_DENOM:
                    continue
                
                # Distinctness
                if x == y or x == z or y == z:
                    continue
                
                # Verify
                # 1/x + 1/y + 1/z == 4/n
                # n(yz + xz + xy) == 4xyz
                if n * (y * z + x * z + x * y) == 4 * x * y * z:
                    found = [x, y, z]
                    break
            
            if found:
                break
        
        answers.append(found)
    
    return answers
