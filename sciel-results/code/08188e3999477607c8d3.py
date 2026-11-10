def construct():
    MAX_DENOM = 10**12
    answers = []
    
    # Precompute smallest prime factor for numbers up to 1000
    MAX_N = 1000
    spf = list(range(MAX_N + 1))
    for i in range(2, int(MAX_N**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, MAX_N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def factorize(num):
        if num <= 1:
            return {}
        factors = {}
        while num > 1:
            p = spf[num]
            cnt = 0
            while num % p == 0:
                cnt += 1
                num //= p
            factors[p] = cnt
        return factors
    
    def get_divisors_sorted(factors):
        if not factors:
            return [1]
        primes = list(factors.keys())
        divisors = [1]
        for p in primes:
            e = factors[p]
            new_divs = []
            for d in divisors:
                pk = 1
                for k in range(e + 1):
                    new_divs.append(d * pk)
                    pk *= p
            divisors = new_divs
        divisors.sort()
        return divisors
    
    for n in range(3, 257):
        found = None
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        # Pre-factorize n
        fac_n = factorize(n)
        
        for x in range(x_min, x_max + 1):
            A = 4 * x - n
            if A <= 0:
                continue
            
            nx = n * x
            C = nx * nx
            
            fac_x = factorize(x)
            fac_C = {}
            all_primes = set(fac_n.keys()) | set(fac_x.keys())
            for p in all_primes:
                exp = 2 * fac_n.get(p, 0) + 2 * fac_x.get(p, 0)
                if exp > 0:
                    fac_C[p] = exp
            
            divisors = get_divisors_sorted(fac_C)
            
            # We need u * v = C, y = (u + nx)/A, z = (v + nx)/A
            # Assume y <= z, so u <= v, so u <= sqrt(C) = nx
            # Also need y > x (distinct, and x is smallest), so (u + nx)/A > x
            # => u + nx > A*x = (4x-n)*x = 4x^2 - nx
            # => u > 4x^2 - nx - nx = 4x^2 - 2nx
            # Also y < MAX_DENOM: (u + nx)/A < MAX_DENOM => u < A*MAX_DENOM - nx
            # And z < MAX_DENOM: (v + nx)/A < MAX_DENOM => v < A*MAX_DENOM - nx
            
            u_min_needed = max(1, 4*x*x - 2*n*x + 1)
            
            for u in divisors:
                if u > C // u:  # u > v, skip (we want u <= v)
                    break
                if u < u_min_needed:
                    continue
                v = C // u
                if C % u != 0:
                    continue
                
                num_y = u + nx
                num_z = v + nx
                if num_y % A != 0 or num_z % A != 0:
                    continue
                y = num_y // A
                z = num_z // A
                if y <= 0 or z <= 0:
                    continue
                if y > MAX_DENOM or z > MAX_DENOM:
                    continue
                if x == y or x == z or y == z:
                    continue
                # Verify
                lhs = n * (y * z + x * z + x * y)
                rhs = 4 * x * y * z
                if lhs == rhs:
                    found = [x, y, z]
                    break
            if found:
                break
        answers.append(found)
    
    return answers
