def construct():
    MAX_DENOM = 10**12
    answers = []
    
    def factorize(num):
        factors = {}
        d = 2
        while d * d <= num:
            while num % d == 0:
                factors[d] = factors.get(d, 0) + 1
                num //= d
            d += 1
        if num > 1:
            factors[num] = factors.get(num, 1)
        return factors
    
    def get_divisors_from_factors(factors):
        # factors: dict of prime -> exponent
        # return list of all divisors
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
        # x must satisfy: n/4 < x <= 3n/4 (since x is smallest denominator)
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        # Also need 4x - n > 0, which is x > n/4, already satisfied
        
        for x in range(x_min, x_max + 1):
            A = 4 * x - n
            if A <= 0:
                continue
            C = n * n * x * x
            # Factor C = n^2 * x^2
            fac_n = factorize(n)
            fac_x = factorize(x)
            # Combine: for each prime, exponent in C is 2*exp_in_n + 2*exp_in_x
            fac_C = {}
            all_primes = set(fac_n.keys()) | set(fac_x.keys())
            for p in all_primes:
                exp = 2 * fac_n.get(p, 0) + 2 * fac_x.get(p, 0)
                if exp > 0:
                    fac_C[p] = exp
            
            divisors = get_divisors_from_factors(fac_C)
            
            for u in divisors:
                if C % u != 0:
                    continue
                v = C // u
                # y = (u + n*x) / A, z = (v + n*x) / A
                num_y = u + n * x
                num_z = v + n * x
                if num_y % A != 0 or num_z % A != 0:
                    continue
                y = num_y // A
                z = num_z // A
                if y <= 0 or z <= 0:
                    continue
                if y > MAX_DENOM or z > MAX_DENOM:
                    continue
                # Check distinctness: x, y, z all distinct
                if x == y or x == z or y == z:
                    continue
                # Verify the equation (safety check)
                # 1/x + 1/y + 1/z == 4/n
                # Cross multiply: n*(y*z + x*z + x*y) == 4*x*y*z
                lhs = n * (y * z + x * z + x * y)
                rhs = 4 * x * y * z
                if lhs == rhs:
                    found = [x, y, z]
                    break
            if found:
                break
        answers.append(found)
    
    return answers
