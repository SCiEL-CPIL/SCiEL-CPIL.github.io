import time
import math

def construct():
    start_time = time.time()
    MAX_TIME = 85.0 # Safety margin for 90s limit
    
    # Precompute Smallest Prime Factor (SPF) up to 256 * 256
    LIMIT = 256 * 256
    spf = list(range(LIMIT + 1))
    for i in range(2, int(math.isqrt(LIMIT)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, LIMIT + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def factorize(num):
        """Returns dictionary of prime factors and exponents for num."""
        if num == 1:
            return {}
        factors = {}
        while num > 1:
            p = spf[num]
            count = 0
            while num % p == 0:
                num //= p
                count += 1
            factors[p] = count
        return factors
    
    def get_divisors(factors):
        """Generates all divisors from a factorization dictionary."""
        divs = [1]
        for p, e in factors.items():
            new_divs = []
            for d in divs:
                for k in range(e + 1):
                    new_divs.append(d * (p ** k))
            divs = new_divs
        return divs

    answers = []
    for n in range(3, 257):
        if time.time() - start_time > MAX_TIME:
            answers.append(None)
            continue
            
        found = None
        # x must be > n/4 and < 3n/4 (approx)
        # 1/x < 4/n => x > n/4
        # 1/x > 4/n - 2/n (since 1/y, 1/z < 1/x) => 1/x > 2/n => x < n/2? No.
        # 1/x + 1/y + 1/z = 4/n. Since y,z > x, 3/x > 4/n => x < 3n/4.
        x_min = n // 4 + 1
        x_max = (3 * n) // 4 + 1
        
        for x in range(x_min, x_max + 1):
            if time.time() - start_time > MAX_TIME:
                answers.append(None)
                break
                
            A = 4 * x - n
            B = n * x
            if A <= 0:
                continue
                
            # We need divisors of B^2
            # Factor B
            b_factors = factorize(B)
            # Factor B^2: double exponents
            b2_factors = {}
            for p, e in b_factors.items():
                b2_factors[p] = 2 * e
            
            divisors = get_divisors(b2_factors)
            
            for d in divisors:
                # Check if Ay - B = d has integer solution y
                # y = (B + d) / A
                if (B + d) % A == 0:
                    y = (B + d) // A
                    if y > x:
                        # Check if z is integer and z > y
                        # z = (n * x * y) / d
                        num_z = n * x * y
                        if num_z % d == 0:
                            z = num_z // d
                            if z > y:
                                found = [x, y, z]
                                break
            if found:
                break
        
        answers.append(found)
        
    return answers
