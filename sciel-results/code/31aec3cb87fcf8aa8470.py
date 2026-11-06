import time
import math

def construct():
    deadline = time.time() + 85.0
    answers = [None] * 254
    BOUND = 10**12
    
    # Precompute smallest prime factor for fast factorization
    max_val = 500000 # Enough for B = n*x
    spf = list(range(max_val + 1))
    for i in range(2, int(math.isqrt(max_val)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, max_val + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def get_factors(n):
        factors = {}
        while n > 1:
            p = spf[n]
            count = 0
            while n % p == 0:
                n //= p
                count += 1
            factors[p] = count
        return factors

    def generate_divisors(factors):
        divs = [1]
        for p, e in factors.items():
            current_divs = []
            for d in divs:
                val = d
                for _ in range(e + 1):
                    current_divs.append(val)
                    val *= p
            divs = current_divs
        return divs

    for n in range(3, 257):
        idx = n - 3
        if time.time() > deadline:
            break
            
        found = None
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        # Iterate x
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
                
            A = 4 * x - n
            if A <= 0:
                continue
                
            B = n * x
            B_sq = B * B
            
            # Factorize B
            fac_B = get_factors(B)
            # Factorize B^2: exponents doubled
            fac_B_sq = {p: 2 * e for p, e in fac_B.items()}
            
            # Generate divisors of B^2
            divisors = generate_divisors(fac_B_sq)
            
            # We need d | B^2, d < B (to ensure y < z and d != B)
            # y = B + d, z = B + B^2/d
            # Distinctness: x != y, x != z, y != z (guaranteed by d != B)
            
            for d in divisors:
                if d >= B:
                    continue
                y = B + d
                z = B + B_sq // d
                
                if x == y or x == z:
                    continue
                if y > BOUND or z > BOUND:
                    continue
                
                found = [x, y, z]
                break
            
            if found:
                break
                
        answers[idx] = found
        
    return answers
