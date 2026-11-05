import time
from math import gcd

def construct():
    answers = []
    deadline = time.time() + 50  # 50s safety margin
    
    # Precompute smallest prime factor (SPF) for fast factorization
    MAX_B = 256 * 192  # max n * x where x <= 3n/4
    spf = list(range(MAX_B + 1))
    for i in range(2, int(MAX_B ** 0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, MAX_B + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def factorize_spf(num):
        """Factorize using precomputed SPF table."""
        factors = {}
        while num > 1:
            p = spf[num]
            count = 0
            while num % p == 0:
                count += 1
                num //= p
            factors[p] = count
        return factors

    def get_divisors_from_factors(factors):
        """Generate all divisors from prime factorization."""
        divisors = [1]
        for p, e in factors.items():
            new_divs = []
            p_power = 1
            for _ in range(e + 1):
                for d in divisors:
                    new_divs.append(d * p_power)
                p_power *= p
            divisors = new_divs
        return divisors

    for n in range(3, 257):
        found = None
        # x must satisfy n/4 < x < 3n/4
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
                
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x
            
            # We need (Ay - B)(Az - B) = B^2
            # Let u = Ay - B, v = Az - B
            # u * v = B^2
            # y = (u + B) / A
            # z = (v + B) / A
            
            # Factorize B using SPF
            b_factors = factorize_spf(B)
            b_sq_factors = {p: 2*e for p, e in b_factors.items()}
            
            # Count divisors of B^2 - if too many, skip this x
            num_divs = 1
            for e in b_sq_factors.values():
                num_divs *= (e + 1)
            if num_divs > 10000:
                continue
            
            divs = get_divisors_from_factors(b_sq_factors)
            
            # Sort divisors to try smaller y first (more likely to be within bounds)
            divs.sort()
            
            for u in divs:
                # Check if y is integer
                if (u + B) % A != 0:
                    continue
                y = (u + B) // A
                
                # Early exit: if y is too large, larger u will give larger y
                if y > 10**12:
                    break
                
                # Calculate v
                v = (B * B) // u
                
                # Check if z is integer
                if (v + B) % A != 0:
                    continue
                z = (v + B) // A
                
                # Check distinctness and bounds
                if y == z:
                    continue
                if y == x or z == x:
                    continue
                if z > 10**12:
                    continue
                
                # We found a valid solution
                found = [x, y, z]
                break
            
            if found:
                break
        
        answers.append(found)
        
    return answers
