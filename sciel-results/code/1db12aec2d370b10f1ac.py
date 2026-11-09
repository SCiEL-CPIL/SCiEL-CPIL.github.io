import math
import time

def construct():
    answers = [None] * 254
    deadline = time.time() + 85  # Safety margin before 90s kill
    
    def get_prime_factors(n):
        factors = {}
        d = 2
        while d * d <= n:
            if n % d:
                d += 1
            else:
                nf = 0
                while not n % d:
                    n //= d
                    nf += 1
                factors[d] = nf
        if n > 1:
            factors[n] = factors.get(n, 0) + 1
        return factors

    def generate_divisors(factors):
        divs = [1]
        for p, e in factors.items():
            new_divs = []
            for d in divs:
                for i in range(e + 1):
                    new_divs.append(d * (p ** i))
            divs = new_divs
        return divs

    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        found = None
        # x must be > n/4 and < 3n/4
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
            A = 4 * x - n
            B = n * x
            
            # We need (Ay - B)(Az - B) = B^2
            # Let u = Ay - B. Then u divides B^2.
            # y = (u + B) / A must be integer.
            # z = (B^2/u + B) / A must be integer.
            # We assume y < z, which implies u < B^2/u, so u < B.
            
            # Instead of enumerating all divisors of B^2, 
            # iterate u in arithmetic progression: u ≡ -B (mod A)
            # u = k*A - B for k = 1, 2, 3, ...
            # u must be positive and < B (for y < z)
            # So k*A - B > 0 => k > B/A
            # And k*A - B < B => k < 2B/A
            
            # Factorize B to check if u divides B^2
            factors_B = get_prime_factors(B)
            
            # For efficiency, precompute the prime factorization info
            # to quickly check if a number divides B^2
            
            def divides_B2(u):
                """Check if u divides B^2 using precomputed factors of B."""
                temp = u
                for p, e in factors_B.items():
                    max_exp = 2 * e
                    count = 0
                    while temp % p == 0:
                        temp //= p
                        count += 1
                        if count > max_exp:
                            return False
                return temp == 1
            
            # Iterate k such that u = k*A - B is in (0, B)
            # k*A - B > 0 => k >= floor(B/A) + 1
            # k*A - B < B => k <= ceil(2B/A) - 1 (approximately)
            k_min = B // A + 1
            k_max = (2 * B) // A + 1
            
            for k in range(k_min, k_max + 1):
                if time.time() > deadline:
                    break
                u = k * A - B
                if u <= 0 or u >= B:
                    continue
                
                if not divides_B2(u):
                    continue
                
                # u divides B^2, so v = B^2 / u is integer
                v = (B * B) // u
                
                # Check that (v + B) is divisible by A
                if (v + B) % A != 0:
                    continue
                
                y = (u + B) // A
                z = (v + B) // A
                
                # Check distinctness
                if x == y or x == z or y == z:
                    continue
                
                # Denominator bound check
                if x > 10**12 or y > 10**12 or z > 10**12:
                    continue
                
                found = [x, y, z]
                break
            
            if found:
                break
        
        answers[idx] = found
    
    return answers
