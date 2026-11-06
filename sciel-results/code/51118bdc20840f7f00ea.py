import time
import math

def construct():
    deadline = time.time() + 78.0
    answers = [None] * 254
    BOUND = 10**12
    
    # Precompute primes up to 10000 for fast factorization
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.isqrt(n)) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    small_primes = sieve(10000)
    
    def factorize(num, primes):
        """Factorize num using trial division with precomputed primes."""
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
    
    def generate_divisors(factors):
        """Generate all divisors from prime factorization dict."""
        divisors = [1]
        for prime, exp in factors.items():
            new_divs = []
            pow_val = 1
            for e in range(exp + 1):
                for d in divisors:
                    new_divs.append(d * pow_val)
                pow_val *= prime
            divisors = new_divs
        return divisors
    
    for n in range(3, 257):
        idx = n - 3
        if time.time() > deadline:
            break
        found = None
        
        # x must satisfy: 1/x <= 4/n => x >= n/4
        # and 3/x >= 4/n => x <= 3n/4 (since y,z >= x means 1/y+1/z <= 2/x, so 4/n - 1/x <= 2/x => 4/n <= 3/x)
        x_start = n // 4 + 1
        if n % 4 == 0:
            x_start = n // 4 + 1
        else:
            x_start = n // 4 + 1  # ceil(n/4) when n%4!=0, but let's be safe
        # Actually x > n/4 strictly since 1/x < 4/n for the sum to work with positive y,z
        x_start = (n + 3) // 4  # floor(n/4) + 1 = ceil((n+1)/4)... let me just use n//4 + 1
        x_start = n // 4 + 1
        x_end = (3 * n) // 4
        
        for x in range(x_start, x_end + 1):
            if time.time() > deadline:
                break
            if found:
                break
            
            a = 4 * x - n  # positive since x > n/4
            b = n * x
            
            if a <= 0:
                continue
            
            b_sq = b * b
            
            # Factorize b^2 = (n*x)^2
            # First factorize b = n*x
            b_factors = factorize(b, small_primes)
            # b^2 has doubled exponents
            b_sq_factors = {}
            for p, e in b_factors.items():
                b_sq_factors[p] = e * 2
            
            # Generate divisors of b^2
            divisors = generate_divisors(b_sq_factors)
            divisors.sort()
            
            # For each divisor u of b^2, check if u*v = b^2 and
            # y = (u + b) / a, z = (v + b) / a are integers
            # We need (u+b) % a == 0 and (v+b) % a == 0
            
            for u in divisors:
                if u > b_sq:
                    break
                if b_sq % u != 0:
                    continue
                v = b_sq // u
                
                if (u + b) % a != 0:
                    continue
                if (v + b) % a != 0:
                    continue
                
                y = (u + b) // a
                z = (v + b) // a
                
                if y <= 0 or z <= 0:
                    continue
                if y > BOUND or z > BOUND or x > BOUND:
                    continue
                if x == y or x == z or y == z:
                    continue
                
                # Verify: 1/x + 1/y + 1/z = 4/n
                # n*(y*z + x*z + x*y) == 4*x*y*z
                lhs = n * (y * z + x * z + x * y)
                rhs = 4 * x * y * z
                if lhs == rhs:
                    found = [x, y, z]
                    break
            
            if found:
                break
    
        answers[idx] = found
    
    return answers
