import time
import math

DENOM_BOUND = 10**12

def construct():
    deadline = time.time() + 85.0
    answers = [None] * 254
    
    # Precompute primes up to 10000 for factorization
    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(math.isqrt(n)) + 1):
            if sieve[i]:
                for j in range(i*i, n + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    PRIMES = get_primes(10000)
    
    def factorize(num, primes):
        factors = {}
        for p in primes:
            if p * p > num:
                break
            if num % p == 0:
                count = 0
                while num % p == 0:
                    num //= p
                    count += 1
                factors[p] = count
        if num > 1:
            factors[num] = 1
        return factors

    def generate_divisors(factors):
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

    for i, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
            
        found = None
        nx = n
        
        # x must be > n/4 and <= 3n/4
        # We assume x <= y <= z
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
                
            # A = 4x - n, B = nx
            A = 4 * x - n
            if A <= 0:
                continue
                
            B = nx * x
            
            # We need 1/y + 1/z = A/B
            # (Ay - B)(Az - B) = B^2
            # Let U = Ay - B, V = Az - B. Then UV = B^2.
            # y = (U+B)/A, z = (V+B)/A.
            
            # Factorize B to get divisors of B^2
            factors_B = factorize(B, PRIMES)
            factors_B2 = {p: 2*e for p, e in factors_B.items()}
            
            divs = generate_divisors(factors_B2)
            
            # We assume y <= z, so U <= V, which implies U <= B.
            # Also y > x (distinct and ordered) => (U+B)/A > x => U > Ax - B.
            lower_U = A * x - B
            if lower_U < 0:
                lower_U = 0 # U is positive divisor
            
            upper_U = B
            
            # Iterate divisors U
            # To speed up, we can sort divisors or just iterate. 
            # The number of divisors is small.
            for U in divs:
                if U > upper_U:
                    continue
                if U < lower_U:
                    continue
                
                # Check integrality of y
                if (U + B) % A != 0:
                    continue
                
                y = (U + B) // A
                
                # Check distinctness and order: x < y
                if y <= x:
                    continue
                
                # Calculate V
                V = (B * B) // U
                
                # Check integrality of z
                if (V + B) % A != 0:
                    continue
                
                z = (V + B) // A
                
                # Check order y < z (distinct)
                if z <= y:
                    continue
                
                # Check bounds
                if z > DENOM_BOUND:
                    continue
                
                # Check positive (guaranteed by construction if x,y,z > 0)
                # But let's be safe
                if y < 1 or z < 1:
                    continue
                
                found = [x, y, z]
                break
            
            if found:
                break
                
        answers[i] = found
        
    return answers
