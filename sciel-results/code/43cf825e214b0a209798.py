import math
import time

def construct():
    answers = [None] * 254
    start_time = time.time()
    deadline = start_time + 52  # Safety margin for 90s limit

    # Precompute primes up to 20000 for factoring B = n*x
    # n <= 256, x can be up to ~50000 in our search, so B <= 256*50000 = 12,800,000
    # sqrt(12,800,000) ~ 3578, but let's be safe with 20000
    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    primes = get_primes(20000)

    def factorize(num):
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

    def get_divisors(factors):
        divs = [1]
        for p, e in factors.items():
            new_divs = []
            for d in divs:
                p_pow = 1
                for _ in range(e + 1):
                    new_divs.append(d * p_pow)
                    p_pow *= p
            divs = new_divs
        return divs

    for n in range(3, 257):
        idx = n - 3
        if time.time() > deadline:
            continue

        found = None
        # x must satisfy 1/x < 4/n, so x > n/4
        start_x = n // 4 + 1

        # Search over x values
        # For small n, solutions with small x exist. For larger n, x may need to be larger.
        # Let's try a range that adapts or is generous.
        # The previous solution used 20000. Let's try to be smarter or just generous.
        # Given time limit, 20000 iterations per n * 254 n's = 5M iterations.
        # Each iteration does factorization of B=n*x.
        # Factorization is fast for small numbers.
        # Let's stick to a similar range but optimize the inner loop.
        
        max_x_search = 20000
        
        for x in range(start_x, start_x + max_x_search):
            if time.time() > deadline:
                break
                
            A = 4 * x - n
            if A <= 0:
                continue
            
            B = n * x
            
            # 1/y + 1/z = A/B
            # (y+z)/(yz) = A/B
            # B(y+z) = Ayz
            # Ayz - By - Bz = 0
            # (Ay - B)(Az - B) = B^2
            # Let U = Ay - B, V = Az - B
            # UV = B^2
            # y = (U+B)/A, z = (V+B)/A
            
            # Special case A=1
            if A == 1:
                y = B + 1
                z = B * (B + 1)
                if y > 0 and z > 0 and y != z and y != x and z != x:
                    if y <= 10**12 and z <= 10**12:
                        found = [x, y, z]
                        break
                continue

            # Factor B to get divisors of B^2
            b_factors = factorize(B)
            b2_factors = {p: 2 * e for p, e in b_factors.items()}
            divs = get_divisors(b2_factors)
            
            # We need U such that (U+B) % A == 0 and (B^2/U + B) % A == 0
            # And y, z distinct from x and each other, positive.
            
            # Optimization: Sort divisors? 
            # Smaller U gives smaller y. 
            # If we find a solution, we break.
            
            for U in divs:
                if (U + B) % A != 0:
                    continue
                V = (B * B) // U
                if (V + B) % A != 0:
                    continue
                
                y = (U + B) // A
                z = (V + B) // A
                
                if y > 0 and z > 0:
                    if y != z and y != x and z != x:
                        if y <= 10**12 and z <= 10**12:
                            found = [x, y, z]
                            break
            if found:
                break
        
        answers[idx] = found

    return answers
