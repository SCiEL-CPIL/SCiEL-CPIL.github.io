import math
import time

def construct():
    answers = []
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 90s limit
    
    # Precompute primes up to a limit sufficient for factoring B
    # B = n * x. n <= 256. x can go up to ~10000.
    # B can be up to 256 * 10000 = 2,560,000.
    # sqrt(2,560,000) is ~1600.
    # Let's sieve up to 5000 to be safe.
    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    primes = get_primes(5000)

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
        if time.time() > deadline:
            answers.append(None)
            continue
            
        found = None
        # Search for x
        # x must be > n/4
        start_x = n // 4 + 1
        # We can afford a larger range now.
        for x in range(start_x, start_x + 10000):
            if time.time() > deadline:
                break
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x
            
            # Optimization: If A == 1, direct solution
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
            # Factors of B^2 are p -> 2*e
            b2_factors = {p: 2 * e for p, e in b_factors.items()}
            divs = get_divisors(b2_factors)
            
            # Sort divisors to find smaller y/z first? 
            # Smaller U gives smaller y = (U+B)/A.
            # So sorting ascending might yield smaller solutions.
            divs.sort()
            
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
        answers.append(found)
    return answers
