import math
import time

def construct():
    answers = []
    deadline = time.time() + 55  # Safety margin
    
    # Precompute primes for factorization
    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    # Max B is roughly n * x_max. If x_max is 1000, B is 256000.
    # B^2 is ~ 6 * 10^10. Factorization is fast.
    # We might need larger x if small ones fail.
    # Let's set a dynamic limit or just try up to a certain x.
    
    primes = get_primes(10000) # Enough for B up to 10^8
    
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
        
        # Limit x search to avoid infinite loop or excessive time
        # For n=3, x>0.75 -> x>=1.
        # For n=256, x>64.
        # Usually small x works. Let's try up to 10000 first.
        max_x = 100000 
        
        for x in range(start_x, start_x + max_x):
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x
            
            # We need U | B^2 such that U == -B mod A and (B^2/U) == -B mod A
            # Note: if A == 1, any U works?
            # y = U+B, z = B^2/U + B.
            # If A=1, y, z are always integers.
            # We just need distinctness.
            if A == 1:
                # Pick U=1?
                # y = 1+B, z = B^2 + B
                y = 1 + B
                z = B * B + B
                if y != z and y != x and z != x and y <= 10**12 and z <= 10**12:
                    found = [x, y, z]
                    break
                # Try another U?
                # If distinctness fails, try U=2 if B^2 even?
                # But A=1 is rare (4x-n=1 -> n=4x-1).
                # Let's just stick to the general case or check a few divisors.
                continue

            b_factors = factorize(B)
            b2_factors = {p: 2*e for p, e in b_factors.items()}
            divs = get_divisors(b2_factors)
            
            found_x = False
            # Optimization: Sort divisors? No need, just iterate.
            # We can stop if U > B and we've checked all pairs?
            # Actually U and V are symmetric. We can just check all U.
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
                            found_x = True
                            break
            if found:
                break
        
        answers.append(found)
    return answers
