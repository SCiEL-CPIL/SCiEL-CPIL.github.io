import math

def construct():
    # Precompute primes up to 1000 for factorization
    primes = []
    is_prime = [True] * 1000
    is_prime[0] = is_prime[1] = False
    for i in range(2, 1000):
        if is_prime[i]:
            primes.append(i)
            for j in range(i*i, 1000, i):
                is_prime[j] = False

    def factorize(num):
        if num == 1:
            return {}
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
        for p, exp in factors.items():
            new_divs = []
            p_power = 1
            for _ in range(exp + 1):
                for d in divs:
                    new_divs.append(d * p_power)
                p_power *= p
            divs = new_divs
        return divs

    answers = []
    for n in range(3, 257):
        found = None
        x_min = n // 4 + 1
        x_max = (3 * n - 1) // 4
        
        # Iterate x from large to small (shortest inner loops first)
        for x in range(x_max, x_min - 1, -1):
            A = 4 * x - n
            B = n * x
            if A <= 0:
                continue
            
            g = math.gcd(A, B)
            A_prime = A // g
            B_prime = B // g
            
            # We need u | B_prime^2, u == -B_prime mod A_prime
            # and u < B_prime^2 / u
            B_sq = B_prime * B_prime
            
            # Factorize B_prime
            fac = factorize(B_prime)
            # Get divisors of B_prime^2
            # Exponents are doubled
            fac_sq = {p: 2*e for p, e in fac.items()}
            divisors = get_divisors(fac_sq)
            
            # Filter and check
            # To avoid checking all divisors, we can just iterate them
            # Number of divisors is small
            for u in divisors:
                # Check condition u < v
                v = B_sq // u
                if u >= v:
                    continue
                
                # Check congruence
                # u == -B_prime mod A_prime
                if (u + B_prime) % A_prime != 0:
                    continue
                
                # Calculate y, z
                # y = (u + B_prime) / A_prime
                # z = (v + B_prime) / A_prime
                # Since (u+B_prime) % A_prime == 0, y is integer
                y = (u + B_prime) // A_prime
                z = (v + B_prime) // A_prime
                
                # Check distinctness
                if y == x or z == x or y == z:
                    continue
                
                # Check bounds
                if y > 1000000000000 or z > 1000000000000:
                    continue
                
                # Valid solution
                found = [x, y, z]
                break
            
            if found:
                break
        
        answers.append(found)
    return answers
