import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    MAX_DEN = 10**12
    
    # Sieve for small primes up to 100000
    SIEVE_LIMIT = 100000
    is_prime = [True] * (SIEVE_LIMIT + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(SIEVE_LIMIT)) + 1):
        if is_prime[i]:
            for j in range(i*i, SIEVE_LIMIT + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, SIEVE_LIMIT + 1) if is_prime[i]]
    
    def factorize(n):
        factors = {}
        temp = n
        for p in primes:
            if p * p > temp:
                break
            while temp % p == 0:
                factors[p] = factors.get(p, 0) + 1
                temp //= p
        if temp > 1:
            factors[temp] = factors.get(temp, 0) + 1
        return factors
    
    def divisors_from_factors(factors):
        divs = [1]
        for p, e in factors.items():
            new_divs = []
            for d in divs:
                pk = 1
                for k in range(e + 1):
                    new_divs.append(d * pk)
                    pk *= p
            divs = new_divs
        return divs
    
    def find_solution(n):
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        # Try x values in an order that's likely to find solutions quickly
        # Small x values give larger B'/A' ratio, potentially easier
        x_range = list(range(x_min, x_max + 1))
        
        for x in x_range:
            if time.time() > deadline:
                return None
            
            A = 4 * x - n
            B = n * x
            
            if A <= 0:
                continue
                
            g = math.gcd(A, B)
            A_prime = A // g
            B_prime = B // g
            
            if A_prime == 1:
                # Then y = u' + B', z = v' + B', need u'*v' = B'^2
                # Any divisor u' of B'^2 with u' < B' gives a solution
                # y = u' + B', z = B'^2/u' + B'
                # Need distinct: x, y, z all different
                # Try u' = 1
                y = 1 + B_prime
                z = B_prime * B_prime + B_prime
                if y > x and z > y and z <= MAX_DEN and x != y and y != z:
                    return (x, y, z)
                # Try other small divisors
                b_factors = factorize(B_prime)
                b_sq_factors = {p: 2*e for p, e in b_factors.items()}
                divs = divisors_from_factors(b_sq_factors)
                for u_prime in divs:
                    if u_prime >= B_prime:
                        break
                    y = u_prime + B_prime
                    v_prime = (B_prime * B_prime) // u_prime
                    z = v_prime + B_prime
                    if y > x and z > y and z <= MAX_DEN and x != y and y != z:
                        return (x, y, z)
                continue
            
            # u' * v' = B'^2, u' + B' ≡ 0 (mod A'), u' < B'
            # u' ≡ -B' (mod A')
            # Since gcd(A', B') = 1, -B' mod A' is well-defined
            target = (-B_prime) % A_prime
            
            b_factors = factorize(B_prime)
            b_sq_factors = {p: 2*e for p, e in b_factors.items()}
            divs = divisors_from_factors(b_sq_factors)
            
            for u_prime in divs:
                if u_prime >= B_prime:
                    break
                if u_prime % A_prime == target:
                    y = (u_prime + B_prime) // A_prime
                    v_prime = (B_prime * B_prime) // u_prime
                    z = (v_prime + B_prime) // A_prime
                    if y > x and z > y and z <= MAX_DEN and x != y and y != z:
                        return (x, y, z)
        
        return None
    
    answers = []
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        sol = find_solution(n)
        answers.append(sol)
    
    return answers
