import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit
    MAX_DEN = 10**12
    
    def factorize(n):
        factors = {}
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors[d] = factors.get(d, 0) + 1
                n //= d
            d += 1
        if n > 1:
            factors[n] = factors.get(n, 0) + 1
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
        # x is in (n/4, 3n/4)
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                return None
            
            A = 4 * x - n
            B = n * x
            
            if A <= 0:
                continue
                
            g = math.gcd(A, B)
            A_prime = A // g
            B_prime = B // g
            
            # We need u' * v' = B_prime^2
            # u' < v' => u' < B_prime
            # y > x => (u' + B_prime) / A_prime > x => u' > x * A_prime - B_prime
            u_min = x * A_prime - B_prime + 1
            u_max = B_prime - 1
            
            if u_min > u_max:
                continue
                
            # Factorize B_prime to get divisors of B_prime^2
            # B_prime is relatively small
            b_factors = factorize(B_prime)
            b_sq_factors = {p: 2*e for p, e in b_factors.items()}
            divs = divisors_from_factors(b_sq_factors)
            
            for u_prime in divs:
                if u_prime > u_max:
                    break
                if u_prime < u_min:
                    continue
                # Check congruence
                if (u_prime + B_prime) % A_prime != 0:
                    continue
                
                y = (u_prime + B_prime) // A_prime
                v_prime = (B_prime * B_prime) // u_prime
                z = (v_prime + B_prime) // A_prime
                
                # Verify z is integer (should be, but check just in case of logic errors)
                # Actually, we checked y is integer. z integer follows from congruence if gcd(A',B')=1.
                # But let's be safe.
                if (v_prime + B_prime) % A_prime != 0:
                    continue
                
                if z > MAX_DEN:
                    continue
                
                # Check distinctness and order
                # x < y < z
                if y <= x or z <= y:
                    continue
                if x == y or x == z or y == z:
                    continue
                    
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
