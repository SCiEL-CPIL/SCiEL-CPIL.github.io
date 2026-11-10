import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    answers = []
    MAX_Z = 10**12
    n_total = 254
    
    # Precompute primes up to 300 for factorization
    primes = []
    is_prime = [True] * 300
    for i in range(2, 300):
        if is_prime[i]:
            primes.append(i)
            for j in range(i*i, 300, i):
                is_prime[j] = False

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
            p_power = 1
            for i in range(e + 1):
                for d in divs:
                    new_divs.append(d * p_power)
                p_power *= p
            divs = new_divs
        return divs

    for idx, n in enumerate(range(3, 257)):
        elapsed = time.time() - start_time
        remaining = deadline - start_time - elapsed
        remaining_n = n_total - idx
        
        if remaining <= 0:
            answers.append(None)
            continue
        
        time_per_n = min(remaining / remaining_n, 0.5)
        n_deadline = time.time() + time_per_n
        
        found = None
        
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        # Try x values in reverse order: larger x first.
        for x in range(x_max, x_min - 1, -1):
            if time.time() > n_deadline:
                break
                
            A = 4 * x - n
            if A <= 0:
                continue
            
            B = n * x
            B_sq = B * B
            
            # Factorize B to get divisors of B^2
            # Note: We can factorize B and then square the exponents
            factors_B = factorize(B)
            factors_B_sq = {p: 2*e for p, e in factors_B.items()}
            
            divs = get_divisors(factors_B_sq)
            
            for X in divs:
                Z = B_sq // X
                y_num = X + B
                z_num = Z + B
                
                if y_num % A != 0 or z_num % A != 0:
                    continue
                
                y = y_num // A
                z = z_num // A
                
                if y <= 0 or z <= 0:
                    continue
                
                if y > MAX_Z or z > MAX_Z:
                    continue
                
                # Check distinctness
                if x == y or x == z or y == z:
                    continue
                
                found = [x, y, z]
                break
            
            if found:
                break
        
        answers.append(found)
        
    return answers
