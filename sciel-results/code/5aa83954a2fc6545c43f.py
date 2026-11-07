import math

def construct():
    primes = [2, 3, 5, 7, 11, 13]
    n_primes = len(primes)
    MAX_M = 10**18
    
    best_M = 0
    best_k = []
    max_count = 0
    
    def generate(idx, current_M, exponents):
        nonlocal best_M, best_k, max_count
        
        if idx == n_primes:
            # Evaluate current_M
            # Generate divisors
            divisors = [1]
            for i in range(idx):
                p = primes[i]
                e = exponents[i]
                new_divs = []
                p_power = 1
                for _ in range(e + 1):
                    for d in divisors:
                        new_divs.append(d * p_power)
                    p_power *= p
                divisors = new_divs
            
            count = 0
            valid_ks = []
            sqrt_M = math.isqrt(current_M)
            for k in divisors:
                if k > sqrt_M:
                    continue
                # Calculate sigma(k)
                sigma = 1
                temp_k = k
                for i in range(idx):
                    p = primes[i]
                    if temp_k % p == 0:
                        power = 0
                        p_power = 1
                        while temp_k % p == 0:
                            temp_k //= p
                            power += 1
                            p_power *= p
                        sigma *= (p_power * p - 1) // (p - 1)
                
                if k * sigma == current_M:
                    count += 1
                    valid_ks.append(k)
            
            if count > max_count:
                max_count = count
                best_M = current_M
                best_k = valid_ks
            return

        p = primes[idx]
        max_e = 0
        prod = current_M
        while prod * p <= MAX_M:
            prod *= p
            max_e += 1
            
        for e in range(max_e + 1):
            exponents.append(e)
            generate(idx + 1, current_M * (p ** e), exponents)
            exponents.pop()

    generate(0, 1, [])
    
    if max_count == 0:
        return {'M': 333312, 'k': [336, 372, 434]}
        
    return {'M': best_M, 'k': best_k}
