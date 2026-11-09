import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N_MAX = 1000000
    N_MIN = 10000
    
    # 1. Generate all powerful numbers up to N_MAX
    # A number is powerful if all prime factors have exponent >= 2.
    # We generate them by iterating through primes and their powers.
    
    # Sieve for primes up to sqrt(N_MAX)
    # Primes p > sqrt(N_MAX) cannot have p^2 <= N_MAX, so they don't form the base of a powerful number component > 1.
    limit = int(N_MAX**0.5) + 1
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
                
    primes = [i for i, v in enumerate(is_prime) if v]
    
    powerful_numbers = []
    
    def gen_powerful(current, min_prime_idx, max_val):
        powerful_numbers.append(current)
        for i in range(min_prime_idx, len(primes)):
            p = primes[i]
            if p * p > max_val // current:
                break
            power = p * p
            while current * power <= max_val:
                gen_powerful(current * power, i, max_val)
                power *= p
    
    gen_powerful(1, 0, N_MAX)
    
    # Remove duplicates (shouldn't be any if logic is correct, but safe to do)
    powerful_numbers = sorted(list(set(powerful_numbers)))
    
    # 2. Search for best A, B
    # We want to maximize A * B
    # Such that gcd(A, B) == 1
    # And there exists n in [N_MIN, N_MAX] with n % A == 0 and n % B == -1
    
    best_val = 0
    best_n = N_MIN
    
    pn_desc = sorted(powerful_numbers, reverse=True)
    n_pn = len(pn_desc)
    
    for i in range(n_pn):
        A = pn_desc[i]
        if A > N_MAX:
            continue
        
        # Prune outer loop
        if A * pn_desc[0] <= best_val:
            break
            
        for j in range(n_pn):
            B = pn_desc[j]
            
            # Prune inner loop
            if A * B <= best_val:
                break
                
            if math.gcd(A, B) != 1:
                continue
                
            # Solve n = k*A = m*B - 1
            # We need n such that n % A == 0 and n % B == B-1
            
            try:
                inv_A = pow(A, -1, B)
            except ValueError:
                continue
                
            x = (-inv_A) % B
            n0 = A * x
            L = A * B
            
            # Find if any n in [N_MIN, N_MAX]
            if n0 >= N_MIN:
                k_min = 0
            else:
                k_min = (N_MIN - n0 + L - 1) // L
                
            k_max = (N_MAX - n0) // L
            
            if k_min <= k_max:
                val = A * B
                if val > best_val:
                    best_val = val
                    best_n = n0 + k_min * L
                    
                    # Time check
                    if time.time() > deadline:
                        return best_n

    return best_n
