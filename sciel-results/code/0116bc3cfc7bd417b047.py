import time
import math
import sys
from collections import defaultdict, Counter

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin for 150s limit

    def check_time():
        return time.time() > deadline

    def sieve(n):
        if n < 2:
            return []
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, v in enumerate(is_prime) if v]

    # Step 1: Find "good" primes where p+1 is smooth
    # We want p+1 to have only small prime factors to allow for many combinations
    # Let's target p+1 being 7-smooth (factors 2, 3, 5, 7)
    # We'll generate primes up to 100,000 and filter
    
    limit_p = 100000
    primes = sieve(limit_p)
    
    def is_smooth(n, smooth_primes):
        for p in smooth_primes:
            while n % p == 0:
                n //= p
        return n == 1

    smooth_primes = [2, 3, 5, 7, 11] # Allow 11 as well to get more primes
    
    good_primes = []
    for p in primes:
        if is_smooth(p + 1, smooth_primes):
            good_primes.append(p)
            
    # Sort good_primes
    good_primes.sort()
    
    # We want to form M = prod p_i * (p_i + 1)
    # To keep M <= 10^18, we can't take too many large primes.
    # Let's try to maximize the number of primes in the product.
    # Smaller primes allow more of them to fit.
    
    # We will test subsets of good_primes.
    # Since the number of primes is likely small (dozens), we can be smart.
    # Actually, we just want the product of the first K good primes to be <= 10^18.
    
    best_m = 0
    best_k_list = []
    
    # Helper to factorize and find solutions
    def is_prime(n):
        if n < 2: return False
        if n in [2, 3]: return True
        if n % 2 == 0 or n % 3 == 0: return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    def pollard_rho(n):
        if n % 2 == 0: return 2
        if n % 3 == 0: return 3
        while True:
            x = 2
            y = 2
            d = 1
            c = 1
            while d == 1:
                x = (x * x + c) % n
                y = (y * y + c) % n
                y = (y * y + c) % n
                d = math.gcd(abs(x - y), n)
            if d != n:
                return d
            c += 1

    def factorize(n):
        factors = []
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]:
            while n % p == 0:
                factors.append(p)
                n //= p
        if n == 1:
            return factors
        if is_prime(n):
            factors.append(n)
        else:
            stack = [n]
            while stack:
                x = stack.pop()
                if x == 1: continue
                if is_prime(x):
                    factors.append(x)
                else:
                    d = pollard_rho(x)
                    stack.append(d)
                    stack.append(x // d)
        return sorted(factors)

    def generate_divisors_from_factors(factors):
        c = Counter(factors)
        primes_list = list(c.keys())
        exponents_list = [c[p] for p in primes_list]
        
        divs = [1]
        for i in range(len(primes_list)):
            p = primes_list[i]
            e = exponents_list[i]
            current_divs = []
            power = 1
            for k in range(e + 1):
                for d in divs:
                    current_divs.append(d * power)
                power *= p
            divs = current_divs
        return divs

    def count_solutions(M):
        if M <= 0:
            return 0, []
        
        try:
            factors = factorize(M)
            if not factors:
                return 0, []
            
            divs = generate_divisors_from_factors(factors)
            
            valid_ks = []
            m_primes_set = set(factors)
            
            for d in divs:
                if d > math.isqrt(M):
                    continue
                    
                # Calculate sigma(d)
                temp_d = d
                sigma_d = 1
                for p in m_primes_set:
                    if temp_d % p == 0:
                        e = 0
                        while temp_d % p == 0:
                            temp_d //= p
                            e += 1
                        sigma_d *= (p**(e+1) - 1) // (p - 1)
                
                if temp_d != 1:
                    continue
                
                if d * sigma_d == M:
                    valid_ks.append(d)
            
            return len(valid_ks), valid_ks
        except Exception:
            return 0, []

    # Strategy: Try products of the first N good primes
    # M_N = prod_{i=1}^N p_i (p_i + 1)
    
    current_M = 1
    current_primes_used = []
    
    for p in good_primes:
        term = p * (p + 1)
        if current_M > 10**18 // term:
            break
        
        current_M *= term
        current_primes_used.append(p)
        
        if check_time():
            break
        
        # Check this M
        count, ks = count_solutions(current_M)
        
        if count > len(best_k_list):
            best_k_list = ks
            best_m = current_M
            
        # If we have a very good candidate, we might want to try variations?
        # But for now, let's just keep adding primes to see if count grows.
        
    # It's possible that a subset of these primes (not just prefix) yields more solutions?
    # Or maybe mixing with other structures?
    # Given time constraints, the prefix product of "most compatible" primes is a strong heuristic.
    
    # If we didn't find anything better than a trivial case, fallback
    if not best_k_list:
        # Fallback to a known small example
        # M=333312, k=[336, 372, 434] from seed
        best_m = 333312
        best_k_list = [336, 372, 434]
        # Verify fallback
        # 336 * sigma(336) = 336 * (1+2+4+8+3+6+12+24+7+14+28+56+168+336) ... wait
        # 336 = 2^4 * 3 * 7. sigma(336) = (31)(4)(8) = 992.
        # 336 * 992 = 333312. Correct.
        # 372 = 2^2 * 3 * 31. sigma(372) = (7)(4)(32) = 896.
        # 372 * 896 = 333312. Correct.
        # 434 = 2 * 7 * 31. sigma(434) = (3)(8)(32) = 768.
        # 434 * 768 = 333312. Correct.
        # So this fallback is valid and has 3 solutions.
        
    # Final validation
    distinct_k = list(set(best_k_list))
    sqrt_m = math.isqrt(best_m)
    valid_k = [k for k in distinct_k if k <= sqrt_m]
    valid_k.sort()
    
    M_out = int(best_m)
    k_out = [int(k) for k in valid_k]
    
    return {'M': M_out, 'k': k_out}

if __name__ == "__main__":
    result = construct()
    print(result)
