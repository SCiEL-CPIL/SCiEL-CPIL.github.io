import time
import math
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before 120s limit

    # Generate primes up to 300 for more collision opportunities
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, v in enumerate(is_prime) if v]

    primes = sieve(300)

    max_m = 10**18
    sqrt_max_m = math.isqrt(max_m)  # 10^9

    solutions = defaultdict(set)
    last_time_check = start_time

    def check_time():
        nonlocal last_time_check
        now = time.time()
        if now - last_time_check > 0.5:
            last_time_check = now
            if now > deadline:
                return True
        return False

    # Iterative generation of smooth numbers
    # State: (current_k, current_sigma)
    # Process primes one by one, expanding states by multiplying by p^e
    
    # Start with just (1, 1)
    states = [(1, 1)]
    
    for idx, p in enumerate(primes):
        new_states = []
        for (k, sigma) in states:
            # Exponent 0: don't use this prime - carry forward
            new_states.append((k, sigma))
            
            # Exponent e >= 1: multiply by p^e
            k_curr = k
            sum_powers = 1
            p_power = 1
            
            # Max exponent: p^e <= sqrt_max_m / k
            max_e = 0
            temp = k
            while temp <= sqrt_max_m // p and max_e < 30:
                temp *= p
                max_e += 1
            
            for _ in range(max_e):
                k_curr *= p
                p_power *= p
                sum_powers += p_power
                
                # sigma for k_curr = sigma * sum_powers
                sigma_new = sigma * sum_powers
                
                # Pruning: k * sigma must be <= max_m
                if k_curr > max_m // sigma_new:
                    break
                # Pruning: k must be <= sqrt(M), i.e., k <= sigma
                if k_curr > sigma_new:
                    break
                    
                new_states.append((k_curr, sigma_new))
        
        states = new_states
        
        if check_time():
            break
    
    # Process all final states
    for (k, sigma) in states:
        if k > 1:
            m_val = k * sigma
            if m_val <= max_m and k <= sigma:  # k <= sqrt(M) equivalent
                solutions[m_val].add(k)
    
    # Find best M
    best_m = 0
    best_k_set = set()
    
    for m, k_set in solutions.items():
        if len(k_set) > len(best_k_set):
            best_m = m
            best_k_set = k_set
    
    if not best_k_set or len(best_k_set) < 3:
        return {'M': 333312, 'k': [336, 372, 434]}
    
    sqrt_m = math.isqrt(best_m)
    valid_k = sorted([k for k in best_k_set if k <= sqrt_m])
    
    M_out = int(best_m)
    k_out = [int(k) for k in valid_k]
    
    return {'M': M_out, 'k': k_out}

if __name__ == "__main__":
    result = construct()
    print(result)
