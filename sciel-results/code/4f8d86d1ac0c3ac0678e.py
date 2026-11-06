import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    max_n = 10**18
    
    # Greedy construction of exponents
    # We want a_1 >= a_2 >= ... >= a_k >= 1
    # Start with a_1 = 1, a_2 = 1, ... until product exceeds max_n
    # Then adjust exponents to maximize tau(n) = prod(a_i+1)
    
    # Initial guess: use as many primes as possible with exponent 1
    # 2*3*5*...*47 = 614889782588491410 < 10^18
    # Next prime 53: 614889782588491410 * 53 > 10^18
    # So max 15 primes with exponent 1.
    
    best_n = 1
    best_tau = 1
    best_exponents = []
    
    # Search space for exponents is small.
    # a_1 can be up to 59.
    # a_2 up to 37.
    # a_3 up to 25.
    # etc.
    
    # We will perform a local search starting from the primorial of 15 primes.
    current_exponents = [1] * 15
    current_n = 1
    for i, p in enumerate(primes[:15]):
        current_n *= p
        
    current_tau = 2**15
    
    def calc_tau(exps):
        t = 1
        for e in exps:
            t *= (e + 1)
        return t
        
    def calc_n(exps):
        n = 1
        for i, e in enumerate(exps):
            if e > 0:
                n *= primes[i] ** e
        return n

    # Local search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try increasing exponents
        for i in range(15):
            if time.time() > deadline:
                break
            # Check if we can increase exponent i
            # Must maintain a_i >= a_{i+1}
            # And n * p_i <= max_n
            if i < 14 and current_exponents[i] < current_exponents[i+1]:
                continue
            if current_n * primes[i] > max_n:
                continue
                
            new_exps = current_exponents[:]
            new_exps[i] += 1
            new_n = current_n * primes[i]
            new_tau = calc_tau(new_exps)
            
            if new_tau > current_tau:
                current_exponents = new_exps
                current_n = new_n
                current_tau = new_tau
                improved = True
                
        # Try decreasing exponents to allow increasing others?
        # Or just try swapping?
        # For now, just increasing is a good heuristic for HCNs.
        # But we might get stuck.
        # Let's also try reducing a large exponent to increase a smaller one?
        # Actually, HCNs have non-increasing exponents.
        # The greedy increase works well.
        
        # Let's also try starting from different bases?
        # No, time limit.
        
        # Update best
        if current_tau > best_tau:
            best_tau = current_tau
            best_n = current_n
            best_exponents = current_exponents[:]

    return best_n
