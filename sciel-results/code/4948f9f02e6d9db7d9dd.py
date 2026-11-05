import time
from fractions import Fraction

def _primes_up_to(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin for 150s limit
    
    primes = _primes_up_to(1000)
    
    # Budget
    budget = Fraction(1, 1)
    used_budget = Fraction(0, 1)
    
    # State
    unused_primes = set(primes)
    classes = []
    
    # MAX_L cap from task is 10^5
    MAX_L = 100000
    
    # is_covered array
    # We need indices up to MAX_L
    is_covered = bytearray(MAX_L + 1)
    
    current_L = 0
    
    while current_L < MAX_L:
        if time.time() > DEADLINE:
            break
            
        n = current_L + 1
        
        # Advance L if next numbers are already covered
        while n <= MAX_L and is_covered[n]:
            current_L = n
            n += 1
            
        if n > MAX_L:
            break
            
        # n is the first uncovered integer. We must cover it.
        
        # Identify candidate primes
        candidates = {}
        for p in unused_primes:
            cost = Fraction(1, p)
            if used_budget + cost <= budget:
                r = n % p
                candidates[p] = r
        
        if not candidates:
            # No more primes can be added
            break
            
        active = set(candidates.keys())
        limits = {}
        m = n
        
        # Find the coverage limit for each candidate
        # We scan forward from n
        while active and m <= MAX_L:
            if is_covered[m]:
                # m is covered by existing classes, so all candidates cover it
                m += 1
                continue
                
            # m is NOT covered by existing classes
            # Check which candidates fail to cover m
            to_remove = []
            for p in active:
                r_p = candidates[p]
                if m % p != r_p:
                    # Candidate p does not cover m
                    limits[p] = m - 1
                    to_remove.append(p)
                    
            if to_remove:
                active.difference_update(to_remove)
                
            m += 1
            
        # If active is not empty, those candidates cover up to MAX_L
        for p in active:
            limits[p] = MAX_L
            
        # Select the best candidate (maximizes L)
        best_p = None
        best_L = -1
        for p, L_val in limits.items():
            if L_val > best_L:
                best_L = L_val
                best_p = p
                
        if best_p is None:
            break
            
        # Commit the choice
        r_best = candidates[best_p]
        used_budget += Fraction(1, best_p)
        unused_primes.remove(best_p)
        classes.append([best_p, r_best])
        
        # Update is_covered array
        start = r_best
        if start == 0:
            start = best_p
            
        # Mark all covered numbers
        for val in range(start, MAX_L + 1, best_p):
            is_covered[val] = 1

    return classes
