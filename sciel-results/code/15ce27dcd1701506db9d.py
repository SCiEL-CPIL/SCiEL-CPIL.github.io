import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    TARGET = 123457
    
    # Sieve of Eratosthenes to get primes up to a limit
    # We expect the answer to be small, so 1000 is safe.
    limit = 1000
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    # If the optimal prime is > 1000, we need a fallback.
    # But for N=123457, it's very unlikely.
    # Fallback: use the linear scan method from the current program if no solution found.
    
    smooth_set = {1}
    
    for i, p in enumerate(primes):
        if time.time() > deadline:
            break
            
        # Expand smooth_set with prime p
        # We need to multiply existing smooth numbers by p^k
        # We iterate over a snapshot of the current set
        # Note: The set grows. We only multiply numbers that were in the set BEFORE adding p's multiples?
        # No, we need to generate ALL numbers with prime factors in {primes[0..i]}.
        # The new numbers are those divisible by p.
        # They are of the form p^k * m, where m is not divisible by p.
        # m must be composed of primes < p.
        # So m is in the set of (p-1)-smooth numbers.
        # So we iterate over the set of (p-1)-smooth numbers.
        
        # However, our `smooth_set` currently contains (p-1)-smooth numbers.
        # Let's take a snapshot.
        prev_smooth = list(smooth_set)
        
        for m in prev_smooth:
            val = m
            while val * p <= TARGET:
                val *= p
                smooth_set.add(val)
                
        # Check for solution
        # We need a, b in smooth_set such that a + b = TARGET
        # Iterate through a in smooth_set
        found = False
        for a in smooth_set:
            if a > TARGET:
                continue
            b = TARGET - a
            if b in smooth_set:
                return [a, b]
                
    # Fallback to linear scan if no solution found with primes up to 1000
    # This should not happen for N=123457
    # But just in case, we can run the original logic
    # Or just return [1, TARGET-1]
    # P(1)=1, P(123456)=23.
    # So 23 is a valid upper bound.
    # If the loop didn't return, it means no solution with P <= 1000?
    # That's impossible since P=23 is a solution.
    # So the loop MUST have returned by p=23.
    
    return [1, TARGET - 1]
