import math
import time

def construct():
    N = 123457
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Sieve of Eratosthenes to get primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(N)) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
    
    primes = [i for i, prime in enumerate(is_prime) if prime]
    
    # Set of B-smooth numbers
    # Start with 1, which is 2-smooth (vacuously)
    smooth_numbers = {1}
    
    # We need to check for pairs.
    # If a pair is found with bound B, the max prime factor is <= B.
    # We want to minimize the actual max prime factor.
    # Since we iterate B (primes) in increasing order, the first pair found
    # will have the smallest possible B.
    # However, the actual max prime factor of the pair might be less than B.
    # But any pair with max prime factor p' < B would have been found
    # when the bound was p' (or a prime between p' and B).
    # So the first pair found is optimal.
    
    for B in primes:
        if time.time() > deadline:
            break
            
        # Generate new smooth numbers using prime B
        # Multiply existing smooth numbers by B, B^2, ...
        new_numbers = []
        # We need to iterate over a copy or list to avoid modifying while iterating
        # Or just collect new ones
        for a in list(smooth_numbers):
            val = a * B
            while val <= N:
                new_numbers.append(val)
                val *= B
        
        # Add new numbers to the set
        smooth_numbers.update(new_numbers)
        
        # Check for a valid pair
        # We need a and N-a to be in smooth_numbers
        # To be efficient, we can iterate over the smaller set or just the new numbers?
        # No, the pair could consist of two old numbers that only now form a valid pair?
        # No, if a and N-a were both in the set before, we would have found them in a previous step.
        # So we only need to check if any a in smooth_numbers has N-a in smooth_numbers.
        # But checking all a in smooth_numbers every time might be slow if the set grows large.
        # However, the set of smooth numbers is small for small B.
        
        found_pair = None
        # Optimization: Only check if the new numbers could form a pair, 
        # or if the complement of an old number is a new number?
        # Actually, if a pair (x, y) is valid for bound B, both x and y are B-smooth.
        # If they were both (B_prev)-smooth, we would have found them earlier.
        # So at least one of them must be a "new" number (divisible by B).
        # So we can just check the new numbers.
        for a in new_numbers:
            b = N - a
            if b in smooth_numbers:
                # Found a pair
                # Calculate actual largest prime factors
                pa = largest_prime_factor(a, primes)
                pb = largest_prime_factor(b, primes)
                return [a, b]
                
        # Also check if a=1 or b=1 cases are handled?
        # 1 is in smooth_numbers from the start.
        # If N-1 is smooth, it will be in new_numbers or smooth_numbers.
        # If N-1 was smooth for a previous B, it would have been found.
        # If N-1 becomes smooth for current B, it will be in new_numbers.
        # Then b = N - (N-1) = 1, which is in smooth_numbers.
        # So this logic holds.
        
    # Fallback if no pair found (should not happen for N=123457)
    # Return a trivial decomposition
    return [1, N-1]

def largest_prime_factor(n, primes):
    if n == 1:
        return 1
    lp = 1
    for p in primes:
        if p * p > n:
            break
        if n % p == 0:
            lp = p
            while n % p == 0:
                n //= p
    if n > 1:
        lp = n
    return lp
