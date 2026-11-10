import math
import time

def construct():
    N = 123457
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 60s limit
    
    # Pre-compute primes using Sieve of Eratosthenes
    # We need primes up to a reasonable limit. 
    # The largest prime factor in a minimal solution is likely small.
    # Let's go up to 200.
    limit_primes = 200
    sieve = [True] * (limit_primes + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(math.sqrt(limit_primes)) + 1):
        if sieve[i]:
            for j in range(i*i, limit_primes + 1, i):
                sieve[j] = False
    primes = [i for i, is_p in enumerate(sieve) if is_p]
    
    def generate_smooth_numbers(prime_list, limit):
        """
        Generate all numbers <= limit whose prime factors are in prime_list.
        Uses list expansion for speed, then converts to set for lookup.
        """
        nums = [1]
        for p in prime_list:
            new_nums = []
            for n in nums:
                val = n
                while val * p <= limit:
                    val *= p
                    new_nums.append(val)
            nums.extend(new_nums)
        return set(nums)
    
    # Iterate through increasing sets of primes
    for k in range(1, len(primes)):
        if time.time() > deadline:
            break
            
        current_primes = primes[:k]
        smooths = generate_smooth_numbers(current_primes, N)
        
        # Check for pair a, b such that a+b=N
        found_pair = None
        for a in smooths:
            b = N - a
            if b in smooths:
                if a >= 1 and b >= 1:
                    found_pair = (a, b)
                    break
        
        if found_pair:
            return [found_pair[0], found_pair[1]]
    
    # Fallback: return a valid pair
    # If no smooth solution found in time or with small primes
    return [1, N - 1]
