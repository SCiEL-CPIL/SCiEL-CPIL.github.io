import math
import time

def construct():
    N = 123457
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 60s limit
    
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(math.sqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True

    def get_primes(count):
        primes = []
        candidate = 2
        while len(primes) < count:
            if is_prime(candidate):
                primes.append(candidate)
            candidate += 1
        return primes

    def generate_smooth_numbers(primes, limit):
        nums = {1}
        for p in primes:
            new_nums = set()
            for n in nums:
                val = n
                while val * p <= limit:
                    val *= p
                    if val <= limit:
                        new_nums.add(val)
            nums.update(new_nums)
        return nums

    # Try increasing sets of primes
    # Start with 1 prime (2), then 2 primes (2,3), etc.
    for k in range(1, 25): # Up to 25th prime (79)
        if time.time() > deadline:
            break
            
        primes = get_primes(k)
        
        # Generate all k-smooth numbers <= N
        smooths = generate_smooth_numbers(primes, N)
        
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

    # Fallback: Return a valid pair
    # If we haven't found a smooth pair, we need to return *something*.
    # A valid pair is (1, N-1).
    # P(1)=1. P(N-1) is the largest prime factor of N-1.
    # This might be large, but it's valid.
    # However, we should try to find a slightly better pair if time permits.
    # For now, (1, N-1) is safe.
    return [1, N - 1]
