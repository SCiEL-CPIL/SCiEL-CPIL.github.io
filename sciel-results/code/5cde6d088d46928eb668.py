import math
import time

def construct():
    N = 123457
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 90s limit
    
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
        """Generate all numbers <= limit whose prime factors are all in primes."""
        nums = set([1])
        for p in primes:
            new_nums = set()
            for n in nums:
                val = n
                while val * p <= limit:
                    val *= p
                    new_nums.add(val)
            nums.update(new_nums)
        return nums
    
    # Try increasing sets of primes: {2}, {2,3}, {2,3,5}, {2,3,5,7}, ...
    for k in range(1, 30):  # Up to 29th prime (109)
        if time.time() > deadline:
            break
        
        primes = get_primes(k)
        smooths = generate_smooth_numbers(primes, N)
        
        # Check for pair a, b such that a+b=N, both in smooths
        found_pair = None
        for a in smooths:
            b = N - a
            if b in smooths and a >= 1 and b >= 1:
                found_pair = (a, b)
                break
        
        if found_pair:
            return [found_pair[0], found_pair[1]]
    
    # Fallback: return a valid pair
    return [1, N - 1]
