import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin for safety
    
    target = 123457
    # Fallback solution
    best_a, best_b = 1, target - 1
    
    # Generate primes up to a reasonable limit
    # The largest prime factor of any number <= target is <= target.
    # But we expect to find a solution with much smaller primes.
    # Let's sieve up to 10000. If not found, we can extend or fall back.
    limit_primes = 10000
    is_prime = [True] * (limit_primes + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.sqrt(limit_primes)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit_primes + 1, i):
                is_prime[j] = False
    primes = [i for i, prime in enumerate(is_prime) if prime]
    
    # Set of smooth numbers generated so far
    smooth_nums = {1}
    
    for p in primes:
        if time.time() > deadline:
            break
            
        # Generate new smooth numbers involving prime p
        # We iterate over the current set and multiply by p^k
        new_nums = set()
        # To avoid modifying set while iterating, we use a list or copy
        # But the set grows. It's safer to iterate over a snapshot or just
        # generate all multiples of p from existing numbers.
        # Efficient way:
        # For each x in smooth_nums, compute x*p, x*p^2, ...
        # However, smooth_nums changes as we add to it? No, we add to new_nums first.
        # But we need to generate based on the OLD smooth_nums?
        # Actually, numbers like 2*3 = 6 are generated when p=3 from x=2 (which was in S from p=2).
        # Numbers like 3*3 = 9 are generated from x=3? No, 3 is not in S yet.
        # Wait. S contains numbers formed by primes < p.
        # When we add p, we form x * p^k where x is formed by primes < p.
        # So we should iterate over the S BEFORE adding the new numbers from p.
        
        old_smooth = list(smooth_nums)
        for x in old_smooth:
            val = x
            while val * p <= target:
                val *= p
                new_nums.add(val)
        
        smooth_nums.update(new_nums)
        
        # Check if solution exists
        # We need a in smooth_nums and target - a in smooth_nums
        # To speed up, we can iterate over the smaller set or just one of them
        # Since smooth_nums is a set, lookup is O(1)
        # We can break early if found
        found = False
        # Optimization: only check a up to target/2 to avoid duplicate checks?
        # Not strictly necessary but good practice.
        # Actually, just iterating over smooth_nums is fine.
        # If a is in S, b=target-a. If b is in S, we found it.
        # We can restrict a to be <= target/2 to ensure a <= b, but not required.
        
        # To make it faster, we can check if target - x is in smooth_nums
        # for x in smooth_nums.
        # Since smooth_nums can be large, let's see its size.
        # For p=7, size is small. For p=100, size is larger.
        # Let's just iterate.
        
        for a in smooth_nums:
            b = target - a
            if b in smooth_nums:
                # Found optimal solution
                return [a, b]
                
    # If loop finishes without finding a solution (or timeout), return fallback
    return [best_a, best_b]
