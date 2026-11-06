"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import random
from itertools import combinations

def _factorize(n):
    """Return set of prime factors of n."""
    if n <= 1:
        return set()
    primes = set()
    d = 2
    while d * d <= n:
        if n % d == 0:
            primes.add(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        primes.add(n)
    return primes

def _count_distinct_primes(numbers):
    """Count distinct prime factors across all pairwise sums."""
    all_primes = set()
    n = len(numbers)
    for i in range(n):
        for j in range(i + 1, n):
            s = numbers[i] + numbers[j]
            all_primes |= _factorize(s)
    return len(all_primes), all_primes

def construct():
    # Start with a good candidate: multiples of 252
    # 252 * 12 = 3024 <= 10000
    # Primes in sums: 2, 3, 5, 7, 11, 13, 17, 19, 23 (Count 9)
    
    # Try to improve via local search
    # Initial set: multiples of 30, scaled to fit
    # Let's try multiples of 20: 20, 40, ..., 240
    # Sums: 20(k+l). k+l in 3..23.
    # Primes in 20: 2, 5.
    # Primes in k+l: 2, 3, 5, 7, 11, 13, 17, 19, 23.
    # Union: 2, 3, 5, 7, 11, 13, 17, 19, 23. Count 9.
    
    # Let's try a random local search
    random.seed(42)
    
    def get_primes(nums):
        return _count_distinct_primes(nums)[1]
    
    # Start with multiples of 24 (2,3)
    # 24, 48, ..., 288
    # Sums 24(k+l).
    # Primes in 24: 2, 3.
    # Primes in k+l: 2, 3, 5, 7, 11, 13, 17, 19, 23.
    # Union: 9.
    
    # Try to eliminate 23.
    # We need to avoid sums divisible by 23? No, sums *equal* to multiples of 23?
    # If a sum is 23, 46, 69...
    # With small numbers, sums are small.
    
    # Let's run a simple hill climbing
    # Start with 12 random multiples of 10
    best_nums = list(range(10, 121, 10)) # 10, 20, ..., 120
    best_count, best_primes = _count_distinct_primes(best_nums)
    
    improved = True
    iterations = 0
    while improved and iterations < 1000:
        improved = False
        iterations += 1
        for i in range(12):
            # Try replacing nums[i] with a random multiple of 10
            for _ in range(20):
                new_val = random.randint(1, 1000) * 10
                if new_val in best_nums:
                    continue
                old_val = best_nums[i]
                best_nums[i] = new_val
                count, primes = _count_distinct_primes(best_nums)
                if count < best_count:
                    best_count = count
                    best_primes = primes
                    improved = True
                else:
                    best_nums[i] = old_val
    
    # Also try the structured candidates from the original code
    candidates = [best_nums]
    
    for M in [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 40, 48, 60, 120, 180, 210, 240, 360, 420, 504, 600, 840, 1008, 1260, 1680, 2520]:
        if M * 12 <= 10000:
            nums = list(range(M, M * 12 + 1, M))
            candidates.append(nums)
            
    # Evaluate all
    final_best = None
    final_count = float('inf')
    for nums in candidates:
        if not nums or len(set(nums)) != 12:
            continue
        if any(x < 1 or x > 10000 for x in nums):
            continue
        count, primes = _count_distinct_primes(nums)
        if count < final_count:
            final_count = count
            final_best = nums
            
    if final_best is None:
        return list(range(1, 13))
        
    return final_best
# EVOLVE-BLOCK-END
