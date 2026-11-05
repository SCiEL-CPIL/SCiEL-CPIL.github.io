"""Improved program: clique-based search over small prime sets."""
import math
import random
import time
from itertools import combinations

# EVOLVE-BLOCK-START

def get_prime_factors(n):
    """Return set of distinct prime factors of n."""
    factors = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors

def count_distinct_primes(numbers):
    """Count distinct prime factors across all pairwise sums."""
    primes = set()
    n = len(numbers)
    for i in range(n):
        for j in range(i + 1, n):
            s = numbers[i] + numbers[j]
            primes |= get_prime_factors(s)
    return len(primes)

def is_smooth(n, prime_set):
    """Check if n is smooth over prime_set (all prime factors in prime_set)."""
    for p in prime_set:
        while n % p == 0:
            n //= p
    return n == 1

def find_numbers_for_primes(prime_set, count=12, max_val=10000, timeout=30):
    """Find 'count' distinct positive integers <= max_val such that all pairwise sums are smooth over prime_set."""
    deadline = time.time() + timeout
    
    # Start with small numbers and try to build up
    best_set = None
    best_score = float('inf')
    
    # Try different starting seeds
    for seed in range(20):
        if time.time() > deadline:
            break
        random.seed(seed)
        
        # Generate candidate pool: numbers that are sums of small multiples of primes
        candidates = set()
        # Add small numbers
        for x in range(1, 200):
            candidates.add(x)
        
        # Add numbers whose prime factors are all in prime_set
        for x in range(1, max_val + 1):
            if is_smooth(x, prime_set):
                candidates.add(x)
            if time.time() > deadline:
                break
        
        candidates = sorted(candidates)
        
        # Greedy + local search
        current = []
        for c in candidates:
            if time.time() > deadline:
                break
            if len(current) >= count:
                break
            # Check if adding c keeps all pairwise sums smooth
            valid = True
            for existing in current:
                if not is_smooth(existing + c, prime_set):
                    valid = False
                    break
            if valid:
                current.append(c)
        
        if len(current) == count:
            score = count_distinct_primes(current)
            if score < best_score:
                best_score = score
                best_set = list(current)
    
    return best_set, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    best_result = None
    best_score = float('inf')
    
    # Try small prime sets first (fewer primes = better score)
    # Prime sets to try, ordered by size
    prime_sets_to_try = [
        {2},
        {2, 3},
        {2, 3, 5},
        {2, 3, 5, 7},
        {2, 3, 5, 7, 11},
        {2, 3, 5, 7, 11, 13},
        {2, 3, 5, 7, 11, 13, 17},
    ]
    
    for ps in prime_sets_to_try:
        if time.time() > deadline:
            break
        
        # Allocate remaining time proportionally
        remaining = deadline - time.time()
        timeout_for_this = max(5, min(remaining / 2, 20))
        
        result, score = find_numbers_for_primes(ps, count=12, max_val=10000, timeout=timeout_for_this)
        
        if result is not None:
            if score < best_score:
                best_score = score
                best_result = result
        
        # If we found a solution with score equal to len(ps), that's optimal for this prime set
        if best_score == len(ps):
            break
    
    # If no solution found, fall back to a reasonable heuristic
    if best_result is None:
        # Use powers of 2: 1,2,4,8,...,2048,4096,8192 -> 12 numbers
        # Sums will be of form 2^a + 2^b = 2^min(a,b)*(1+2^|a-b|)
        # The odd factor is 1+2^k which may have various prime factors
        # Let's try a different approach: use numbers that are all even
        # If all numbers are even, all sums are even, so 2 is always a factor
        # Then we need the odd parts to have few prime factors
        
        # Try: multiples of small smooth numbers
        # Use numbers like 2*k where k is smooth
        smooth_numbers = []
        for x in range(1, 5000):
            if is_smooth(x, {2, 3, 5, 7, 11}):
                smooth_numbers.append(2 * x)
            if len(smooth_numbers) >= 12:
                break
        
        if len(smooth_numbers) >= 12:
            best_result = smooth_numbers[:12]
            best_score = count_distinct_primes(best_result)
        else:
            # Fallback: just use 1-12
            best_result = list(range(1, 13))
            best_score = count_distinct_primes(best_result)
    
    return best_result

# EVOLVE-BLOCK-END
