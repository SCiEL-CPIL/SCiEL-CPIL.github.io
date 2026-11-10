"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import Counter

# EVOLVE-BLOCK-START

def _prime_factors(n):
    """Return set of prime factors of n."""
    factors = set()
    if n <= 1:
        return factors
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors

def _score_set(arr):
    """Count distinct prime factors across all pairwise sums."""
    prime_set = set()
    n = len(arr)
    for i in range(n):
        for j in range(i+1, n):
            s = arr[i] + arr[j]
            prime_set.update(_prime_factors(s))
    return len(prime_set)

def _prime_factors_fast(n, smallest_prime):
    """Use precomputed smallest prime factor for speed."""
    factors = set()
    if n <= 1:
        return factors
    while n > 1:
        p = smallest_prime[n]
        factors.add(p)
        while n % p == 0:
            n //= p
    return factors

def _score_set_fast(arr, smallest_prime):
    """Fast scoring using precomputed smallest prime factor."""
    prime_set = set()
    n = len(arr)
    for i in range(n):
        ai = arr[i]
        for j in range(i+1, n):
            s = ai + arr[j]
            if s > 1:
                m = s
                while m > 1:
                    p = smallest_prime[m]
                    prime_set.add(p)
                    while m % p == 0:
                        m //= p
    return len(prime_set)

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s deadline, 10s safety margin
    
    # Precompute smallest prime factor up to 20000
    MAX_VAL = 20000
    smallest_prime = list(range(MAX_VAL + 1))
    for i in range(2, int(MAX_VAL**0.5) + 1):
        if smallest_prime[i] == i:
            for j in range(i*i, MAX_VAL + 1, i):
                if smallest_prime[j] == j:
                    smallest_prime[j] = i
    
    # Initial candidate: try multiples of 6 (all sums divisible by 6)
    # But we need to minimize OTHER prime factors too
    # Start with a random set and improve
    
    random.seed(42)
    
    # Try several initial strategies
    best_set = None
    best_score = float('inf')
    
    # Strategy 1: Small consecutive even numbers
    candidates = list(range(2, 10000, 2))
    initial = candidates[:12]
    score = _score_set_fast(initial, smallest_prime)
    if score < best_score:
        best_score = score
        best_set = initial[:]
    
    # Strategy 2: Multiples of 6
    candidates6 = list(range(6, 10000, 6))
    initial6 = candidates6[:12]
    score6 = _score_set_fast(initial6, smallest_prime)
    if score6 < best_score:
        best_score = score6
        best_set = initial6[:]
    
    # Strategy 3: Random search with local improvement
    for trial in range(50):
        if time.time() > deadline:
            break
        # Random initial set
        rand_set = random.sample(range(1, 10001), 12)
        rand_set.sort()
        s = _score_set_fast(rand_set, smallest_prime)
        if s < best_score:
            best_score = s
            best_set = rand_set[:]
        
        # Local search: try swapping one element
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(12):
                if time.time() > deadline:
                    break
                old = best_set[i]
                # Try replacing with a random value
                for _ in range(20):
                    new_val = random.randint(1, 10000)
                    if new_val in best_set and new_val != old:
                        continue
                    test_set = best_set[:]
                    test_set[i] = new_val
                    test_set.sort()
                    ns = _score_set_fast(test_set, smallest_prime)
                    if ns < best_score:
                        best_score = ns
                        best_set = test_set
                        improved = True
                        break
    
    # More systematic local search on the best set
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(12):
                if time.time() > deadline:
                    break
                old = best_set[i]
                # Try nearby values
                for offset in range(-50, 51):
                    if offset == 0:
                        continue
                    new_val = old + offset
                    if new_val < 1 or new_val > 10000:
                        continue
                    if new_val in best_set:
                        continue
                    test_set = best_set[:]
                    test_set[i] = new_val
                    test_set.sort()
                    ns = _score_set_fast(test_set, smallest_prime)
                    if ns < best_score:
                        best_score = ns
                        best_set = test_set
                        improved = True
                        break
    
    return best_set

# EVOLVE-BLOCK-END
