"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random

# EVOLVE-BLOCK-START
def _factorize(n):
    """Return set of prime factors of n."""
    if n <= 1:
        return set()
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

def _count_distinct_primes(arr):
    """Count distinct prime factors across all pairwise sums of distinct elements."""
    prime_set = set()
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            s = arr[i] + arr[j]
            prime_set |= _factorize(s)
    return len(prime_set)

def _get_prime_set(arr):
    """Get the set of distinct prime factors across all pairwise sums."""
    prime_set = set()
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            s = arr[i] + arr[j]
            prime_set |= _factorize(s)
    return prime_set

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin before 90s limit
    
    # Start with multiples of 6
    best_arr = list(range(6, 78, 6))  # 6, 12, ..., 72
    best_count = _count_distinct_primes(best_arr)
    
    # Precompute candidate multiples of 6 up to 10000
    candidates = list(range(6, 10001, 6))
    
    # Local search: try replacing one element at a time
    random.seed(42)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(12):
            if time.time() >= deadline:
                break
            # Try various candidates to replace arr[idx]
            # Sample a subset of candidates for efficiency
            trial_candidates = random.sample(candidates, min(200, len(candidates)))
            for new_val in trial_candidates:
                if new_val in best_arr:
                    continue
                trial = best_arr[:]
                trial[idx] = new_val
                cnt = _count_distinct_primes(trial)
                if cnt < best_count:
                    best_count = cnt
                    best_arr = trial
                    improved = True
                    break
    
    # Second phase: try pairs of replacements
    if time.time() < deadline:
        for idx1 in range(12):
            if time.time() >= deadline:
                break
            for idx2 in range(idx1 + 1, 12):
                if time.time() >= deadline:
                    break
                trial_c1 = random.sample(candidates, min(50, len(candidates)))
                trial_c2 = random.sample(candidates, min(50, len(candidates)))
                for v1 in trial_c1:
                    if v1 in best_arr:
                        continue
                    for v2 in trial_c2:
                        if v2 in best_arr or v2 == v1:
                            continue
                        trial = best_arr[:]
                        trial[idx1] = v1
                        trial[idx2] = v2
                        cnt = _count_distinct_primes(trial)
                        if cnt < best_count:
                            best_count = cnt
                            best_arr = trial
                            break
    
    return best_arr
# EVOLVE-BLOCK-END
