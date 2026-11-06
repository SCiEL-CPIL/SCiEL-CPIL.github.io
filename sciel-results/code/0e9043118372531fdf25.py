import time
import random
from itertools import combinations

# Precompute prime factors for all numbers up to 20000
_MAX_SUM = 20000
_pf_cache = [None] * (_MAX_SUM + 1)

def _build_pf_cache():
    """Precompute prime factors for all numbers up to _MAX_SUM."""
    _pf_cache[0] = frozenset()
    _pf_cache[1] = frozenset()
    # Sieve-like approach
    for i in range(2, _MAX_SUM + 1):
        if _pf_cache[i] is None:
            # i is prime
            for j in range(i, _MAX_SUM + 1, i):
                if _pf_cache[j] is None:
                    _pf_cache[j] = frozenset({i})
                else:
                    _pf_cache[j] = _pf_cache[j] | {i}

def _prime_factors(n):
    """Return the set of distinct prime factors of n."""
    if n <= _MAX_SUM and _pf_cache[n] is not None:
        return _pf_cache[n]
    factors = set()
    if n % 2 == 0:
        factors.add(2)
        while n % 2 == 0:
            n //= 2
    f = 3
    while f * f <= n:
        if n % f == 0:
            factors.add(f)
            while n % f == 0:
                n //= f
        f += 2
    if n > 1:
        factors.add(n)
    return factors

def _count_distinct_primes(nums):
    """Count distinct prime factors across all pairwise sums."""
    all_primes = set()
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            s = nums[i] + nums[j]
            all_primes |= _prime_factors(s)
    return len(all_primes)

def _count_distinct_primes_fast(nums):
    """Faster version using precomputed cache."""
    all_primes = set()
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            s = nums[i] + nums[j]
            if s <= _MAX_SUM:
                all_primes |= _pf_cache[s]
            else:
                all_primes |= _prime_factors(s)
    return len(all_primes)

def _is_smooth(n, allowed_primes):
    """Check if all prime factors of n are in allowed_primes."""
    if n <= 1:
        return True
    pf = _prime_factors(n)
    return pf.issubset(allowed_primes)

def _search_smooth_k(allowed_primes, max_k, n_needed, deadline):
    """
    Search for n_needed distinct integers k_i <= max_k such that
    for all i != j, prime_factors(k_i + k_j) is a subset of allowed_primes.
    """
    # Backtracking search
    k_values = []
    
    def backtrack(start_idx, current_k_list):
        if time.time() > deadline:
            return None
        if len(current_k_list) == n_needed:
            return list(current_k_list)
        
        for k in range(start_idx, max_k + 1):
            valid = True
            for existing_k in current_k_list:
                if not _is_smooth(k + existing_k, allowed_primes):
                    valid = False
                    break
            if valid:
                current_k_list.append(k)
                result = backtrack(k + 1, current_k_list)
                if result is not None:
                    return result
                current_k_list.pop()
        return None

    return backtrack(1, [])

def _construct_smooth(nums_prime_target, deadline):
    """
    Try to construct numbers of form k_i * P where P is product of primes in nums_prime_target.
    """
    n = 12
    max_val = 10000
    
    P = 1
    for p in nums_prime_target:
        P *= p
        
    max_k = max_val // P
    if max_k < n:
        return None, float('inf')
        
    # We need 12 distinct k's.
    # The sums k_i + k_j will have prime factors from allowed_primes.
    # The actual numbers are x_i = k_i * P.
    # Sums x_i + x_j = P * (k_i + k_j).
    # Prime factors of x_i + x_j = prime_factors(P) U prime_factors(k_i + k_j).
    # Since prime_factors(k_i + k_j) subset allowed_primes = prime_factors(P),
    # The union is just allowed_primes.
    
    k_list = _search_smooth_k(set(nums_prime_target), max_k, n, deadline)
    if k_list is None:
        return None, float('inf')
        
    nums = [k * P for k in k_list]
    # Verify distinctness and bounds
    if len(set(nums)) != n or any(x > max_val or x < 1 for x in nums):
        return None, float('inf')
        
    score = _count_distinct_primes_fast(nums)
    return nums, score

def _local_search(nums, deadline):
    """Perform local search to minimize distinct prime factors."""
    best = list(nums)
    best_score = _count_distinct_primes_fast(best)
    
    n = len(best)
    max_val = 10000
    
    # Try to improve by replacing elements
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        for i in range(n):
            # Try replacing best[i] with a nearby value
            old_val = best[i]
            # Generate candidate replacements
            candidates = set()
            # Nearby values
            for d in range(1, 80):
                for v in [old_val + d, old_val - d]:
                    if 1 <= v <= max_val and v not in best:
                        candidates.add(v)
            # Also try some random values
            for _ in range(30):
                v = random.randint(1, max_val)
                if v not in best:
                    candidates.add(v)
            
            for v in candidates:
                trial = list(best)
                trial[i] = v
                score = _count_distinct_primes_fast(trial)
                if score < best_score:
                    best = trial
                    best_score = score
                    improved = True
                    break
                elif score == best_score:
                    # Accept equal-score moves to escape local minima
                    if random.random() < 0.05:
                        best = trial
                        improved = True
                        break
    
    return best, best_score

def _ap_search(deadline):
    """Search arithmetic progressions for good starting points."""
    n = 12
    max_val = 10000
    best = None
    best_score = float('inf')
    
    # Try various arithmetic progressions: a, a+d, a+2d, ..., a+11d
    # Sums: 2a + (i+j)*d for i < j
    # We want these to have few prime factors
    
    # Try small steps and offsets
    for d in [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 18, 20, 21, 24, 30]:
        if time.time() > deadline - 5:
            break
        max_a = max_val - (n - 1) * d
        if max_a < 1:
            continue
        # Try various starting points
        step = max(1, max_a // 50)
        for a in range(1, max_a + 1, step):
            nums = [a + i * d for i in range(n)]
            score = _count_distinct_primes_fast(nums)
            if score < best_score:
                best_score = score
                best = list(nums)
            if score <= 3:  # Can't do much better than this
                break
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    # Build prime factor cache
    _build_pf_cache()
    
    n = 12
    max_val = 10000
    
    best_result = list(range(1, 13))
    best_score = _count_distinct_primes_fast(best_result)
    
    random.seed(42)
    
    # Strategy 1: Construct smooth numbers with specific prime targets
    # This is the most promising for a significant improvement.
    prime_sets_to_try = [
        [2, 3, 5, 7],
        [2, 3, 5],
        [2, 3],
        [2, 5],
        [3, 5],
        [2, 3, 5, 7, 11],
        [2, 3, 5, 7, 13],
    ]
    
    for primes in prime_sets_to_try:
        if time.time() > deadline - 10:
            break
        result, score = _construct_smooth(primes, deadline)
        if result is not None and score < best_score:
            best_score = score
            best_result = result
            
    # Strategy 2: Arithmetic progression search
    result, score = _ap_search(deadline)
    if result is not None and score < best_score:
        best_score = score
        best_result = result
    
    # Strategy 3: Local search from various starting points
    # Starting point: all even numbers
    start_even = list(range(2, 2*n+1, 2))
    if len(start_even) == n:
        result, score = _local_search(start_even, deadline)
        if score < best_score:
            best_score = score
            best_result = result
    
    # Starting point: multiples of 6
    start_mult6 = [i*6 for i in range(1, n+1)]
    if all(x <= max_val for x in start_mult6):
        result, score = _local_search(start_mult6, deadline)
        if score < best_score:
            best_score = score
            best_result = result
    
    # Random restarts
    restarts = 0
    while time.time() < deadline - 10:
        restarts += 1
        # Generate a random starting set
        nums = random.sample(range(1, max_val+1), n)
        result, score = _local_search(nums, deadline)
        if score < best_score:
            best_score = score
            best_result = result
        if restarts >= 50:
            break
    
    # Final polish: try to improve the best result
    if time.time() < deadline - 2:
        result, score = _local_search(best_result, deadline)
        if score < best_score:
            best_score = score
            best_result = result
    
    return best_result
