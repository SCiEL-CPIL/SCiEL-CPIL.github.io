import time
import random
from itertools import combinations
from collections import defaultdict

# ---------- primes and factorization ----------

def get_primes(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

PRIMES = get_primes(20000)
PRIME_SET = set(PRIMES)

def factorize(n):
    factors = set()
    d = 2
    while d * d <= n:
        if n % d == 0:
            while n % d == 0:
                n //= d
            factors.add(d)
        d += 1 if d == 2 else 2
    if n > 1:
        factors.add(n)
    return factors

# Precompute prime factors for all possible sums (2..20000)
SUM_PRIME_FACTORS = [set() for _ in range(20001)]
for s in range(2, 20001):
    SUM_PRIME_FACTORS[s] = factorize(s)

def count_distinct_primes(nums):
    all_primes = set()
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            all_primes.update(SUM_PRIME_FACTORS[nums[i] + nums[j]])
    return len(all_primes)

# ---------- candidate pool ----------

def generate_smooth_numbers(max_val, primes):
    nums = {1}
    for p in primes:
        current = list(nums)
        for x in current:
            y = x
            while y * p <= max_val:
                y *= p
                nums.add(y)
    return sorted(nums)

# Build a focused candidate pool: small integers + smooth numbers
SMOOTH_2357 = generate_smooth_numbers(10000, [2, 3, 5, 7])
SMALL_INTS = list(range(1, 301))
CANDIDATE_POOL = sorted(set(SMALL_INTS + SMOOTH_2357))
# Keep pool size manageable for graph construction
if len(CANDIDATE_POOL) > 800:
    CANDIDATE_POOL = CANDIDATE_POOL[:800]

# ---------- backtracking with forward checking ----------

def backtracking_select(pool, prime_set, target_size=12, deadline=None):
    """
    Select target_size distinct values from pool such that all pairwise sums
    have prime factors only in prime_set. Uses backtracking with forward checking.
    """
    if deadline is None:
        deadline = time.time() + 10
    
    # Precompute compatibility: compat[a] = set of b in pool where a+b is S-smooth
    # Only consider pairs where both are in pool
    pool_set = set(pool)
    compat = defaultdict(set)
    for i, a in enumerate(pool):
        for b in pool[i+1:]:
            pf = SUM_PRIME_FACTORS[a + b]
            if pf.issubset(prime_set):
                compat[a].add(b)
                compat[b].add(a)
    
    # Sort pool by degree (most connected first) for better pruning
    sorted_pool = sorted(pool, key=lambda x: -len(compat[x]))
    
    result = []
    
    def backtrack(remaining, current):
        if time.time() > deadline:
            return False
        if len(current) == target_size:
            result.extend(current)
            return True
        if not remaining:
            return False
        
        # Try candidates in order of decreasing compatibility with current
        candidates = sorted(remaining, key=lambda x: -sum(1 for c in current if x in compat[c]))
        
        for cand in candidates:
            # Check if cand is compatible with all in current
            if all(cand in compat[c] for c in current):
                # Forward check: ensure enough remaining candidates can complete
                new_remaining = [x for x in remaining if x != cand and all(x in compat[c] for c in current + [cand])]
                if len(new_remaining) >= target_size - len(current) - 1:
                    if backtrack(new_remaining, current + [cand]):
                        return True
        
        return False
    
    backtrack(sorted_pool, [])
    return result if len(result) == target_size else None

# ---------- local search ----------

def local_search(initial_set, pool, deadline, max_iters=3000):
    s = initial_set[:]
    score = count_distinct_primes(s)
    s_set = set(s)
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        i = random.randint(0, 11)
        old_val = s[i]
        new_val = random.choice(pool)
        if new_val in s_set:
            continue
        s[i] = new_val
        s_set.discard(old_val)
        s_set.add(new_val)
        new_score = count_distinct_primes(s)
        if new_score <= score:
            score = new_score
        else:
            s[i] = old_val
            s_set.discard(new_val)
            s_set.add(old_val)
    return s, score

# ---------- main construct ----------

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill
    
    best_set = None
    best_score = float('inf')
    
    # Strategy: Try small prime sets S in order, use backtracking to find 12 values
    # whose pairwise sums are S-smooth
    
    # Candidate prime sets to try, in order of increasing size
    candidate_prime_sets = [
        {2, 3, 5},
        {2, 3, 5, 7},
        {2, 3, 5, 7, 11},
        {2, 3, 5, 7, 11, 13},
        {2, 3, 5, 7, 11, 13, 17},
        {2, 3, 5, 7, 11, 13, 17, 19},
    ]
    
    for S in candidate_prime_sets:
        if time.time() > deadline - 20:
            break
        
        # Use only smooth numbers as candidates (they're more likely to work)
        smooth_candidates = SMOOTH_2357[:200]
        
        # Try backtracking with this prime set
        bt_deadline = min(deadline - 15, time.time() + 12)
        result = backtracking_select(smooth_candidates, S, 12, bt_deadline)
        
        if result and len(result) == 12:
            score = count_distinct_primes(result)
            if score < best_score:
                best_score = score
                best_set = result
                # Found a solution; try to improve with local search but keep going
                # to see if a smaller prime set works (unlikely but check)
                break
        
        # If backtracking fails, try a smaller pool
        if time.time() < deadline - 20:
            smaller_pool = SMOOTH_2357[:100]
            bt_deadline = min(deadline - 15, time.time() + 8)
            result = backtracking_select(smaller_pool, S, 12, bt_deadline)
            if result and len(result) == 12:
                score = count_distinct_primes(result)
                if score < best_score:
                    best_score = score
                    best_set = result
                    break
    
    # Local search refinement
    if time.time() < deadline - 5:
        # Refine the best solution found
        if best_set is not None:
            refined, refined_score = local_search(best_set, CANDIDATE_POOL, deadline - 2)
            if refined_score < best_score:
                best_score = refined_score
                best_set = refined
        
        # Try a few random initializations from smooth numbers
        smooth_pool = SMOOTH_2357[:100]
        for restart in range(5):
            if time.time() > deadline - 3:
                break
            init_set = random.sample(smooth_pool, 12)
            s, score = local_search(init_set, smooth_pool, deadline - 2, max_iters=1500)
            if score < best_score:
                best_score = score
                best_set = s
    
    # Fallback
    if best_set is None:
        best_set = SMOOTH_2357[:12]
    
    return best_set
