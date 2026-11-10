"""
Structural Change: Clique Search on Prime-Compatibility Graphs.
Instead of generic local search, we formulate the problem as finding a clique of size 12
in a graph where vertices are integers 1..10000 and an edge exists between u and v 
if the prime factors of (u+v) are a subset of a predefined small set of primes S.
We iterate over small sets S (e.g., {2,3}, {2,3,5}, {2,3,5,7}) and attempt to find 
such a clique using a greedy heuristic. If a clique is found, the score is bounded 
by |S|. If not, we use the best partial clique as a seed for a targeted local search.
"""
import time
import math
import random
import sys

_START_TIME = time.time()
_DEADLINE_SECONDS = 85.0
MAX_VAL = 10000
N = 12

def _elapsed():
    return time.time() - _START_TIME

def _deadline_hit():
    return _elapsed() > _DEADLINE_SECONDS

# ---------- Precomputation ----------
_MAX_SUM = 2 * MAX_VAL

# Generate primes up to 20000
def _sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i, v in enumerate(is_prime) if v]

_primes = _sieve(_MAX_SUM)
_prime_to_bit = {p: i for i, p in enumerate(_primes)}

# Precompute bitmask of prime factors for all sums 2..20000
# Using a smallest prime factor (SPF) sieve for speed in factorization
spf = list(range(_MAX_SUM + 1))
for i in range(2, int(_MAX_SUM**0.5) + 1):
    if spf[i] == i:
        for j in range(i*i, _MAX_SUM + 1, i):
            if spf[j] == j:
                spf[j] = i

_mask_table = [0] * (_MAX_SUM + 1)
for n in range(2, _MAX_SUM + 1):
    x = n
    mask = 0
    while x > 1:
        p = spf[x]
        mask |= (1 << _prime_to_bit[p])
        while x % p == 0:
            x //= p
    _mask_table[n] = mask

def _score(nums):
    m = 0
    n = len(nums)
    for i in range(n):
        ai = nums[i]
        for j in range(i + 1, n):
            m |= _mask_table[ai + nums[j]]
    return bin(m).count("1")

# ---------- Clique Search Heuristic ----------
def _find_clique_for_prime_set(target_primes, limit=10000):
    """
    Attempts to find a clique of size N in the graph where edge (u,v) exists 
    iff prime_factors(u+v) is a subset of target_primes.
    Uses a greedy approach with backtracking-lite (restarts).
    """
    target_mask = 0
    for p in target_primes:
        if p in _prime_to_bit:
            target_mask |= (1 << _prime_to_bit[p])
    
    # Pre-filter candidates: 
    # If u is very large, u+v is large. 
    # We just iterate through all numbers 1..limit.
    # To make it fast, we can group numbers by their "type" relative to target primes?
    # Simpler: Greedy selection.
    
    best_clique = []
    best_size = 0
    
    # We try multiple random orderings to escape local greedy traps
    rng = random.Random(12345)
    candidates = list(range(1, limit + 1))
    
    for attempt in range(50):
        if _deadline_hit():
            break
        rng.shuffle(candidates)
        current_clique = []
        
        for u in candidates:
            # Check if u can be added to current_clique
            compatible = True
            for v in current_clique:
                if (_mask_table[u + v] & ~target_mask) != 0:
                    compatible = False
                    break
            if compatible:
                current_clique.append(u)
                if len(current_clique) >= N:
                    return current_clique
        
        if len(current_clique) > best_size:
            best_size = len(current_clique)
            best_clique = current_clique[:]
            
    return best_clique

# ---------- Local Search Refinement ----------
def _local_search(nums, budget_iters=50000):
    """
    Hill-climb to minimize score. 
    Uses bitmask scoring for speed.
    """
    current = sorted(nums)
    current_score = _score(current)
    iters = 0
    improved = True
    rng = random.Random(42)
    
    # Precompute a list of "good" candidates: numbers that are multiples of small highly composite numbers
    # This biases the search towards numbers that share factors in sums.
    good_candidates = set()
    for k in (6, 12, 24, 30, 60, 120, 240, 360, 720):
        x = k
        while x <= MAX_VAL:
            good_candidates.add(x)
            x += k
    
    # Also add numbers that are powers of 2, 3, 5
    for base in (2, 3, 5):
        x = base
        while x <= MAX_VAL:
            good_candidates.add(x)
            x *= base
            
    good_candidates_list = sorted(good_candidates)
    
    while improved and iters < budget_iters and not _deadline_hit():
        improved = False
        iters += 1
        n = len(current)
        
        # Try single-element replacement
        for idx in range(n):
            old = current[idx]
            best_new = None
            best_new_score = current_score
            
            # Candidate generation
            candidates = set()
            
            # 1. Neighborhood
            for d in range(-10, 11):
                v = old + d
                if 1 <= v <= MAX_VAL:
                    candidates.add(v)
            
            # 2. Good candidates near old
            # Find position of old in good_candidates_list
            # Since good_candidates_list is sorted, we can use bisect
            import bisect
            pos = bisect.bisect_left(good_candidates_list, old)
            for i in range(max(0, pos-10), min(len(good_candidates_list), pos+11)):
                candidates.add(good_candidates_list[i])
                
            # 3. Random good candidates
            for _ in range(5):
                if good_candidates_list:
                    candidates.add(rng.choice(good_candidates_list))
            
            # 4. Random random
            for _ in range(5):
                candidates.add(rng.randint(1, MAX_VAL))

            for v in candidates:
                if v == old or v in current:
                    continue
                # Fast check: does this v introduce new primes?
                # We calculate the new score.
                # Optimization: We can compute the mask of the new set quickly?
                # No, just recompute full score for simplicity, it's O(66) bit ops.
                new_list = current[:]
                new_list[idx] = v
                ns = _score(new_list)
                if ns < best_new_score:
                    best_new_score = ns
                    best_new = (idx, v)
            
            if best_new is not None:
                idx, v = best_new
                current[idx] = v
                current = sorted(current)
                current_score = best_new_score
                improved = True
                break
        
        # If no single improvement, try a 2-swap? 
        # For now, stick to single replacement to keep it fast.
        
    return current, current_score

def construct():
    rng = random.Random(987654321)
    best_result = None
    best_score = float("inf")
    
    # Strategy 1: Try to find a clique with small prime sets
    # This gives a hard upper bound on the score.
    prime_sets_to_try = [
        [2, 3],
        [2, 3, 5],
        [2, 3, 5, 7],
        [2, 3, 5, 7, 11],
        [2, 3, 5, 7, 11, 13]
    ]
    
    initial_seed = None
    
    for p_set in prime_sets_to_try:
        if _deadline_hit():
            break
        # Try to find a clique of size 12
        clique = _find_clique_for_prime_set(p_set)
        
        if len(clique) >= N:
            # We found a set of 12 numbers where all sums only use primes in p_set
            # The score is at most len(p_set).
            candidate = clique[:N]
            sc = _score(candidate)
            if sc < best_score:
                best_score = sc
                best_result = candidate[:]
            # Since we found a valid clique for this small set, 
            # it's likely optimal or near-optimal. 
            # We can try to polish it, but it's already very good.
            # Let's run a quick local search to see if we can reduce score further 
            # (e.g. if some primes in p_set weren't actually used, or if we can swap to use even fewer).
            # Actually, if we swap, we might break the clique property and increase score.
            # So we just return this if it's good enough.
            # But let's do a small polish just in case.
            if not _deadline_hit():
                polished, sc2 = _local_search(candidate, budget_iters=2000)
                if sc2 < best_score:
                    best_score = sc2
                    best_result = polished
            # If we found a clique for {2,3,5}, score is <= 3. 
            # It's very hard to beat 3 (minimum possible is 2? sums of 2 distinct ints >= 3, so 3 is a prime. 
            # Can we have score 2? Primes must be {2,3}. Sums must be 2^a 3^b.
            # If we found it, great. If not, we move to next set.
            
            # If we found a solution with score <= 5, it's probably optimal.
            if best_score <= 5:
                # Quick check: can we do better? 
                # If score is 3, min possible is 2? 
                # Let's just return if we have a very low score.
                # But let's try the next prime set just in case {2,3} works? 
                # The loop tries {2,3} first. If it fails, it tries {2,3,5}.
                # So if we are here, {2,3} failed.
                # So best_score is likely 3 or 4.
                # We can stop if we find a clique for {2,3,5}.
                break

    # Strategy 2: If no small clique found, use local search with diverse seeds
    if best_result is None or best_score > 10:
        seeds = []
        # Multiples of 30
        for k in (30, 60, 120):
            s = []
            x = k
            while len(s) < N and x <= MAX_VAL:
                s.append(x)
                x += k
            if len(s) == N:
                seeds.append(s)
        
        # Random multiples of 6
        for _ in range(10):
            candidates = [x for x in range(1, MAX_VAL + 1) if x % 6 == 0]
            seeds.append(sorted(rng.sample(candidates, N)))
            
        # Random general
        for _ in range(10):
            seeds.append(sorted(rng.sample(range(1, MAX_VAL + 1), N)))

        for seed in seeds:
            if _deadline_hit():
                break
            sc = _score(seed)
            if sc < best_score:
                best_score = sc
                best_result = seed[:]
            
            # Local search
            if not _deadline_hit():
                # Budget depends on time remaining
                remaining = _DEADLINE_SECONDS - _elapsed()
                budget = int(min(10000, remaining * 200))
                if budget > 100:
                    improved_list, sc2 = _local_search(seed, budget_iters=budget)
                    if sc2 < best_score:
                        best_score = sc2
                        best_result = improved_list

    # Final safety
    if best_result is None:
        # Fallback: first 12 multiples of 6
        best_result = [6*i for i in range(1, N+1)]
        best_score = _score(best_result)

    # Validation
    assert best_result is not None
    assert len(set(best_result)) == N
    assert all(1 <= x <= MAX_VAL for x in best_result)

    return best_result
