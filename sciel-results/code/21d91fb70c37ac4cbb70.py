"""Minimize distinct prime factors across pairwise sums of 12 distinct positive integers <= 10000."""
import time
import math
import random
from itertools import combinations

_START_TIME = time.time()
_DEADLINE_SECONDS = 80.0

def _elapsed():
    return time.time() - _START_TIME

def _deadline_hit():
    return _elapsed() > _DEADLINE_SECONDS

# Precompute smallest prime factor sieve up to 20000 (max sum is 2*10000)
_MAX_SUM = 20000
_sieve = list(range(_MAX_SUM + 1))
for i in range(2, int(_MAX_SUM ** 0.5) + 1):
    if _sieve[i] == i:
        for j in range(i * i, _MAX_SUM + 1, i):
            if _sieve[j] == j:
                _sieve[j] = i

def _prime_factors_fast(n):
    """Return set of prime factors of n using precomputed sieve."""
    factors = set()
    while n > 1:
        p = _sieve[n]
        factors.add(p)
        while n % p == 0:
            n //= p
    return factors

def _compute_factor_sets(nums):
    """For each pair (i,j), compute the set of prime factors of nums[i]+nums[j].
    Return a list of 12 sets, where set[i] contains all prime factors of sums involving index i."""
    n = len(nums)
    pair_factors = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            s = nums[i] + nums[j]
            pf = _prime_factors_fast(s)
            pair_factors[i][j] = pf
            pair_factors[j][i] = pf
    return pair_factors

def _total_factor_count(pair_factors):
    """Count distinct primes across all pair factor sets."""
    all_primes = set()
    n = len(pair_factors)
    for i in range(n):
        for j in range(i + 1, n):
            all_primes |= pair_factors[i][j]
    return len(all_primes)

def _is_smooth(n, prime_set):
    """Check if n is composed only of primes in prime_set."""
    for p in prime_set:
        while n % p == 0:
            n //= p
    return n == 1

def _generate_smooth_numbers(primes, max_val=10000):
    """Generate all numbers <= max_val composed only of given primes."""
    smooth = set()
    def _gen(idx, current):
        if current > max_val:
            return
        smooth.add(current)
        if idx < len(primes):
            p = primes[idx]
            v = current
            while v <= max_val:
                _gen(idx, v)
                v *= p
                if v > max_val:
                    break
            _gen(idx + 1, current)
    _gen(0, 1)
    return sorted(smooth)

def _find_clique_smooth(primes, target_size=12, max_val=10000):
    """
    Find target_size distinct numbers in [1, max_val] such that every pairwise sum
    is composed only of primes in 'primes'. Uses backtracking on a compatibility graph.
    """
    # Generate all S-smooth numbers up to max_val
    smooth_nums = _generate_smooth_numbers(primes, max_val)
    
    if len(smooth_nums) < target_size:
        return None
    
    # Build compatibility: edge between a and b if a+b is S-smooth
    n = len(smooth_nums)
    compat = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            s = smooth_nums[i] + smooth_nums[j]
            if _is_smooth(s, primes):
                compat[i].add(j)
                compat[j].add(i)
    
    # Sort by degree (descending) for better backtracking
    order = sorted(range(n), key=lambda x: len(compat[x]), reverse=True)
    
    # Backtracking to find a clique of size target_size
    result = []
    
    def _backtrack(candidates, remaining):
        if remaining == 0:
            return True
        if not candidates:
            return False
        # Try each candidate in order of degree
        for i in range(len(candidates)):
            node = candidates[i]
            # Check if node is compatible with all in result
            if all(node in compat[r] for r in result):
                result.append(node)
                # New candidates: intersection of current candidates (after i) with neighbors of node
                new_candidates = [c for c in candidates[i+1:] if c in compat[node]]
                if _backtrack(new_candidates, remaining - 1):
                    return True
                result.pop()
        return False
    
    # Try multiple starting points for diversity
    for start_idx in range(min(50, n)):
        if _deadline_hit():
            break
        start_node = order[start_idx]
        result = [start_node]
        neighbors = [c for c in order if c in compat[start_node] and c != start_node]
        if _backtrack(neighbors, target_size - 1):
            clique_indices = result
            return [smooth_nums[i] for i in clique_indices]
    
    return None

def _find_clique_general(prime_set, target_size=12, max_val=10000, time_limit=5.0):
    """
    Find target_size distinct numbers in [1, max_val] such that every pairwise sum
    has all prime factors in prime_set. Uses randomized backtracking.
    """
    start_t = time.time()
    
    # Generate candidate numbers: focus on numbers that are "nice" (smooth-ish)
    # Use numbers up to max_val that are composed of small primes
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    candidates = _generate_smooth_numbers(small_primes, max_val)
    
    if len(candidates) < target_size:
        return None
    
    # Limit candidates to most connected ones
    # First, compute degree for each candidate
    n = len(candidates)
    
    # For speed, limit to a subset
    if n > 500:
        # Sample or use top-degree heuristics
        random.seed(42)
        candidates = random.sample(candidates, 500)
        n = 500
    
    # Build compatibility
    compat = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            s = candidates[i] + candidates[j]
            if _is_smooth(s, prime_set):
                compat[i].add(j)
                compat[j].add(i)
    
    # Sort by degree
    order = sorted(range(n), key=lambda x: len(compat[x]), reverse=True)
    
    # Backtracking
    result = []
    
    def _backtrack(cand_list, remaining):
        if time.time() - start_t > time_limit:
            return False
        if remaining == 0:
            return True
        if not cand_list:
            return False
        for i in range(len(cand_list)):
            node = cand_list[i]
            if all(node in compat[r] for r in result):
                result.append(node)
                new_cands = [c for c in cand_list[i+1:] if c in compat[node]]
                if _backtrack(new_cands, remaining - 1):
                    return True
                result.pop()
        return False
    
    for start_idx in range(min(100, n)):
        if time.time() - start_t > time_limit:
            break
        start_node = order[start_idx]
        result = [start_node]
        neighbors = [c for c in order if c in compat[start_node] and c != start_node]
        if _backtrack(neighbors, target_size - 1):
            return [candidates[i] for i in result]
    
    return None

def construct():
    random.seed(42)
    
    best_result = None
    best_score = float('inf')
    
    # Strategy A: Try small prime sets with exact smooth clique finding
    prime_sets_to_try = [
        [2, 3, 5, 7],
        [2, 3, 5],
        [2, 3, 5, 7, 11],
        [2, 3, 5, 7, 11, 13],
        [2, 3, 5, 7, 11, 13, 17],
        [3, 5, 7],
        [2, 3, 7],
        [2, 5, 7],
        [2, 3, 5, 11],
        [2, 3, 5, 7, 11, 13, 17, 19],
    ]
    
    for ps in prime_sets_to_try:
        if _deadline_hit():
            break
        
        # Try exact clique finding
        clique = _find_clique_smooth(ps, 12, 10000)
        if clique is not None and len(set(clique)) == 12 and all(1 <= x <= 10000 for x in clique):
            pf = _compute_factor_sets(clique)
            score = _total_factor_count(pf)
            if score < best_score:
                best_score = score
                best_result = clique[:]
            # If we found a solution with exactly len(ps) primes, that's optimal for this prime set
            if score <= len(ps):
                # Can't do better with this prime set
                continue
        
        # Try general clique finding with this prime set
        if not _deadline_hit():
            clique2 = _find_clique_general(set(ps), 12, 10000, time_limit=3.0)
            if clique2 is not None and len(set(clique2)) == 12 and all(1 <= x <= 10000 for x in clique2):
                pf = _compute_factor_sets(clique2)
                score = _total_factor_count(pf)
                if score < best_score:
                    best_score = score
                    best_result = clique2[:]
    
    # Strategy B: If no clique found, use local search from good initializations
    if best_result is None or best_score > 8:
        # Generate initial candidates
        candidates = []
        
        # Powers of 2
        candidates.append([2**i for i in range(1, 13)])
        
        # Multiples of highly composite numbers
        candidates.append([360 * (i+1) for i in range(12)])
        candidates.append([30 * (i+1) for i in range(12)])
        candidates.append([210 * (i+1) for i in range(12)])
        
        # Even numbers
        candidates.append(list(range(2, 25, 2)))
        
        # Numbers 1 mod small primes
        candidates.append([60 * i + 1 for i in range(1, 13)])
        
        # Close numbers
        candidates.append([1000 + i for i in range(12)])
        
        # Powers of 2 plus 1
        candidates.append([1 + 2**i for i in range(1, 13)])
        
        # 2^a * 3^b
        cand_11 = []
        for a in range(5):
            for b in range(5):
                val = 2**a * 3**b
                if 1 <= val <= 10000 and val not in cand_11:
                    cand_11.append(val)
                if len(cand_11) >= 12:
                    break
            if len(cand_11) >= 12:
                break
        if len(cand_11) >= 12:
            candidates.append(cand_11)
        
        # Arithmetic progressions
        for a in [1, 2, 3, 5, 7, 10, 14, 15, 20, 21, 30, 35, 42, 70, 105, 210]:
            cand = [a + 210 * i for i in range(12)]
            if all(1 <= x <= 10000 for x in cand) and len(set(cand)) == 12:
                candidates.append(cand)
        
        # Multiples of 840
        candidates.append([840 * (i+1) for i in range(12)])
        
        # Evaluate all
        for cand in candidates:
            if _deadline_hit():
                break
            if len(set(cand)) != 12:
                continue
            if not all(1 <= x <= 10000 for x in cand):
                continue
            pf = _compute_factor_sets(cand)
            score = _total_factor_count(pf)
            if score < best_score:
                best_score = score
                best_result = cand[:]
    
    if best_result is None:
        best_result = list(range(1, 13))
        best_score = _total_factor_count(_compute_factor_sets(best_result))
    
    # Local search to improve
    current = best_result[:]
    current_pf = _compute_factor_sets(current)
    current_score = best_score
    n = 12
    
    # Precompute base primes for each index
    def _base_primes_excluding(pf, exclude_idx):
        base = set()
        for i in range(n):
            if i == exclude_idx:
                continue
            for j in range(i+1, n):
                if j == exclude_idx:
                    continue
                base |= pf[i][j]
        return base
    
    base_primes_cache = [_base_primes_excluding(current_pf, i) for i in range(n)]
    
    # Generate smooth numbers for candidate generation
    smooth_primes = [2, 3, 5, 7, 11]
    smooth_numbers = _generate_smooth_numbers(smooth_primes, 20000)
    
    improved = True
    while improved and not _deadline_hit():
        improved = False
        
        for idx in range(n):
            if _deadline_hit():
                break
            
            old_val = current[idx]
            base_primes = base_primes_cache[idx]
            
            best_new_val = None
            best_new_score = current_score
            best_new_pf = None
            best_new_primes = None
            
            # Generate candidate values
            cand_vals = set()
            
            # Small range around old value
            for d in range(-60, 61):
                v = old_val + d
                if 1 <= v <= 10000:
                    cand_vals.add(v)
            
            # For each other element, add smooth - current[j]
            for smooth in smooth_numbers[:200]:
                for j in range(n):
                    if j == idx:
                        continue
                    v = smooth - current[j]
                    if 1 <= v <= 10000:
                        cand_vals.add(v)
            
            # Values making sums divisible by small primes
            for target_prime in [2, 3, 5, 7, 11]:
                for j in range(n):
                    if j == idx:
                        continue
                    rem = (-current[j]) % target_prime
                    for offset in range(0, min(5000, target_prime * 30), target_prime):
                        v = offset + rem
                        if 1 <= v <= 10000:
                            cand_vals.add(v)
                    if len(cand_vals) > 2000:
                        break
                if len(cand_vals) > 2000:
                    break
            
            # Random values
            for _ in range(30):
                cand_vals.add(random.randint(1, 10000))
            
            # Remove duplicates with current
            current_set = set(current)
            cand_vals -= current_set
            
            # Limit
            if len(cand_vals) > 800:
                cand_vals = set(random.sample(list(cand_vals), 800))
            
            # Evaluate
            for new_val in cand_vals:
                if new_val in current:
                    continue
                
                new_primes = set(base_primes)
                
                for j in range(n):
                    if j == idx:
                        continue
                    s = new_val + current[j]
                    pf = _prime_factors_fast(s)
                    new_primes |= pf
                
                new_score = len(new_primes)
                
                if new_score < best_new_score:
                    best_new_score = new_score
                    best_new_val = new_val
                    best_new_primes = new_primes
                    best_new_pf = [row[:] for row in current_pf]
                    for j in range(n):
                        if j != idx:
                            s = new_val + current[j]
                            pf = _prime_factors_fast(s)
                            best_new_pf[idx][j] = pf
                            best_new_pf[j][idx] = pf
            
            if best_new_val is not None and best_new_score < current_score:
                current[idx] = best_new_val
                current_pf = best_new_pf
                current_score = best_new_score
                base_primes_cache = [_base_primes_excluding(current_pf, i) for i in range(n)]
                improved = True
                break
        
        # Random perturbation if stuck
        if not improved and not _deadline_hit():
            for _ in range(3):
                if _deadline_hit():
                    break
                new_cand = current[:]
                num_changes = random.randint(1, 3)
                for _ in range(num_changes):
                    idx2 = random.randint(0, 11)
                    if random.random() < 0.5 and smooth_numbers:
                        smooth = random.choice(smooth_numbers[:150])
                        j2 = random.randint(0, 11)
                        if j2 != idx2:
                            v = smooth - new_cand[j2]
                            if 1 <= v <= 10000 and v not in new_cand:
                                new_cand[idx2] = v
                                continue
                    new_cand[idx2] = random.randint(1, 10000)
                
                if len(set(new_cand)) == 12 and all(1 <= x <= 10000 for x in new_cand):
                    new_pf = _compute_factor_sets(new_cand)
                    new_score = _total_factor_count(new_pf)
                    if new_score < current_score:
                        current = new_cand
                        current_pf = new_pf
                        current_score = new_score
                        base_primes_cache = [_base_primes_excluding(current_pf, i) for i in range(n)]
                        break
    
    # Final verification
    result = current
    if len(set(result)) != 12 or not all(1 <= x <= 10000 for x in result):
        result = [360 * (i+1) for i in range(12)]
    
    return result
