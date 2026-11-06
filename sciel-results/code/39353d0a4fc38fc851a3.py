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
    for i in range(2, _MAX_SUM + 1):
        if _pf_cache[i] is None:
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

def _count_distinct_primes_fast(nums):
    """Count distinct prime factors across all pairwise sums using cache."""
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

def _get_primes_set(nums):
    """Return the set of distinct primes in pairwise sums."""
    all_primes = set()
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            s = nums[i] + nums[j]
            if s <= _MAX_SUM:
                all_primes |= _pf_cache[s]
            else:
                all_primes |= _prime_factors(s)
    return all_primes

def _search_smooth_k(allowed_primes, max_k, n_needed, deadline):
    """
    Backtracking search for n_needed distinct integers k_i <= max_k such that
    for all i != j, prime_factors(k_i + k_j) is a subset of allowed_primes.
    """
    k_values = []
    
    def backtrack(start_idx, current_k_list):
        if time.time() > deadline:
            return None
        if len(current_k_list) == n_needed:
            return list(current_k_list)
        
        for k in range(start_idx, max_k + 1):
            valid = True
            for existing_k in current_k_list:
                s = k + existing_k
                pf = _pf_cache[s] if s <= _MAX_SUM else _prime_factors(s)
                if not pf.issubset(allowed_primes):
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

def _construct_smooth(primes_target, deadline):
    """
    Try to construct numbers x_i = k_i * P where P = product of primes_target.
    Then x_i + x_j = P * (k_i + k_j), so prime_factors(x_i+x_j) = primes_target ∪ prime_factors(k_i+k_j).
    If prime_factors(k_i+k_j) ⊆ primes_target, total distinct primes = len(primes_target).
    """
    n = 12
    max_val = 10000
    
    P = 1
    for p in primes_target:
        P *= p
        
    max_k = max_val // P
    if max_k < n:
        return None, float('inf')
        
    k_list = _search_smooth_k(set(primes_target), max_k, n, deadline - 5)
    if k_list is None:
        return None, float('inf')
        
    nums = [k * P for k in k_list]
    if len(set(nums)) != n or any(x > max_val or x < 1 for x in nums):
        return None, float('inf')
        
    score = _count_distinct_primes_fast(nums)
    return nums, score

def _spectral_greedy_select(candidates, target_size, deadline):
    """
    Given a list of candidate numbers, greedily select target_size numbers
    to minimize the union of prime factors in pairwise sums.
    Uses a scoring heuristic: at each step, pick the number that adds the
    fewest new primes to the current prime set.
    """
    if len(candidates) < target_size:
        return None, float('inf')
    
    if time.time() > deadline:
        return None, float('inf')
    
    # Compute prime factors for each candidate's pairwise interactions
    # For efficiency, we'll use a greedy approach with incremental prime tracking
    
    selected = []
    prime_set = set()
    remaining = list(candidates)
    
    for _ in range(target_size):
        best_candidate = None
        best_new_primes = float('inf')
        best_primes_after = None
        
        for cand in remaining:
            # Compute what primes this candidate would add
            added_primes = set()
            for sel in selected:
                s = cand + sel
                pf = _pf_cache[s] if s <= _MAX_SUM else _prime_factors(s)
                added_primes |= (pf - prime_set)
            
            new_count = len(added_primes)
            if new_count < best_new_primes or (new_count == best_new_primes and (best_candidate is None or cand < best_candidate)):
                best_new_primes = new_count
                best_candidate = cand
                best_primes_after = added_primes
        
        if best_candidate is None:
            break
        
        selected.append(best_candidate)
        prime_set |= best_primes_after
        remaining.remove(best_candidate)
        
        if time.time() > deadline:
            break
    
    if len(selected) < target_size:
        return None, float('inf')
    
    score = len(prime_set)
    return selected, score

def _cluster_based_search(deadline):
    """
    Build a compatibility structure: for each candidate, compute its "cost profile"
    (which primes it introduces). Then find dense clusters of compatible numbers
    and greedily select 12 from the best cluster.
    
    Strategy: 
    1. Generate a pool of candidates (multiples of small products of primes)
    2. For each candidate, compute the set of primes it would contribute
    3. Group candidates by their prime-contribution patterns
    4. Try to build a 12-set from compatible groups
    """
    n = 12
    max_val = 10000
    best_result = None
    best_score = float('inf')
    
    # Generate candidate pools based on smooth number structure
    # Numbers that are multiples of P (product of small primes) have sums
    # that are also multiples of P, constraining their prime factors.
    
    prime_combos = [
        [2], [3], [5], [7],
        [2,3], [2,5], [3,5], [2,7],
        [2,3,5], [2,3,7], [3,5,7], [2,5,7],
        [2,3,5,7],
    ]
    
    for primes in prime_combos:
        if time.time() > deadline - 3:
            break
        
        P = 1
        for p in primes:
            P *= p
        
        if P > max_val:
            continue
        
        max_k = max_val // P
        if max_k < n:
            continue
        
        # Generate candidates: k * P for k in some range
        # Focus on k values that produce smooth sums
        candidates = []
        for k in range(1, min(max_k + 1, 200)):
            val = k * P
            if val <= max_val:
                candidates.append(val)
        
        if len(candidates) < n:
            continue
        
        # Try greedy selection from this candidate pool
        result, score = _spectral_greedy_select(candidates, n, deadline - 2)
        if result is not None and score < best_score:
            best_score = score
            best_result = result
    
    # Also try a broader pool with mixed candidates
    if time.time() < deadline - 5:
        # Generate a diverse candidate pool
        diverse_candidates = set()
        for P in [2, 3, 4, 5, 6, 7, 10, 12, 14, 15, 20, 21, 30, 42, 60, 70, 105, 210]:
            max_k = max_val // P
            for k in range(1, min(max_k + 1, 100)):
                diverse_candidates.add(k * P)
        
        diverse_list = sorted(diverse_candidates)
        if len(diverse_list) >= n:
            # Take a subset for efficiency
            sample_size = min(len(diverse_list), 300)
            sampled = random.sample(diverse_list, sample_size) if len(diverse_list) > sample_size else diverse_list
            result, score = _spectral_greedy_select(sampled, n, deadline - 2)
            if result is not None and score < best_score:
                best_score = score
                best_result = result
    
    return best_result, best_score

def _local_search(nums, deadline):
    """Local search: try replacing elements to reduce distinct prime count."""
    best = list(nums)
    best_score = _count_distinct_primes_fast(best)
    best_primes = _get_primes_set(best)
    
    n = len(best)
    max_val = 10000
    improved = True
    
    while improved and time.time() < deadline - 2:
        improved = False
        for i in range(n):
            old_val = best[i]
            # Compute primes contributed by old_val
            old_primes = set()
            for j in range(n):
                if j != i:
                    s = old_val + best[j]
                    pf = _pf_cache[s] if s <= _MAX_SUM else _prime_factors(s)
                    old_primes |= pf
            
            # Try candidate replacements
            candidates = set()
            # Nearby values
            for d in range(1, 100):
                for v in [old_val + d, old_val - d]:
                    if 1 <= v <= max_val and v not in best:
                        candidates.add(v)
            # Values that are multiples of small numbers (more likely smooth)
            for mult in [2, 3, 4, 5, 6, 7, 10, 12, 15, 20, 30]:
                base = old_val // mult * mult
                for offset in range(-3, 4):
                    v = base + offset * mult
                    if 1 <= v <= max_val and v not in best:
                        candidates.add(v)
            # Random
            for _ in range(20):
                v = random.randint(1, max_val)
                if v not in best:
                    candidates.add(v)
            
            for v in sorted(candidates):
                # Compute new primes if we replace best[i] with v
                new_primes = set()
                for j in range(n):
                    if j != i:
                        s = v + best[j]
                        pf = _pf_cache[s] if s <= _MAX_SUM else _prime_factors(s)
                        new_primes |= pf
                
                # Primes that are only in old_primes and not in new_primes can be removed
                # Primes in new_primes but not in (old_primes ∪ (best_primes - old_primes)) are added
                # Simplified: compute total prime set with replacement
                # Primes from pairs not involving i stay the same
                other_primes = set()
                for j in range(n):
                    for k in range(j+1, n):
                        if j != i and k != i:
                            s = best[j] + best[k]
                            pf = _pf_cache[s] if s <= _MAX_SUM else _prime_factors(s)
                            other_primes |= pf
                
                total_primes = other_primes | new_primes
                total_count = len(total_primes)
                
                if total_count < best_score:
                    best[i] = v
                    best_score = total_count
                    best_primes = total_primes
                    improved = True
                    break
                elif total_count == best_score:
                    if random.random() < 0.03:
                        best[i] = v
                        best_primes = total_primes
                        improved = True
                        break
    
    return best, best_score

def _ap_search(deadline):
    """Search arithmetic progressions for good starting points."""
    n = 12
    max_val = 10000
    best = None
    best_score = float('inf')
    
    for d in [2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 15, 18, 20, 24, 30, 42, 60]:
        if time.time() > deadline - 5:
            break
        max_a = max_val - (n - 1) * d
        if max_a < 1:
            continue
        step = max(1, max_a // 80)
        for a in range(1, max_a + 1, step):
            nums = [a + i * d for i in range(n)]
            score = _count_distinct_primes_fast(nums)
            if score < best_score:
                best_score = score
                best = list(nums)
            if score <= 2:
                break
        if best_score <= 2:
            break
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 82  # Safety margin for 90s limit
    
    # Build prime factor cache
    _build_pf_cache()
    
    n = 12
    max_val = 10000
    
    best_result = list(range(1, 13))
    best_score = _count_distinct_primes_fast(best_result)
    
    random.seed(42)
    
    # Strategy 1: Smooth number construction with specific prime targets
    prime_sets_to_try = [
        [2, 3],
        [2, 3, 5],
        [2, 5],
        [3, 5],
        [2, 7],
        [2, 3, 7],
        [2, 5, 7],
        [2, 3, 5, 7],
        [2, 3, 5, 11],
        [3, 5, 7],
        [5, 7, 11],
    ]
    
    for primes in prime_sets_to_try:
        if time.time() > deadline - 15:
            break
        result, score = _construct_smooth(primes, deadline - 10)
        if result is not None and score < best_score:
            best_score = score
            best_result = result
        if best_score <= 2:
            break
    
    # Strategy 2: Cluster/spectral-based greedy selection
    if time.time() < deadline - 10:
        result, score = _cluster_based_search(deadline - 8)
        if result is not None and score < best_score:
            best_score = score
            best_result = result
    
    # Strategy 3: Arithmetic progression search
    if time.time() < deadline - 10:
        result, score = _ap_search(deadline - 8)
        if result is not None and score < best_score:
            best_score = score
            best_result = result
    
    # Strategy 4: Local search from multiple starting points
    start_points = [
        list(range(2, 2*n+1, 2)),  # Even numbers
        [i*6 for i in range(1, n+1)],  # Multiples of 6
        [i*10 for i in range(1, n+1)],  # Multiples of 10
        [i*30 for i in range(1, n+1)],  # Multiples of 30
        [i*42 for i in range(1, n+1)],  # Multiples of 42
    ]
    
    for sp in start_points:
        if time.time() > deadline - 5:
            break
        if all(x <= max_val and x >= 1 for x in sp) and len(set(sp)) == n:
            result, score = _local_search(sp, deadline - 4)
            if score < best_score:
                best_score = score
                best_result = result
    
    # Random restarts
    restarts = 0
    while time.time() < deadline - 8 and restarts < 40:
        restarts += 1
        # Generate random starting set biased toward smooth numbers
        nums = []
        while len(nums) < n:
            v = random.randint(1, max_val)
            if v not in nums:
                nums.append(v)
        result, score = _local_search(nums, deadline - 5)
        if score < best_score:
            best_score = score
            best_result = result
    
    # Final polish
    if time.time() < deadline - 2:
        result, score = _local_search(best_result, deadline - 1)
        if score < best_score:
            best_score = score
            best_result = result
    
    return best_result
