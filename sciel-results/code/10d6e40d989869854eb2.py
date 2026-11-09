"""
Find 12 distinct positive integers <= 10000 minimizing the number of distinct
primes dividing sums of two distinct elements.

Strategy:
- Precompute smallest-prime-factor (SPF) up to 20000 for fast factorization.
- Build a compatibility graph: edge (a,b) exists if the prime factors of (a+b)
  are a subset of a chosen prime set P.
- For several candidate prime sets P (small sets of primes), find a clique of
  size 12 in the induced subgraph.
- Use backtracking with forward checking (or greedy + local search) to find
  such a clique.
- If a clique of size 12 is found for a set P of size k, the answer is <= k.
- Also run a local-search refinement on the best found set.
"""
import time
import random
import sys

_DEADLINE = 85.0
_START_TIME = None

def _time_left():
    if _START_TIME is None:
        return _DEADLINE
    return _DEADLINE - (time.time() - _START_TIME)

def _build_spf(limit):
    """Build smallest-prime-factor array up to limit."""
    spf = list(range(limit + 1))
    for i in range(2, int(limit ** 0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    return spf

def _prime_factors_spf(n, spf):
    """Return set of prime factors of n using SPF table."""
    factors = set()
    while n > 1:
        p = spf[n]
        factors.add(p)
        while n % p == 0:
            n //= p
    return factors

def _score(arr, spf):
    """Compute number of distinct primes dividing all pairwise sums."""
    prime_union = set()
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            s = arr[i] + arr[j]
            while s > 1:
                p = spf[s]
                prime_union.add(p)
                while s % p == 0:
                    s //= p
    return len(prime_union)

def _build_compatibility(prime_set, max_val=10000, spf=None):
    """
    Build adjacency: for each pair (a,b) with a<b, check if all prime factors
    of (a+b) are in prime_set. Return a dict mapping a -> set of compatible b's.
    """
    if spf is None:
        spf = _build_spf(20000)
    
    compat = {}
    # We only consider numbers 1..max_val
    for a in range(1, max_val + 1):
        compat[a] = set()
    
    # This is O(max_val^2) which is 100M for 10000. Too slow.
    # Instead, for a given prime_set, determine which sums are "good"
    # (all prime factors in prime_set), then for each good sum s,
    # all pairs (a, s-a) with 1<=a<s-a<=max_val are compatible.
    
    good_sums = set()
    max_sum = 2 * max_val
    for s in range(2, max_sum + 1):
        pf = _prime_factors_spf(s, spf)
        if pf.issubset(prime_set):
            good_sums.add(s)
    
    for a in range(1, max_val + 1):
        for s in good_sums:
            b = s - a
            if 1 <= b <= max_val and b != a and b > a:
                compat[a].add(b)
                compat[b].add(a)
    
    return compat

def _find_clique_greedy(compat, target_size=12, max_val=10000):
    """
    Greedy + backtracking to find a clique of target_size in the compatibility graph.
    Uses degree-based ordering for speed.
    """
    # Start with nodes that have high degree
    candidates = list(range(1, max_val + 1))
    # Sort by degree descending
    candidates.sort(key=lambda x: len(compat[x]), reverse=True)
    
    # Try to build a clique greedily, then backtrack if needed
    def backtrack(current_clique, remaining_candidates):
        if len(current_clique) == target_size:
            return current_clique
        if not remaining_candidates:
            return None
        if _time_left() < 3.0:
            return None
        
        for i, node in enumerate(remaining_candidates):
            # Check if node is compatible with all in current_clique
            if all(node in compat[c] for c in current_clique):
                new_clique = current_clique + [node]
                new_remaining = [r for r in remaining_candidates[i+1:] 
                                  if all(r in compat[c] for c in new_clique)]
                result = backtrack(new_clique, new_remaining)
                if result is not None:
                    return result
        return None
    
    # To make this tractable, limit the search
    # Use a limited candidate pool
    limited_candidates = candidates[:200]
    result = backtrack([], limited_candidates)
    return result

def _find_clique_limited(prime_set, target_size=12, max_val=10000, spf=None):
    """
    Find a clique of target_size where all pairwise sums have prime factors subset of prime_set.
    Uses a smarter approach: enumerate good sums, then find 12 numbers such that
    all pairwise sums are in the good_sums set.
    """
    if spf is None:
        spf = _build_spf(20000)
    
    max_sum = 2 * max_val
    good_sums = set()
    for s in range(2, max_sum + 1):
        pf = _prime_factors_spf(s, spf)
        if pf.issubset(prime_set):
            good_sums.add(s)
    
    # For each number a, the set of b such that a+b is in good_sums:
    # compat[a] = {b : 1<=b<=max_val, b!=a, a+b in good_sums}
    # We need a clique of size 12.
    
    # Build adjacency lists but only for a restricted set of candidates
    # to keep it manageable.
    
    # Strategy: pick candidate numbers from a small pool, then use backtracking.
    # Pool: numbers that appear in many good sums.
    
    # Count how many good sums each number participates in
    sum_count = {}
    for s in good_sums:
        for a in range(1, max_val + 1):
            b = s - a
            if 1 <= b <= max_val and b != a:
                sum_count[a] = sum_count.get(a, 0) + 1
                sum_count[b] = sum_count.get(b, 0) + 1
    
    # Get top candidates by count
    top_candidates = sorted(sum_count.keys(), key=lambda x: sum_count[x], reverse=True)[:300]
    
    # Build adjacency among top_candidates
    adj = {a: set() for a in top_candidates}
    for i, a in enumerate(top_candidates):
        for b in top_candidates[i+1:]:
            if (a + b) in good_sums:
                adj[a].add(b)
                adj[b].add(a)
    
    # Now find a clique of size target_size in this graph using backtracking
    # Order by degree
    ordered = sorted(top_candidates, key=lambda x: len(adj[x]), reverse=True)
    
    def backtrack(clique, start_idx):
        if len(clique) == target_size:
            return clique
        if _time_left() < 3.0:
            return None
        for idx in range(start_idx, len(ordered)):
            node = ordered[idx]
            if all(node in adj[c] for c in clique):
                result = backtrack(clique + [node], idx + 1)
                if result is not None:
                    return result
        return None
    
    result = backtrack([], 0)
    return result

def _local_search_improve(arr, spf, max_rounds=20):
    """Local search: try replacing one element to reduce score."""
    arr = list(arr)
    best_score = _score(arr, spf)
    n = len(arr)
    max_val = 10000
    
    for rnd in range(max_rounds):
        if _time_left() < 5.0:
            break
        improved = False
        for i in range(n):
            if _time_left() < 5.0:
                break
            old_val = arr[i]
            best_for_i = old_val
            best_score_i = best_score
            # Try random candidates
            for _ in range(30):
                new_val = random.randint(1, max_val)
                if new_val == old_val or new_val in arr:
                    continue
                arr[i] = new_val
                s = _score(arr, spf)
                if s < best_score_i:
                    best_score_i = s
                    best_for_i = new_val
                arr[i] = old_val
            if best_for_i != old_val:
                arr[i] = best_for_i
                best_score = best_score_i
                improved = True
        if not improved:
            break
    
    return arr, best_score

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    random.seed(42)
    
    spf = _build_spf(20000)
    
    best_arr = list(range(1, 13))
    best_score = _score(best_arr, spf)
    
    # Try various prime sets and find cliques
    # The smallest possible answer is 1 (all sums are powers of 2, impossible for 12 distinct)
    # Try prime sets of increasing size
    
    candidate_prime_sets = [
        {2},
        {2, 3},
        {2, 5},
        {2, 7},
        {3, 5},
        {2, 3, 5},
        {2, 3, 7},
        {2, 5, 7},
        {3, 5, 7},
        {2, 3, 11},
        {2, 3, 13},
        {2, 5, 11},
        {2, 3, 5, 7},
        {2, 3, 5, 11},
        {2, 3, 5, 13},
        {2, 3, 7, 11},
        {2, 3, 11, 13},
        {2, 5, 7, 11},
        {3, 5, 7, 11},
    ]
    
    for pset in candidate_prime_sets:
        if _time_left() < 10.0:
            break
        if len(pset) >= best_score:
            continue  # Can't do better than current
        
        result = _find_clique_limited(pset, target_size=12, max_val=10000, spf=spf)
        if result is not None:
            sc = _score(result, spf)
            if sc < best_score:
                best_score = sc
                best_arr = result
                # Try to improve further with local search
                arr2, sc2 = _local_search_improve(result, spf, max_rounds=10)
                if sc2 < best_score:
                    best_score = sc2
                    best_arr = arr2
            # If we found a solution with score == len(pset), that's optimal for this pset
            # But maybe a smaller pset works too; continue checking
    
    # If we haven't found a very good solution, do more local search
    if _time_left() > 15.0:
        best_arr, best_score = _local_search_improve(best_arr, spf, max_rounds=30)
    
    # Additional random restarts with local search
    if _time_left() > 20.0:
        for _ in range(50):
            if _time_left() < 10.0:
                break
            # Generate a random set biased toward small numbers and even numbers
            if random.random() < 0.5:
                candidates = list(range(2, 10001, 2))
            else:
                candidates = list(range(1, 10001))
            arr = random.sample(candidates, 12)
            sc = _score(arr, spf)
            if sc < best_score:
                best_score = sc
                best_arr = arr
            arr2, sc2 = _local_search_improve(arr, spf, max_rounds=5)
            if sc2 < best_score:
                best_score = sc2
                best_arr = arr2
    
    return best_arr
