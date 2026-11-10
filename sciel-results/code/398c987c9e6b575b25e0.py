"""Rewritten program using a direct, constructive approach to minimize distinct prime factors
across pairwise sums of 12 distinct positive integers <= 10000."""
import time
import math
from itertools import combinations

# Global deadline guard
_START_TIME = time.time()
_DEADLINE_SECONDS = 85.0  # leave 5s margin before 90s kill

def _elapsed():
    return time.time() - _START_TIME

def _deadline_hit():
    return _elapsed() > _DEADLINE_SECONDS

def _prime_factors(n):
    """Return set of prime factors of n."""
    factors = set()
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.add(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors

def _all_prime_factors_of_sums(nums):
    """Return set of all distinct prime factors of sums of distinct-index pairs."""
    factors = set()
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            s = nums[i] + nums[j]
            factors |= _prime_factors(s)
    return factors

def _score(nums):
    """Lower is better: number of distinct prime factors."""
    return len(_all_prime_factors_of_sums(nums))

def _is_smooth(n, allowed_primes):
    """Check if n is composed only of primes in allowed_primes."""
    for p in allowed_primes:
        while n % p == 0:
            n //= p
    return n == 1

def _find_clique_in_compatibility_graph(candidate_nums, allowed_primes, target_size=12):
    """
    Given a list of candidate numbers and a set of allowed primes,
    find a subset of size target_size where every pair's sum is smooth
    (only has prime factors from allowed_primes).
    Uses greedy approach with backtracking.
    """
    n = len(candidate_nums)
    if n < target_size:
        return None
    
    # Build compatibility: compatible[i][j] = True if candidate_nums[i] + candidate_nums[j] is smooth
    compatible = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            s = candidate_nums[i] + candidate_nums[j]
            if _is_smooth(s, allowed_primes):
                compatible[i][j] = True
                compatible[j][i] = True
    
    # Greedy: try to build a clique of size target_size
    # Sort candidates by degree (number of compatible neighbors)
    degrees = [sum(compatible[i]) for i in range(n)]
    order = sorted(range(n), key=lambda i: -degrees[i])
    
    # Try greedy selection with backtracking
    def greedy_clique(remaining, selected, max_size):
        if len(selected) == max_size:
            return selected[:]
        if not remaining:
            return None
        if _deadline_hit():
            return None
        
        # Try each candidate in remaining
        for idx in range(len(remaining)):
            node = remaining[idx]
            # Check if node is compatible with all selected
            if all(node in selected or compatible[node][s] for s in selected):
                new_remaining = [r for r in remaining if r != node and (not selected or all(compatible[node][s] for s in selected))]
                result = greedy_clique(new_remaining, selected + [node], max_size)
                if result is not None:
                    return result
        return None
    
    result = greedy_clique(order, [], target_size)
    return result

def construct():
    """
    Strategy: Try different small prime sets {p1, p2, ...} and find 12 numbers
    such that all pairwise sums have only those prime factors.
    """
    import random
    random.seed(42)
    
    best = list(range(2, 25, 2))  # fallback: 2,4,...,24
    best_score = _score(best)
    
    # Try various small prime sets
    prime_sets_to_try = [
        {2},
        {2, 3},
        {2, 5},
        {3, 5},
        {2, 3, 5},
        {2, 3, 7},
        {2, 5, 7},
        {3, 5, 7},
        {2, 3, 5, 7},
        {2, 3, 11},
        {2, 3, 13},
        {2, 5, 11},
        {2, 3, 5, 11},
        {2, 3, 5, 7, 11},
    ]
    
    for allowed_primes in prime_sets_to_try:
        if _deadline_hit():
            break
        
        # Generate candidate numbers: numbers <= 10000 whose value is smooth
        # (composed only of allowed_primes). These are more likely to have
        # smooth pairwise sums.
        # Actually, we need numbers such that a+b is smooth for all pairs.
        # Let's generate candidates more strategically.
        
        # Strategy: pick numbers that are multiples of the product of some subset
        # of allowed_primes, so sums are also multiples.
        
        # Simpler: just try a large pool of random numbers and check compatibility.
        # Generate candidates: all numbers 1..500 that are smooth w.r.t. allowed_primes
        candidates = []
        for x in range(1, 501):
            if _is_smooth(x, allowed_primes):
                candidates.append(x)
        
        # Also add some larger smooth numbers
        for x in range(501, 10001, 100):
            if _is_smooth(x, allowed_primes):
                candidates.append(x)
        
        # Remove duplicates
        candidates = list(set(candidates))
        
        if len(candidates) < 12:
            continue
        
        # Shuffle for diversity
        random.shuffle(candidates)
        
        # Try to find a clique of size 12
        clique = _find_clique_in_compatibility_graph(candidates[:200], allowed_primes, 12)
        if clique is not None:
            nums = [candidates[i] for i in clique]
            s = _score(nums)
            if s < best_score:
                best_score = s
                best = nums[:]
                if best_score <= len(allowed_primes):
                    # Can't do better than the size of the prime set
                    break
        
        if _deadline_hit():
            break
    
    # If we haven't found something great, try a different approach:
    # Use numbers that are all congruent mod small primes
    if best_score > 5 and not _deadline_hit():
        # Try: all numbers ≡ r (mod 2) and ≡ s (mod 3) etc.
        # If all numbers ≡ 0 (mod 6), then all sums ≡ 0 (mod 6), so 2 and 3 are factors.
        # Try to find 12 multiples of 6 such that sums/6 have few prime factors.
        
        multiples_of_6 = [6 * k for k in range(1, 1667)]  # up to 10000
        random.shuffle(multiples_of_6)
        
        # Try to find 12 numbers from this pool where pairwise sums have few prime factors
        # Use a greedy approach: start with 6, then add numbers whose sum with existing ones
        # introduces few new primes.
        
        current = [6]
        current_primes = set()
        
        for candidate in multiples_of_6[1:]:
            if _deadline_hit():
                break
            if len(current) >= 12:
                break
            
            # Check if adding this candidate introduces few new primes
            new_primes = set()
            for existing in current:
                s = candidate + existing
                new_primes |= _prime_factors(s)
            
            added = new_primes - current_primes
            if len(added) <= 1:  # at most 1 new prime
                current.append(candidate)
                current_primes |= new_primes
        
        if len(current) >= 12:
            current = current[:12]
            s = _score(current)
            if s < best_score:
                best_score = s
                best = current[:]
    
    # Final local search refinement
    if not _deadline_hit() and best_score > 4:
        for iteration in range(50):
            if _deadline_hit():
                break
            
            improved = False
            for i in range(12):
                if _deadline_hit():
                    break
                old_val = best[i]
                # Try nearby values
                for delta in range(-5, 6):
                    if delta == 0:
                        continue
                    new_val = old_val + delta
                    if new_val < 1 or new_val > 10000:
                        continue
                    if new_val in best[:i] or new_val in best[i+1:]:
                        continue
                    best[i] = new_val
                    s = _score(best)
                    if s < best_score:
                        best_score = s
                        improved = True
                        break
                    else:
                        best[i] = old_val
                if improved:
                    break
            
            if not improved:
                break
    
    # Ensure validity
    result = best
    assert len(set(result)) == 12
    assert all(1 <= x <= 10000 for x in result)
    
    return result

# EVOLVE-BLOCK-END
