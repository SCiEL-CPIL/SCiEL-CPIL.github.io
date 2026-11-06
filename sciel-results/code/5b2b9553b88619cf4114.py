"""Improved program for the suite task; uses greedy + local search with deadline guard."""
import math
import time
from itertools import combinations

# Deadline: must finish within 120s wall-clock. We'll aim to return well before 150s.
_START_TIME = time.time()
TIME_LIMIT = 110.0  # safety margin

def _elapsed():
    return time.time() - _START_TIME

def _time_exhausted():
    return _elapsed() > TIME_LIMIT

def construct():
    """
    Find distinct pairwise coprime integers a in 1..511 that maximize sum 1/(512 - a).
    
    Strategy:
    1. Greedy initialization: pick numbers in descending order of 1/(512-a), maintaining coprime constraint.
    2. Local search: try single-swap, add, and 2-swap moves to improve the score.
    3. Use a deadline guard to ensure we return the best result found before time runs out.
    """
    # Precompute scores
    scores = [1.0 / (512 - a) for a in range(512)]  # scores[a] = 1/(512-a)
    
    # Precompute prime factors for each number up to 511
    def _get_prime_factors(n):
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
    
    prime_factors = [set() for _ in range(512)]
    for i in range(2, 512):
        prime_factors[i] = _get_prime_factors(i)
    prime_factors[1] = set()
    
    def _coprime(a, b):
        """Check if a and b are coprime using precomputed factors."""
        if a == 1 or b == 1:
            return True
        return len(prime_factors[a] & prime_factors[b]) == 0
    
    # Greedy initialization: pick numbers with highest scores first, ensuring pairwise coprime
    def _greedy_init():
        chosen = []
        chosen_factors = []  # list of factor sets for quick intersection checks
        candidates = list(range(511, 0, -1))
        for a in candidates:
            if all(_coprime(a, b) for b in chosen):
                chosen.append(a)
                chosen_factors.append(prime_factors[a])
        return chosen
    
    best_set = _greedy_init()
    best_score = sum(scores[a] for a in best_set)
    
    def _is_coprime_with_all(a, chosen):
        """Check if a is coprime with all elements in chosen."""
        for b in chosen:
            if not _coprime(a, b):
                return False
        return True
    
    def _score_of(s):
        return sum(scores[a] for a in s)
    
    # Local search iterations
    iteration = 0
    improved = True
    
    while improved and not _time_exhausted():
        improved = False
        
        # Try single-swap moves: replace one element with another
        for i in range(len(best_set)):
            if _time_exhausted():
                break
            old_elem = best_set[i]
            remaining = [x for j, x in enumerate(best_set) if j != i]
            for candidate in range(511, 0, -1):
                if candidate == old_elem:
                    continue
                if candidate in best_set:
                    continue
                if _is_coprime_with_all(candidate, remaining):
                    delta = scores[candidate] - scores[old_elem]
                    if delta > 1e-15:
                        new_set = remaining + [candidate]
                        best_set = new_set
                        best_score += delta
                        improved = True
                        break
            if _time_exhausted():
                break
            if improved:
                break
        
        # Try adding a new element if possible
        if not _time_exhausted() and not improved:
            for candidate in range(511, 0, -1):
                if candidate in best_set:
                    continue
                if _is_coprime_with_all(candidate, best_set):
                    best_set = best_set + [candidate]
                    best_score += scores[candidate]
                    improved = True
                    break
        
        # Try 2-swap moves: replace two elements with two new elements
        if not _time_exhausted() and not improved:
            n = len(best_set)
            for i in range(n):
                if _time_exhausted():
                    break
                for j in range(i + 1, n):
                    if _time_exhausted():
                        break
                    old1 = best_set[i]
                    old2 = best_set[j]
                    remaining = [x for k, x in enumerate(best_set) if k != i and k != j]
                    # Try to find two new elements c1, c2 that are coprime with remaining and with each other
                    found = False
                    for c1 in range(511, 0, -1):
                        if found or _time_exhausted():
                            break
                        if c1 == old1 or c1 == old2 or c1 in best_set:
                            continue
                        if not _is_coprime_with_all(c1, remaining):
                            continue
                        for c2 in range(511, 0, -1):
                            if _time_exhausted():
                                break
                            if c2 == old1 or c2 == old2 or c2 == c1 or c2 in best_set:
                                continue
                            if not _coprime(c1, c2):
                                continue
                            if not _is_coprime_with_all(c2, remaining):
                                continue
                            delta = scores[c1] + scores[c2] - scores[old1] - scores[old2]
                            if delta > 1e-15:
                                new_set = remaining + [c1, c2]
                                best_set = new_set
                                best_score += delta
                                improved = True
                                found = True
                                break
        
        iteration += 1
        if iteration % 500 == 0 and _time_exhausted():
            break
    
    # Sort the result for consistency (optional)
    return sorted(best_set)

# Keep the original function name and signature
# The function construct() is defined above.
