"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def _factorize(n):
    """Return the set of prime factors of n."""
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

def _is_coprime_to_all_fast(a, chosen_factors):
    """Check if a is coprime to all elements whose prime factors are in chosen_factors."""
    af = _factorize(a)
    for cf in chosen_factors:
        if af & cf:
            return False
    return True

def _score(chosen):
    """Compute the score: sum of 1/(512 - a) for all chosen a."""
    return sum(1.0 / (512 - a) for a in chosen)

def _greedy_from_high(deadline=None):
    """Greedy: pick from high to low, adding if coprime to all previously chosen."""
    chosen = []
    chosen_factors = []
    for a in range(511, 0, -1):
        if deadline is not None and time.time() > deadline:
            break
        if _is_coprime_to_all_fast(a, chosen_factors):
            chosen.append(a)
            chosen_factors.append(_factorize(a))
    return chosen

def _local_search(chosen, deadline):
    """Local search: try swapping one element with another to improve score."""
    best = list(chosen)
    best_score = _score(best)
    best_factors = [_factorize(a) for a in best]
    
    # Build a set of chosen values for O(1) lookup
    chosen_set = set(best)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try removing each element
        for i in range(len(best)):
            if time.time() > deadline:
                break
            removed = best[i]
            removed_factors = best_factors[i]
            
            # Try adding each candidate not in best
            for a in range(511, 0, -1):
                if time.time() > deadline:
                    break
                if a in chosen_set:
                    continue
                # Check if a is coprime to all remaining elements
                if _is_coprime_to_all_fast(a, best_factors[:i] + best_factors[i+1:]):
                    new_score = best_score - 1.0/(512 - removed) + 1.0/(512 - a)
                    if new_score > best_score + 1e-12:
                        best[i] = a
                        best_factors[i] = _factorize(a)
                        chosen_set.remove(removed)
                        chosen_set.add(a)
                        best_score = new_score
                        improved = True
                        break
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Phase 1: Greedy initialization
    chosen = _greedy_from_high(deadline)
    if not chosen:
        return [511]
    
    # Phase 2: Local search
    if time.time() < deadline:
        chosen, best_score = _local_search(chosen, deadline)
    
    return chosen
# EVOLVE-BLOCK-END
