"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from fractions import Fraction

# Precompute prime factors for all numbers 1..511
def _factorize(n):
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

_PRIME_FACTORS = {}
for _i in range(1, 512):
    _PRIME_FACTORS[_i] = _factorize(_i)

# Precompute exact weights as Fractions
_WEIGHTS = {a: Fraction(1, 512 - a) for a in range(1, 512)}

def _is_coprime_to_all_fast(a, chosen_set):
    """Check if a is coprime to all elements in chosen_set using precomputed factors."""
    fa = _PRIME_FACTORS[a]
    if not fa:
        return True
    for b in chosen_set:
        if fa & _PRIME_FACTORS[b]:
            return False
    return True

def _score(chosen):
    return sum(1.0 / (512 - a) for a in chosen)

def _score_exact(chosen):
    return sum((_WEIGHTS[a] for a in chosen), Fraction(0, 1))

def _greedy_from_high():
    chosen = []
    chosen_set = set()
    for a in range(511, 0, -1):
        if _is_coprime_to_all_fast(a, chosen_set):
            chosen.append(a)
            chosen_set.add(a)
    return chosen

def _greedy_fill(remaining_set):
    """Greedily add all coprime candidates in descending order of weight."""
    new_list = []
    for a in range(511, 0, -1):
        if a in remaining_set:
            continue
        if _is_coprime_to_all_fast(a, remaining_set):
            new_list.append(a)
            remaining_set.add(a)
    return new_list

def _local_search(chosen, deadline):
    """Improve using 1-for-k and 2-for-k moves with exact rational gain comparison."""
    best = list(chosen)
    best_set = set(best)
    best_score_exact = _score_exact(best)
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
        n = len(best)
        
        # Try 1-for-k: remove one element, greedily fill
        for i in range(n):
            if time.time() > deadline:
                break
            remaining_set = set(best)
            removed = best[i]
            remaining_set.discard(removed)
            new_list = [x for x in best if x != removed]
            additions = _greedy_fill(remaining_set)
            new_list.extend(additions)
            new_score_exact = _score_exact(new_list)
            if new_score_exact > best_score_exact:
                best = new_list
                best_set = set(best)
                best_score_exact = new_score_exact
                improved = True
                break
        
        if improved:
            continue
        
        # Try 2-for-k: remove two elements, greedily fill
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                if time.time() > deadline:
                    break
                remaining_set = set(best)
                remaining_set.discard(best[i])
                remaining_set.discard(best[j])
                new_list = [x for x in best if x != best[i] and x != best[j]]
                additions = _greedy_fill(remaining_set)
                new_list.extend(additions)
                new_score_exact = _score_exact(new_list)
                if new_score_exact > best_score_exact:
                    best = new_list
                    best_set = set(best)
                    best_score_exact = new_score_exact
                    improved = True
                    break
            if improved:
                break
    return best

def construct():
    start = time.time()
    deadline = start + 145.0  # Safety margin before 150s kill

    # Initial greedy from high
    chosen = _greedy_from_high()

    # Local search with 1-for-k and 2-for-k moves
    chosen = _local_search(chosen, deadline)

    # Final local search polish
    if time.time() < deadline:
        chosen = _local_search(chosen, deadline)

    return chosen
# EVOLVE-BLOCK-END
