"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import lru_cache

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

def _is_coprime_to_all_fast(a, chosen_set):
    """Check if a is coprime to all elements in chosen_set using precomputed factors."""
    fa = _PRIME_FACTORS[a]
    for b in chosen_set:
        if fa & _PRIME_FACTORS[b]:
            return False
    return True

def _score(chosen):
    return sum(1.0 / (512 - a) for a in chosen)

def _greedy_from_high():
    chosen = []
    chosen_set = set()
    for a in range(511, 0, -1):
        if _is_coprime_to_all_fast(a, chosen_set):
            chosen.append(a)
            chosen_set.add(a)
    return chosen

def _greedy_by_weight(order):
    """Greedy selection given an ordering of candidates."""
    chosen = []
    chosen_set = set()
    for a in order:
        if _is_coprime_to_all_fast(a, chosen_set):
            chosen.append(a)
            chosen_set.add(a)
    return chosen

def _local_search(chosen, deadline):
    """Improve by trying removals and swaps with re-greedy fill."""
    best = list(chosen)
    best_set = set(best)
    best_score = _score(best)
    n = len(best)
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
        # Try removing each element and re-greeding
        for i in range(n):
            if time.time() > deadline:
                break
            remaining = [x for j, x in enumerate(best) if j != i]
            remaining_set = set(remaining)
            new_list = list(remaining)
            for a in range(511, 0, -1):
                if a in remaining_set:
                    continue
                if _is_coprime_to_all_fast(a, remaining_set):
                    new_list.append(a)
                    remaining_set.add(a)
            new_score = _score(new_list)
            if new_score > best_score + 1e-12:
                best = new_list
                best_set = set(best)
                best_score = new_score
                n = len(best)
                improved = True
                break  # restart outer while
        # Try swaps: remove one, add a different one, then re-greedy
        if not improved and time.time() < deadline:
            for i in range(n):
                if time.time() > deadline:
                    break
                removed = best[i]
                remaining = [x for j, x in enumerate(best) if j != i]
                remaining_set = set(remaining)
                # Try adding each candidate not already in set
                best_swap = None
                best_swap_score = best_score
                for a in range(511, 0, -1):
                    if a in remaining_set or a == removed:
                        continue
                    if not _is_coprime_to_all_fast(a, remaining_set):
                        continue
                    # Add a, then re-greedy fill
                    new_list = list(remaining) + [a]
                    new_set = remaining_set | {a}
                    for b in range(511, 0, -1):
                        if b in new_set:
                            continue
                        if _is_coprime_to_all_fast(b, new_set):
                            new_list.append(b)
                            new_set.add(b)
                    sc = _score(new_list)
                    if sc > best_swap_score + 1e-12:
                        best_swap_score = sc
                        best_swap = new_list
                if best_swap is not None:
                    best = best_swap
                    best_set = set(best)
                    best_score = best_swap_score
                    n = len(best)
                    improved = True
                    break
    return best

def construct():
    start = time.time()
    deadline = start + 140.0  # Safety margin before 150s kill
    
    # Initial greedy from high
    chosen = _greedy_from_high()
    
    # Local search
    chosen = _local_search(chosen, deadline)
    
    # Try different greedy orderings as additional restarts
    if time.time() < deadline:
        # Weighted ordering: sort by 1/(512-a) descending (same as high-to-low)
        # Try a few perturbed orderings
        import random
        rng = random.Random(42)
        candidates = list(range(511, 0, -1))
        
        for restart in range(10):
            if time.time() > deadline:
                break
            # Perturb the ordering slightly
            order = list(range(511, 0, -1))
            # Swap a few random pairs to create variation
            for _ in range(5):
                i, j = rng.randint(0, 510), rng.randint(0, 510)
                order[i], order[j] = order[j], order[i]
            
            trial = _greedy_by_weight(order)
            trial = _local_search(trial, deadline)
            if _score(trial) > _score(chosen):
                chosen = trial
    
    return chosen
# EVOLVE-BLOCK-END
