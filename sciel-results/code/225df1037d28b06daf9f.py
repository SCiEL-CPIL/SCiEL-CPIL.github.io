"""Model-and-optimize: maximize cardinality of a dissociated reciprocal subset of [1,22].

A set S is dissociated (all reciprocal subset sums distinct) iff adding each new
reciprocal never produces a subset-sum collision. We build greedily over random
orderings of [1..22] using exact integer arithmetic (scaled by LCM), with a
wall-clock deadline guard, and keep the largest valid set found.
"""
import time
import random
from math import gcd

_DEADLINE = 75.0  # seconds; safety margin before the 90 s evaluator kill
_START = time.time()

# LCM of 1..22
_LCM = 232792560
# Precompute integer values for 1/a
_VALS = {a: _LCM // a for a in range(1, 23)}

def _deadline_hit():
    return (time.time() - _START) > _DEADLINE

def _try_greedy(order):
    """Greedily add elements in `order`, keeping a set only if no sum collision.
    Returns the final list (possibly small)."""
    sums = {0}          # current subset sums (integers)
    chosen = []
    for a in order:
        v = _VALS[a]
        # new sums are {s + v for s in sums}; must be disjoint from `sums`
        new_sums = [s + v for s in sums]
        collides = False
        for ns in new_sums:
            if ns in sums:
                collides = True
                break
        if collides:
            continue
        # commit
        sums.update(new_sums)
        chosen.append(a)
    return chosen

def _try_greedy_with_local_search(order, best_k):
    """Greedily add elements in `order`, then try local search to improve.
    Returns the final list (possibly small)."""
    sums = {0}          # current subset sums (integers)
    chosen = []
    chosen_set = set()
    
    # First pass: greedy
    for a in order:
        if a in chosen_set:
            continue
        v = _VALS[a]
        new_sums = [s + v for s in sums]
        collides = False
        for ns in new_sums:
            if ns in sums:
                collides = True
                break
        if collides:
            continue
        sums.update(new_sums)
        chosen.append(a)
        chosen_set.add(a)
    
    if len(chosen) <= best_k:
        return chosen
    
    # Local search: try to swap out elements to make room for better ones
    full_set = set(range(1, 23))
    current_set = set(chosen)
    
    for _ in range(50):  # increased iterations due to speedup
        if _deadline_hit():
            break
        
        # Try each possible swap
        found_improvement = False
        for remove_a in list(current_set):
            if len(current_set) - 1 + 1 <= best_k:
                continue
                
            # Compute sums without remove_a
            # This is expensive, but we only do it if we actually try to remove
            # Optimization: We can maintain the set of sums incrementally?
            # For now, recompute. It's O(2^k). k is ~16, so 65536 ops.
            # 50 iterations * 22 removes * 65536 ops = ~70M ops. Might be too slow in Python?
            # Let's check timing. 1M ops ~ 0.05s. 70M ops ~ 3.5s.
            # We have 75s budget. So we can afford a few of these.
            
            sums_without = {0}
            for b in current_set:
                if b == remove_a:
                    continue
                v_b = _VALS[b]
                new_sums = [s + v_b for s in sums_without]
                # Check if all new sums are distinct from existing
                valid = True
                for ns in new_sums:
                    if ns in sums_without:
                        valid = False
                        break
                if valid:
                    sums_without.update(new_sums)
            
            if len(sums_without) != 2 ** (len(current_set) - 1):
                continue  # Invalid state (should not happen if we started valid)
            
            # Try to add an element not in current_set
            for add_a in full_set - current_set:
                v_add = _VALS[add_a]
                new_sums = [s + v_add for s in sums_without]
                collides = False
                for ns in new_sums:
                    if ns in sums_without:
                        collides = True
                        break
                if not collides:
                    # Found a valid swap
                    current_set.remove(remove_a)
                    current_set.add(add_a)
                    found_improvement = True
                    break
            if found_improvement:
                break
        
        if not found_improvement:
            break
    
    return list(current_set)

def construct():
    global _START
    _START = time.time()
    random.seed()
    full = list(range(1, 23))

    # Deterministic seed baseline (known valid, 9 elements).
    seed = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    best = seed
    best_k = len(seed)

    # Random restarts: greedy over random permutations with local search.
    while not _deadline_hit():
        order = full[:]
        random.shuffle(order)
        cand = _try_greedy_with_local_search(order, best_k)
        if len(cand) > best_k:
            best_k = len(cand)
            best = cand
            # If we reached a very large set, stop early to save time for final polish?
            # Or keep going?
            if best_k >= 18:
                break

    if best is None or len(best) == 0:
        best = [1]
    return list(best)
