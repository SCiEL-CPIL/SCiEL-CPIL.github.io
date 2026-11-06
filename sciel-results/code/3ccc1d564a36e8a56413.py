"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from itertools import combinations

_start_time = time.time()
_DEADLINE = 60.0  # seconds

def _check_dissociated(subset):
    """Check if all 2^k reciprocal subset sums are distinct using exact Fraction arithmetic."""
    k = len(subset)
    if k == 0:
        return True
    # Generate all 2^k subset sums
    sums = [Fraction(0)]
    for x in subset:
        recip = Fraction(1, x)
        # Add recip to all existing sums
        new_sums = [s + recip for s in sums]
        sums.extend(new_sums)
    # Check all distinct
    return len(set(sums)) == len(sums)

def _try_add(subset, candidate):
    """Try adding candidate to subset and check if still dissociated."""
    new_subset = list(subset) + [candidate]
    return _check_dissociated(new_subset)

def construct():
    global _start_time
    best = [1, 2, 3, 5, 7, 11, 13, 17, 19]  # known size-9 dissociated set
    
    # Verify the seed is valid
    if not _check_dissociated(best):
        best = [1, 2, 3, 5, 7, 11, 13, 17, 19]  # fallback
    
    all_candidates = list(range(1, 23))
    best_size = len(best)
    
    # Phase 1: Greedy extension - try to add elements one by one
    while True:
        added = False
        for c in all_candidates:
            if c not in best:
                if time.time() - _start_time > _DEADLINE - 5:
                    return best
                if _try_add(best, c):
                    best.append(c)
                    best_size = len(best)
                    added = True
                    break
        if not added:
            break
    
    # Phase 2: Local search - try swaps to find larger sets
    # For each element in best, try swapping with an element not in best
    improved = True
    while improved:
        improved = False
        if time.time() - _start_time > _DEADLINE - 5:
            return best
        for i, elem in enumerate(best):
            for c in all_candidates:
                if c not in best:
                    if time.time() - _start_time > _DEADLINE - 5:
                        return best
                    # Try swapping elem with c
                    new_subset = best[:i] + [c] + best[i+1:]
                    if _check_dissociated(new_subset):
                        best = new_subset
                        improved = True
                        break
            if improved:
                break
    
    # Phase 3: Try greedy construction from scratch with different orderings
    # Sort candidates by some heuristic and greedily build
    for order in [
        sorted(all_candidates),
        sorted(all_candidates, reverse=True),
        sorted(all_candidates, key=lambda x: 1/x),
        sorted(all_candidates, key=lambda x: -1/x),
    ]:
        if time.time() - _start_time > _DEADLINE - 5:
            return best
        current = []
        for c in order:
            if time.time() - _start_time > _DEADLINE - 5:
                break
            if _try_add(current, c):
                current.append(c)
        if len(current) > best_size:
            best = current
            best_size = len(current)
    
    # Phase 4: Randomized greedy with restarts
    import random
    rng = random.Random(42)
    for trial in range(200):
        if time.time() - _start_time > _DEADLINE - 5:
            break
        shuffled = all_candidates[:]
        rng.shuffle(shuffled)
        current = []
        for c in shuffled:
            if time.time() - _start_time > _DEADLINE - 5:
                break
            if _try_add(current, c):
                current.append(c)
        if len(current) > best_size:
            best = current
            best_size = len(current)
    
    # Phase 5: Local search on best - try adding any remaining element
    for c in all_candidates:
        if time.time() - _start_time > _DEADLINE - 5:
            break
        if c not in best:
            if _try_add(best, c):
                best.append(c)
                best_size = len(best)
    
    # Final safety: ensure we return a valid nonempty subset
    if not best:
        best = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    
    return best
# EVOLVE-BLOCK-END
