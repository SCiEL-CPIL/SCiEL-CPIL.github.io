"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

_P0 = time.time()
_DEADLINE = _P0 + 85.0  # safety margin under 90s


def _can_add(prods, x):
    """True iff adding x keeps all subset products distinct."""
    for p in prods:
        if (p * x) in prods:
            return False
    return True


def _try_add(prods, x):
    """Try to add x; returns (success, new_prods)."""
    if not _can_add(prods, x):
        return False, prods
    new_prods = set(prods)
    for p in prods:
        new_prods.add(p * x)
    return True, new_prods


def _verify_distinct(selected):
    """Verify all 2^k subset products are distinct. Returns True if valid."""
    prods = {1}
    for x in selected:
        new_items = [p * x for p in prods]
        for item in new_items:
            if item in prods:
                return False
        prods.update(new_items)
    return True


def _greedy_build(base_set, candidates):
    """Build a set starting with base_set, greedily adding from candidates."""
    selected = list(base_set)
    prods = {1}
    for x in base_set:
        ok, prods = _try_add(prods, x)
        if not ok:
            return None
    for x in candidates:
        ok, prods = _try_add(prods, x)
        if ok:
            selected.append(x)
    return selected


def construct():
    if time.time() > _DEADLINE:
        return [2]

    # All primes in 2..40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    # Non-square-free composites in 2..40
    non_sqfree_composites = []
    for c in range(4, 41):
        if c in primes:
            continue
        is_sqfree = True
        d = 2
        n = c
        while d * d <= n:
            if n % (d * d) == 0:
                is_sqfree = False
                break
            d += 1
        if not is_sqfree:
            non_sqfree_composites.append(c)

    best_selected = []
    best_len = 0

    # Phase 1: Greedy with all primes, ascending composites
    selected = _greedy_build(primes, sorted(non_sqfree_composites))
    if selected and len(selected) > best_len:
        best_len = len(selected)
        best_selected = selected[:]

    if best_len >= 18:
        if _verify_distinct(best_selected[:18]):
            return list(best_selected[:18])

    # Phase 2: Exhaustive combo search with all primes
    for k in [6, 7, 8, 9]:
        if time.time() > _DEADLINE:
            break
        for combo in combinations(non_sqfree_composites, k):
            if time.time() > _DEADLINE:
                break
            selected = list(primes) + list(combo)
            if len(selected) <= best_len:
                continue
            if _verify_distinct(selected):
                best_len = len(selected)
                best_selected = selected[:]
                if best_len >= 18:
                    return list(best_selected[:18])

    # Phase 3: Randomized restarts with dropped primes
    rng = random.Random(12345)
    for trial in range(500):
        if time.time() > _DEADLINE:
            break
        drop_count = rng.choice([0, 1, 2, 3])
        primes_subset = list(primes)
        rng.shuffle(primes_subset)
        primes_subset = primes_subset[:12 - drop_count]

        composites_order = non_sqfree_composites[:]
        rng.shuffle(composites_order)

        selected = _greedy_build(primes_subset, composites_order)
        if selected and len(selected) > best_len:
            if _verify_distinct(selected):
                best_len = len(selected)
                best_selected = selected[:]
                if best_len >= 18:
                    return list(best_selected[:18])

    # Phase 4: Systematic - drop each single prime, try all composite combos
    for i in range(len(primes)):
        if time.time() > _DEADLINE:
            break
        primes_subset = [p for j, p in enumerate(primes) if j != i]
        for k in [6, 7, 8, 9, 10]:
            if time.time() > _DEADLINE:
                break
            for combo in combinations(non_sqfree_composites, k):
                if time.time() > _DEADLINE:
                    break
                selected = list(primes_subset) + list(combo)
                if len(selected) <= best_len:
                    continue
                if _verify_distinct(selected):
                    best_len = len(selected)
                    best_selected = selected[:]
                    if best_len >= 18:
                        return list(best_selected[:18])

    if len(best_selected) >= 2:
        return list(best_selected)

    return [2]
# EVOLVE-BLOCK-END
