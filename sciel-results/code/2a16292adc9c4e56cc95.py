"""Program for large sets with distinct subset products."""
# EVOLVE-BLOCK-START
import time
import random
from math import isqrt

_DEADLINE = 80.0  # safety margin before 90s kill

def _is_prime(a):
    if a < 2:
        return False
    if a % 2 == 0:
        return a == 2
    d = 3
    while d * d <= a:
        if a % d == 0:
            return False
        d += 2
    return True

def _all_subset_products_distinct(nums):
    """Check that all 2^k subset products are distinct. nums is a list of ints."""
    prods = {1}
    for x in nums:
        new = set()
        for p in prods:
            q = p * x
            if q in prods or q in new:
                return False
            new.add(q)
        prods |= new
    return True

def _can_add(current, x):
    """Check whether adding x to current preserves distinct subset products."""
    # current is a list; build subset products of current, then test adding x
    prods = {1}
    for x0 in current:
        new = set()
        for p in prods:
            new.add(p * x0)
        prods |= new
    # now adding x: new products are p*x for p in prods; must not collide with prods or each other
    existing = set(prods)
    for p in prods:
        q = p * x
        if q in existing:
            return False
        existing.add(q)
    return True

def construct():
    t0 = time.time()
    primes = [a for a in range(2, 41) if _is_prime(a)]
    # primes: 2,3,5,7,11,13,17,19,23,29,31,37 -> 12 primes
    current = list(primes)
    best = list(current)

    composites = [a for a in range(2, 41) if not _is_prime(a)]

    # Precompute subset products of current for fast checks
    def subset_prods(nums):
        prods = {1}
        for x in nums:
            new = set()
            for p in prods:
                new.add(p * x)
            prods |= new
        return prods

    # Greedy pass: try to add composites in increasing order
    for x in composites:
        if time.time() - t0 > _DEADLINE:
            break
        if _can_add(current, x):
            current.append(x)
            if len(current) > len(best):
                best = list(current)

    # Randomized improvement: try swapping / adding via hill-climbing
    rng = random.Random(12345)
    improved = True
    while improved and (time.time() - t0) < _DEADLINE:
        improved = False
        # Try each composite not in current to add
        for x in composites:
            if x in current:
                continue
            if _can_add(current, x):
                current.append(x)
                improved = True
                if len(current) > len(best):
                    best = list(current)
                break
        # Try removing one element and adding two others (swap improvement)
        if not improved and (time.time() - t0) < _DEADLINE:
            # attempt a 1-for-2 swap
            for i in range(len(current)):
                if time.time() - t0 > _DEADLINE:
                    break
                saved = current.pop(i)
                added_any = False
                for x in composites:
                    if x == saved or x in current:
                        continue
                    if _can_add(current, x):
                        current.append(x)
                        added_any = True
                        break
                for x in composites:
                    if x == saved or x in current:
                        continue
                    if _can_add(current, x):
                        current.append(x)
                        added_any = True
                        break
                if added_any:
                    if len(current) > len(best):
                        best = list(current)
                    improved = True
                else:
                    current.append(saved)
                if improved:
                    break

    # Final verification
    if not _all_subset_products_distinct(best):
        # Fallback to primes
        best = list(primes)

    # Ensure Python ints and list
    return [int(x) for x in best]
# EVOLVE-BLOCK-END
