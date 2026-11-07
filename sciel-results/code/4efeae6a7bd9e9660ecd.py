"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

def _is_valid_subset(subset):
    k = len(subset)
    if k <= 1:
        return True
    products = set()
    for mask in range(1 << k):
        prod = 1
        for i in range(k):
            if mask & (1 << i):
                prod *= subset[i]
        if prod in products:
            return False
        products.add(prod)
    return True

def _prime_factorization(n):
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    all_nums = list(range(2, 41))
    factorizations = {n: _prime_factorization(n) for n in all_nums}
    
    best = []
    
    # Multiple greedy orderings
    def sort_key1(n):
        pf = factorizations[n]
        distinct = len(pf)
        is_prime = (distinct == 1 and n in pf)
        return (0 if is_prime else 1, distinct, n)
    
    orderings = [
        sorted(all_nums, key=sort_key1),
        sorted(all_nums),
        sorted(all_nums, reverse=True),
        sorted(all_nums, key=lambda n: (len(factorizations[n]), n)),
    ]
    
    for order in orderings:
        if time.time() > deadline:
            break
        result = []
        for num in order:
            if time.time() > deadline:
                break
            trial = result + [num]
            if _is_valid_subset(trial):
                result = trial
        if len(result) > len(best):
            best = result
    
    # Try to extend
    if time.time() < deadline:
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for candidate in all_nums:
                if time.time() > deadline:
                    break
                if candidate in best:
                    continue
                trial = best + [candidate]
                if _is_valid_subset(trial):
                    best = trial
                    changed = True
    
    # Local search with swaps to enable additions
    if time.time() < deadline:
        current = list(best)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                for candidate in all_nums:
                    if time.time() > deadline:
                        break
                    if candidate in current:
                        continue
                    trial = current.copy()
                    trial[i] = candidate
                    if _is_valid_subset(trial):
                        # Try to add more
                        for add_cand in all_nums:
                            if add_cand in trial:
                                continue
                            trial2 = trial + [add_cand]
                            if _is_valid_subset(trial2):
                                current = trial2
                                improved = True
                                break
                        if not improved and len(trial) >= len(current):
                            current = trial
                            improved = True
                        break
                if improved:
                    break
        if len(current) > len(best):
            best = current
    
    return [int(x) for x in best]
# EVOLVE-BLOCK-END
