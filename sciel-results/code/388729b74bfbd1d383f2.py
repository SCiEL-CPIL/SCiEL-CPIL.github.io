"""Rewritten program: maximize number of distinct integers in 2..40 with distinct subset products."""
# EVOLVE-BLOCK-START

import time
import random

def _check_distinct(nums):
    """Check if all subset products of nums are distinct."""
    k = len(nums)
    if k == 0:
        return True
    products = [1]
    for x in nums:
        new_products = [p * x for p in products]
        products.extend(new_products)
    return len(products) == len(set(products))

def construct():
    """
    Returns a list of integers in 2..40 with distinct subset products.
    Starts with prime-power construction (16 elements), then uses local
    search to try to improve within the time budget.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Baseline: prime-power construction giving 16 elements
    best = [2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Verify baseline
    if not _check_distinct(best):
        best = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    all_candidates = list(range(2, 41))
    
    # Phase 1: Try to add elements one at a time
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for cand in all_candidates:
            if cand in best:
                continue
            trial = best + [cand]
            if _check_distinct(trial):
                best = trial
                improved = True
                break
    
    # Phase 2: Try swaps to potentially enable further additions
    for _ in range(200):
        if time.time() > deadline:
            break
        current_size = len(best)
        # Try a swap
        i = random.randint(0, len(best) - 1)
        cand = random.choice(all_candidates)
        if cand == best[i]:
            continue
        trial = best[:]
        trial[i] = cand
        if _check_distinct(trial):
            best = trial
            # Try to add after swap
            for c2 in all_candidates:
                if c2 in best:
                    continue
                trial2 = best + [c2]
                if _check_distinct(trial2):
                    best = trial2
                    break
    
    return [int(x) for x in best]

# EVOLVE-BLOCK-END
