"""Divisibility-incomparable subset-sum search.

Combined design:
  * Structural core (from inspirations): two-step swap-then-add local search,
    which escapes local optima that pure single-step greedy addition cannot,
    plus the fast interval shortcut (max_sum < 2*min_sum  =>  no divisibility).
  * Refinement machinery (from current program): simulated-annealing-style
    random perturbations with greedy refill, multiple diverse candidate pools,
    and a strict wall-clock deadline guard with a safety margin.
"""
# EVOLVE-BLOCK-START
import time
import random

def _subset_sums(nums):
    """Return the set of all non-empty subset sums of nums."""
    sums = set()
    for x in nums:
        new_sums = set()
        for s in sums:
            new_sums.add(s + x)
        new_sums.add(x)
        sums |= new_sums
    return sums

def _check_validity(nums):
    """Full validity check: distinct sums AND no two distinct sums divide one another."""
    n = len(nums)
    if n == 0:
        return True
    sums = _subset_sums(nums)
    if len(sums) != (1 << n) - 1:
        return False
    sums_list = sorted(sums)
    sums_set = set(sums_list)
    max_sum = sums_list[-1]
    for s in sums_list:
        for k in range(2, max_sum // s + 1):
            if k * s in sums_set:
                return False
    return True

def _check_validity_fast(nums):
    """Fast validity check using the interval shortcut.

    If the largest subset sum is less than twice the smallest, then no two
    distinct positive sums can divide one another (any multiple k*s with k>=2
    is >= 2*min > max), so only sum-distinctness matters.
    """
    n = len(nums)
    if n == 0:
        return True
    sums = _subset_sums(nums)
    if len(sums) != (1 << n) - 1:
        return False
    sums_list = sorted(sums)
    min_s = sums_list[0]
    max_s = sums_list[-1]
    if max_s < 2 * min_s:
        return True
    sums_set = set(sums_list)
    for s in sums_list:
        for k in range(2, max_s // s + 1):
            if k * s in sums_set:
                return False
    return True

def _try_add(current, candidate, deadline):
    """Return True if adding candidate to current keeps the set valid."""
    if time.time() > deadline:
        return False
    if candidate in current:
        return False
    return _check_validity_fast(current + [candidate])

def _greedy_extend(current, candidates, deadline, cap):
    """Greedily add as many valid candidates as possible (in given order)."""
    current = current[:]
    for num in candidates:
        if len(current) >= cap:
            break
        if num in current:
            continue
        if _try_add(current, num, deadline):
            current.append(num)
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 78  # explicit safety margin before the 90s kill

    def timed():
        return time.time() > deadline

    # ---- Verified starting point: a tight high range where the interval
    #      shortcut (max < 2*min) tends to hold, making validity cheap. ----
    best = [128, 129, 130, 131, 132, 133, 134]
    if not _check_validity(best):
        best = [128, 131, 134, 137, 140, 143, 146]
    if not _check_validity(best):
        best = [31, 32, 34]
    best = best[:]
    best_size = len(best)

    all_candidates = list(range(1, 257))

    # Diverse candidate pools so the search explores different structure families.
    pools = [
        all_candidates,
        list(range(120, 257)),
        list(range(90, 257)),
        list(range(60, 257)),
        list(range(1, 120)),
    ]

    # ---- Phase 1: greedy construction from many random orderings. ----
    for trial in range(120):
        if timed():
            break
        rng = random.Random(trial * 7919 + 13)
        pool = rng.choice(pools)
        order = pool[:]
        rng.shuffle(order)
        current = []
        for num in order:
            if len(current) >= 16:
                break
            if _try_add(current, num, deadline):
                current.append(num)
        if len(current) > best_size:
            best = current[:]
            best_size = len(best)

    # ---- Phase 2: two-step (swap-then-add) local search with greedy refill.
    #      This is the structural core: removing an element and replacing it
    #      with a different one can open room for a net size increase that a
    #      pure single-addition step can never achieve. ----
    no_improve = 0
    iter_id = 0
    while not timed():
        iter_id += 1
        if no_improve > 400:
            break
        rng = random.Random(int(time.time() * 1000) * 31 + iter_id * 104729)
        move_type = rng.random()
        perturbed = best[:]

        if move_type < 0.45 and len(perturbed) > 1:
            # Remove 1 (mostly) or 2 elements, then refill greedily.
            num_remove = 1 if rng.random() < 0.7 else 2
            idxs = rng.sample(range(len(perturbed)), min(num_remove, len(perturbed)))
            for idx in sorted(idxs, reverse=True):
                perturbed.pop(idx)
        elif move_type < 0.75 and len(perturbed) > 0:
            # Swap one element for a random candidate (net size unchanged),
            # then try to greedily grow.
            idx = rng.randint(0, len(perturbed) - 1)
            new_val = rng.choice(all_candidates)
            if new_val not in perturbed:
                perturbed[idx] = new_val
        elif move_type < 0.92 and len(perturbed) > 2:
            # Two-step: drop 2, add 1 different (net -1), then refill.
            idxs = rng.sample(range(len(perturbed)), 2)
            for idx in sorted(idxs, reverse=True):
                perturbed.pop(idx)
            new_val = rng.choice(all_candidates)
            if new_val not in perturbed:
                perturbed.append(new_val)
        else:
            # Pure refill attempt with a fresh candidate ordering.
            pass

        pool = rng.choice(pools)
        order = pool[:]
        rng.shuffle(order)
        extended = _greedy_extend(perturbed, order, deadline, cap=20)

        if len(extended) > best_size:
            best = extended
            best_size = len(best)
            no_improve = 0
        else:
            no_improve += 1

    # ---- Final validation with a guaranteed-valid fallback chain. ----
    if not _check_validity(best):
        for fb in ([128, 131, 134, 137, 140, 143, 146], [31, 32, 34], [31, 32]):
            if _check_validity(fb):
                best = fb
                break
    else:
        best = best[:]

    # Exact type conversion: plain Python ints inside a list.
    return [int(x) for x in best]
# EVOLVE-BLOCK-END
