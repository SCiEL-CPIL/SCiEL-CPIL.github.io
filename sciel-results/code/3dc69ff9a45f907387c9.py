import time
from itertools import combinations

L = 5354228880  # lcm(1..24)
N = 24
ALL = list(range(1, 25))

# Precompute reciprocals as L//i
RECIP = [0] + [L // i for i in range(1, 25)]  # RECIP[i] = L//i


_has_subsum_cache = {}

def has_subsum(recips_list, target, exclude_idx=None):
    """
    Check if target can be formed by a subset of recips_list.
    recips_list: list of (L//i) values (positive ints).
    exclude_idx: index in recips_list to exclude (or None).
    Returns True if some non-empty subset sums to exactly target.
    """
    # Build a cache key
    if exclude_idx is None:
        key = (tuple(recips_list), target, None)
    else:
        key = (tuple(recips_list), target, exclude_idx)
    
    if key in _has_subsum_cache:
        return _has_subsum_cache[key]

    # Collect eligible weights
    if exclude_idx is None:
        weights = [w for w in recips_list if 0 < w <= target]
    else:
        weights = [w for i, w in enumerate(recips_list) if i != exclude_idx and 0 < w <= target]

    if not weights:
        _has_subsum_cache[key] = False
        return False
    if target == 0:
        _has_subsum_cache[key] = False
        return False

    # Deduplicate and sort
    weights = sorted(set(weights))
    n = len(weights)
    if n == 0:
        _has_subsum_cache[key] = False
        return False

    # If sum of all < target, impossible
    if sum(weights) < target:
        _has_subsum_cache[key] = False
        return False

    # Quick check: if target is in weights, done
    if target in weights:
        _has_subsum_cache[key] = True
        return True

    # Meet-in-the-middle
    mid = n // 2
    left = weights[:mid]
    right = weights[mid:]

    def gen_sums(arr):
        sums = {0}
        for w in arr:
            new = set()
            for s in sums:
                ns = s + w
                if ns <= target:
                    new.add(ns)
            sums |= new
        return sums

    left_sums = gen_sums(left)
    right_sums = gen_sums(right)

    # Check
    result = False
    for s in left_sums:
        if (target - s) in right_sums:
            result = True
            break
    _has_subsum_cache[key] = result
    return result


def find_violation(S):
    """
    S: list of ints.
    Returns (violator_idx, contributing_subset) if violation, else (None, None).
    A violation: for some x in S, L//x == sum of L//y for a non-empty subset of S\{x}.
    """
    if not S:
        return None, None
    m = len(S)
    # Precompute recip list aligned with S
    recip_list = [L // x for x in S]
    for i, x in enumerate(S):
        target = L // x
        if has_subsum(recip_list, target, exclude_idx=i):
            return i, None
    return None, None


def is_valid(S):
    """
    S: list of ints. Return True if S is a valid reciprocal-subsum-free set.
    """
    if not S:
        return True
    recip_list = [L // x for x in S]
    for i in range(len(S)):
        target = L // S[i]
        if has_subsum(recip_list, target, exclude_idx=i):
            return False
    return True


def construct():
    start = time.time()
    deadline = start + 85.0

    def now():
        return time.time()

    best_set = None
    best_size = 0

    # Start with a known-valid seed: the upper half [13..24] is 12 elements
    # Check it
    seed = list(range(13, 25))
    if is_valid(seed):
        best_set = seed[:]
        best_size = len(seed)
    else:
        # Fallback to a trivially valid small set
        best_set = [24]
        best_size = 1

    # Strategy: start from a large random set, repair by removing violators,
    # then greedily add back missing elements.
    # We also do local search: try swapping.

    # First, try a greedy construction from scratch aiming for max size.
    # Greedy: add elements one by one if it keeps validity, trying to maximize size.
    # Try multiple orderings.

    import random

    def greedy_add(order, start_set=None):
        """
        order: list of candidate elements to try adding (in that order).
        start_set: initial set (list). 
        Returns a valid set.
        """
        S = set(start_set) if start_set is not None else set()
        S_list = list(S)
        for x in order:
            if now() > deadline:
                break
            if x in S:
                continue
            trial = S_list + [x]
            if is_valid(trial):
                S.add(x)
                S_list = list(S)
        return S_list

    # Try greedy from empty with various orderings
    # Ordering by large recip first (small numbers) tends to create more violations.
    # We want to include as many as possible.

    # Orderings:
    orderings = []
    orderings.append(ALL[:])                    # 1..24
    orderings.append(ALL[::-1])                  # 24..1
    orderings.append(sorted(ALL, key=lambda x: L // x))       # smallest recip first
    orderings.append(sorted(ALL, key=lambda x: L // x, reverse=True))  # largest recip first
    orderings.append(sorted(ALL, key=lambda x: x % 7))

    # Add some random orderings
    rng = random.Random(42)
    for _ in range(10):
        o = ALL[:]
        rng.shuffle(o)
        orderings.append(o)

    for o in orderings:
        if now() > deadline:
            break
        res = greedy_add(o, start_set=None)
        if len(res) > best_size:
            best_size = len(res)
            best_set = res[:]

    # Now do local search / repair from full set
    # Start with full set, repeatedly remove a violator until valid, then add back.

    def repair(S):
        """Remove violators until S is valid. Returns list."""
        S = set(S)
        while now() < deadline and S:
            S_list = list(S)
            viol_idx, _ = find_violation(S_list)
            if viol_idx is None:
                return S_list
            S.remove(S_list[viol_idx])
        return list(S)

    # Repair full set
    S_full = repair(ALL[:])
    if len(S_full) > best_size:
        best_size = len(S_full)
        best_set = S_full[:]

    # Now try to add back missing elements to repaired sets
    def add_back(S):
        """Try to add missing elements greedily."""
        S = set(S)
        missing = [x for x in ALL if x not in S]
        # Try in order of largest recip (smallest x) first? or random
        for x in missing:
            if now() > deadline:
                break
            trial = list(S) + [x]
            if is_valid(trial):
                S.add(x)
        return list(S)

    # After repair, add back
    if best_set is not None:
        improved = add_back(best_set)
        if len(improved) > best_size:
            best_size = len(improved)
            best_set = improved[:]

    # Now do multiple random restarts of repair + add back
    # Also try random subsets as starting points.

    # Local search: swap operations
    # Given a valid set, try removing one element and adding another (not in set)
    # if it stays valid and we don't decrease size (size stays same) - but we want to grow.
    # Actually for growing, the add-back already handles it.
    # For improving a stuck local optimum, swap can help.

    def local_search(S, max_iters=200):
        """
        S: valid list. Try swaps and additions to improve.
        Returns improved list (same or larger).
        """
        S = set(S)
        S_list = list(S)
        iters = 0
        improved = False
        while iters < max_iters and now() < deadline:
            iters += 1
            improved = False

            # Try additions first
            missing = [x for x in ALL if x not in S]
            if missing:
                rng2 = random.Random(iters + len(S) * 1000)
                rng2.shuffle(missing)
                for x in missing:
                    if now() > deadline:
                        break
                    trial = list(S) + [x]
                    if is_valid(trial):
                        S.add(x)
                        S_list = list(S)
                        improved = True
                        break
            if improved:
                continue

            # Try swaps: remove a, add b
            if len(S) > 0:
                # Randomly pick a to remove
                a = rng.choice(S_list)
                missing = [x for x in ALL if x not in S]
                if missing:
                    rng.shuffle(missing)
                    for b in missing:
                        if now() > deadline:
                            break
                        new_set = (S - {a}) | {b}
                        if is_valid(list(new_set)):
                            S = new_set
                            S_list = list(S)
                            improved = True
                            break

            if not improved:
                break

        return S_list

    # Apply local search to best
    if best_set is not None:
        ls = local_search(best_set, max_iters=100)
        if len(ls) > best_size:
            best_size = len(ls)
            best_set = ls[:]

    # Random restarts: pick random subsets, repair, add back, local search
    for _ in range(40):
        if now() > deadline:
            break
        # Random subset of size between best_size and 24
        target_size = random.randint(max(2, best_size), 24)
        rand_set = random.sample(ALL, min(target_size, 24))
        repaired = repair(rand_set)
        added = add_back(repaired)
        if len(added) > best_size:
            best_size = len(added)
            best_set = added[:]
            ls = local_search(added, max_iters=80)
            if len(ls) > best_size:
                best_size = len(ls)
                best_set = ls[:]
        else:
            # Even if not larger, try local search on the repaired set
            # as it might find a different local optimum
            ls = local_search(added, max_iters=40)
            if len(ls) > best_size:
                best_size = len(ls)
                best_set = ls[:]

    # Also try to directly find size-24 or 23 if possible
    # Brute force over subsets of missing elements is expensive; skip.

    # Final polish: if we have time, do one more round of local search on best_set
    if best_set is not None and now() < deadline - 5.0:
        ls_final = local_search(best_set, max_iters=200)
        if len(ls_final) > best_size:
            best_size = len(ls_final)
            best_set = ls_final[:]

    # Final: ensure validity
    if best_set is None or not is_valid(best_set):
        # Fallback
        best_set = [24]
        best_size = 1
    else:
        best_set = list(best_set)
        best_size = len(best_set)

    return best_set


if __name__ == "__main__":
    print(construct())
