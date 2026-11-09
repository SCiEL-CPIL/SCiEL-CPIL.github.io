import time
from itertools import combinations
from functools import lru_cache

L = 5354228880  # lcm(1..24)
N = 24
ALL = list(range(1, 25))

# Precompute reciprocals as L//i
RECIP = [0] + [L // i for i in range(1, 25)]  # RECIP[i] = L//i

# Precompute for each element x: the set of all achievable subset sums from {RECIP[y] : y != x}
# This is expensive (2^23 per x) but we can use a smarter approach:
# For each x, compute all subset sums of the other 23 elements once.

# Actually, let's precompute a global structure:
# For each element x, we'll compute the set of all subset sums of {RECIP[y] : y in [1,24], y != x}
# But that's 24 * 2^23 which is too much.

# Instead, use memoized subset-sum checks with a faster algorithm.

@lru_cache(maxsize=200000)
def _can_form(target, weights_tuple):
    """Check if target can be formed by a non-empty subset of weights_tuple (sorted)."""
    if not weights_tuple or target <= 0:
        return False
    # Quick check: if target equals any single weight
    for w in weights_tuple:
        if w == target:
            return True
        if w > target:
            break
    # DP with set of reachable sums, bounded by target
    reachable = {0}
    for w in weights_tuple:
        if w > target:
            break
        new_sums = set()
        for s in reachable:
            ns = s + w
            if ns <= target:
                new_sums.add(ns)
                if ns == target:
                    return True
        reachable |= new_sums
    return False


def has_subsum(weights, target):
    """Check if target can be formed by a non-empty subset of weights."""
    if not weights or target <= 0:
        return False
    if target > sum(weights):
        return False
    sorted_w = tuple(sorted(weights))
    return _can_form(target, sorted_w)


def is_valid(S):
    """
    S: list of ints. Return True if S is a valid reciprocal-subsum-free set.
    """
    if not S:
        return True
    recip_list = [L // x for x in S]
    m = len(S)
    for i in range(m):
        target = recip_list[i]
        others = tuple(sorted(recip_list[j] for j in range(m) if j != i))
        if target > sum(others):
            continue
        if _can_form(target, others):
            return False
    return True


def construct():
    start = time.time()
    deadline = start + 85.0  # safety margin before 90s kill

    def now():
        return time.time()

    import random

    best_set = None
    best_size = 0

    # Try several seed sets
    seeds = [
        list(range(13, 25)),  # upper half
        list(range(10, 25)),  # upper 15
        list(range(8, 25)),   # upper 17
        [1, 2, 4, 8, 16, 24], # powers of 2 and 24
        list(range(1, 13)),   # lower half
    ]
    for seed in seeds:
        if is_valid(seed):
            if len(seed) > best_size:
                best_size = len(seed)
                best_set = seed[:]
    if best_set is None:
        best_set = [24]
        best_size = 1

    # Greedy addition with multiple orderings
    def greedy_add(order, start_set=None):
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

    rng = random.Random(42)
    orderings = []
    # Deterministic orderings
    orderings.append(ALL[:])
    orderings.append(ALL[::-1])
    orderings.append(sorted(ALL, key=lambda x: L // x))
    orderings.append(sorted(ALL, key=lambda x: L // x, reverse=True))
    orderings.append(sorted(ALL, key=lambda x: x % 7))
    orderings.append(sorted(ALL, key=lambda x: x % 13))
    orderings.append(sorted(ALL, key=lambda x: L // x % 100))
    # Random orderings
    for _ in range(40):
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

    # Repair: start from full set, remove violators one by one
    def find_violation(S):
        if not S:
            return None
        recip_list = [L // x for x in S]
        m = len(S)
        for i in range(m):
            target = recip_list[i]
            others = [recip_list[j] for j in range(m) if j != i]
            if target <= sum(others):
                if _can_form(target, tuple(sorted(others))):
                    return i
        return None

    def repair(S):
        S = set(S)
        while now() < deadline and S:
            S_list = list(S)
            viol_idx = find_violation(S_list)
            if viol_idx is None:
                return S_list
            S.remove(S_list[viol_idx])
        return list(S)

    S_full = repair(ALL[:])
    if len(S_full) > best_size:
        best_size = len(S_full)
        best_set = S_full[:]

    # Add back missing elements
    def add_back(S):
        S = set(S)
        missing = [x for x in ALL if x not in S]
        rng_local = random.Random(len(S) * 7 + 13)
        rng_local.shuffle(missing)
        for x in missing:
            if now() > deadline:
                break
            trial = list(S) + [x]
            if is_valid(trial):
                S.add(x)
        return list(S)

    if best_set is not None:
        improved = add_back(best_set)
        if len(improved) > best_size:
            best_size = len(improved)
            best_set = improved[:]

    # Local search with additions and swaps
    def local_search(S, max_iters=300):
        S = set(S)
        S_list = list(S)
        iters = 0
        while iters < max_iters and now() < deadline:
            iters += 1
            improved = False

            # Try additions
            missing = [x for x in ALL if x not in S]
            if missing:
                rng2 = random.Random(iters * 1000 + len(S) * 7)
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

            # Try swaps: remove one, add another
            if len(S) > 0:
                a = rng.choice(list(S))
                missing = [x for x in ALL if x not in S]
                if missing:
                    rng3 = random.Random(iters * 13 + 42)
                    rng3.shuffle(missing)
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

    if best_set is not None:
        ls = local_search(best_set, max_iters=300)
        if len(ls) > best_size:
            best_size = len(ls)
            best_set = ls[:]

    # Random restarts with repair and local search
    for _ in range(80):
        if now() > deadline:
            break
        target_size = random.randint(max(2, best_size - 2), min(24, best_size + 4))
        rand_set = random.sample(ALL, target_size)
        repaired = repair(rand_set)
        added = add_back(repaired)
        if len(added) > best_size:
            best_size = len(added)
            best_set = added[:]
            ls = local_search(added, max_iters=150)
            if len(ls) > best_size:
                best_size = len(ls)
                best_set = ls[:]

    # Final validation
    if best_set is None or not is_valid(best_set):
        best_set = [24]
        best_size = 1
    else:
        best_set = list(best_set)
        best_size = len(best_set)

    return best_set


if __name__ == "__main__":
    print(construct())
