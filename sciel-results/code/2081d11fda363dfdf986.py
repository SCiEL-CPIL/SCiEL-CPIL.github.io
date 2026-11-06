"""Optimized solver: fast incremental validity + multi-restart local search."""
import time
import random
from collections import defaultdict

N = 192
MAX_K = 48


def _check_validity(a, others):
    """Check if a divides sum of any 3 distinct elements in others.
    others is a list of integers (not including a).
    Returns True if valid (no such triple exists).
    """
    m = len(others)
    if m < 3:
        return True
    rem = [v % a for v in others]
    # For each pair (i,j), check if exists k != i,j with rem[k] == (-rem[i]-rem[j]) % a
    # Use suffix approach: for pair (i,j) with i<j, check if any k>j has target remainder
    # Build suffix presence
    suffix = [None] * (m + 1)
    suffix[m] = set()
    for i in range(m - 1, -1, -1):
        s = set(suffix[i + 1])
        s.add(rem[i])
        suffix[i] = s
    for i in range(m):
        for j in range(i + 1, m):
            target = (-rem[i] - rem[j]) % a
            if j + 1 < m and target in suffix[j + 1]:
                return False
    return True


def _is_valid_set(chosen):
    chosen_set = set(chosen)
    for a in chosen:
        others = [v for v in chosen if v != a]
        if not _check_validity(a, others):
            return False
    return True


def _greedy_seed(order=None):
    """Greedy: add x if valid. order is a list of candidates to try."""
    if order is None:
        order = list(range(N, 0, -1))
    selected = []
    selected_set = set()
    for x in order:
        if x in selected_set:
            continue
        others = list(selected)
        # Check x as divisor: triple among selected
        if not _check_validity(x, others):
            continue
        # Check each a in selected: triple among (selected without a) plus x
        ok = True
        for a in selected:
            others_a = [v for v in selected if v != a] + [x]
            if not _check_validity(a, others_a):
                ok = False
                break
        if ok:
            selected.append(x)
            selected_set.add(x)
            if len(selected) >= MAX_K:
                break
    return selected


def _local_search(initial, time_budget, rng):
    """Local search with add and swap moves."""
    best = list(initial)
    best_len = len(best)
    start = time.time()
    best_set = set(best)

    def try_add(x, s, s_set):
        if x in s_set:
            return None, None
        others = list(s)
        if not _check_validity(x, others):
            return None, None
        for a in s:
            others_a = [v for v in s if v != a] + [x]
            if not _check_validity(a, others_a):
                return None, None
        return s + [x], s_set | {x}

    def try_swap(x, remove_idx, s, s_set):
        removed = s[remove_idx]
        if x == removed:
            return None, None
        new_s = [v for i, v in enumerate(s) if i != remove_idx] + [x]
        new_set = s_set - {removed} | {x}
        for a in new_s:
            others = [v for v in new_s if v != a]
            if not _check_validity(a, others):
                return None, None
        return new_s, new_set

    # Phase 1: greedy additions in random order
    pool = [x for x in range(1, N + 1) if x not in best_set]
    rng.shuffle(pool)
    for x in pool:
        if time.time() - start > time_budget:
            break
        if best_len >= MAX_K:
            break
        new_s, new_set = try_add(x, best, best_set)
        if new_s is not None:
            best = new_s
            best_set = new_set
            best_len = len(best)
            pool.remove(x)

    # Phase 2: random swaps
    for _ in range(500000):
        if time.time() - start > time_budget:
            break
        if best_len >= MAX_K:
            break
        pool = [x for x in range(1, N + 1) if x not in best_set]
        if not pool:
            break
        x = rng.choice(pool)
        ri = rng.randrange(len(best))
        new_s, new_set = try_swap(x, ri, best, best_set)
        if new_s is not None:
            best = new_s
            best_set = new_set
            best_len = len(best)

    # Phase 3: final greedy pass
    pool = [x for x in range(1, N + 1) if x not in best_set]
    pool.sort(reverse=True)
    for x in pool:
        if time.time() - start > time_budget:
            break
        if best_len >= MAX_K:
            break
        new_s, new_set = try_add(x, best, best_set)
        if new_s is not None:
            best = new_s
            best_set = new_set
            best_len = len(best)

    return best


def construct():
    start = time.time()
    deadline = start + 75.0  # safety margin

    best_result = []
    best_len = 0

    # Multiple restarts with different orderings
    rng = random.Random(42)

    # First: descending greedy
    seed1 = _greedy_seed(list(range(N, 0, -1)))
    if len(seed1) > best_len:
        best_result = seed1
        best_len = len(seed1)

    # Local search from seed1
    if time.time() < deadline:
        result = _local_search(seed1, min(30.0, deadline - time.time()), random.Random(1))
        if len(result) > best_len:
            best_result = result
            best_len = len(result)

    # Try ascending greedy
    if time.time() < deadline:
        seed2 = _greedy_seed(list(range(1, N + 1)))
        if len(seed2) > best_len:
            best_result = seed2
            best_len = len(seed2)
        result = _local_search(seed2, min(20.0, deadline - time.time()), random.Random(2))
        if len(result) > best_len:
            best_result = result
            best_len = len(result)

    # Random orderings
    for trial in range(10):
        if time.time() >= deadline - 5.0:
            break
        order = list(range(1, N + 1))
        rng.shuffle(order)
        seed = _greedy_seed(order)
        if len(seed) > best_len:
            best_result = seed
            best_len = len(seed)
        remaining = deadline - time.time() - 2.0
        if remaining > 3.0:
            result = _local_search(seed, min(5.0, remaining), random.Random(100 + trial))
            if len(result) > best_len:
                best_result = result
                best_len = len(result)

    # Final: try to add more to best_result
    if time.time() < deadline - 1.0:
        remaining = deadline - time.time() - 1.0
        result = _local_search(best_result, remaining, random.Random(999))
        if len(result) > best_len:
            best_result = result
            best_len = len(result)

    # Cap and validate
    if len(best_result) > MAX_K:
        best_result = best_result[:MAX_K]
    if not _is_valid_set(best_result):
        # Fallback: use greedy seed
        best_result = seed1
        if len(best_result) > MAX_K:
            best_result = best_result[:MAX_K]

    return best_result


if __name__ == "__main__":
    print(construct())
