"""Sum-free subset maximizer: greedy + iterative local search (add / remove / swap)."""
import time
import random

_T0 = time.time()
_DEADLINE = _T0 + 115.0  # safety margin under 120 s wall-clock


def _instance(size, seed, limit=10000):
    rng = random.Random(seed)
    values = set()
    while len(values) < size:
        x = rng.randint(-limit, limit)
        if x:
            values.add(x)
    return sorted(values)


def _conflict_counts(elements):
    """For each element, count how many other elements it would 'break' if we required
    no a+b=c. Used to rank greedily (fewer conflicts = better to keep)."""
    s = set(elements)
    conf = {}
    for a in elements:
        c = 0
        for b in elements:
            if a + b in s:
                c += 1
        conf[a] = c
    return conf


def _greedy_conflict(elements):
    """Greedy: repeatedly add the element with fewest conflicts with already-chosen set."""
    s = set(elements)
    remaining = list(elements)
    chosen = set()
    chosen_list = []
    while remaining:
        best = None
        best_c = None
        for a in remaining:
            c = 0
            for b in chosen:
                if a + b in s:
                    c += 1
            if best_c is None or c < best_c:
                best_c = c
                best = a
        chosen.add(best)
        chosen_list.append(best)
        remaining.remove(best)
    return chosen_list


def _local_search(elements, start, deadline):
    """Iterative local search on a sum-free subset. Returns best set found."""
    s = set(elements)
    S = set(start)
    # ensure S is sum-free (defensive)
    S = _make_sum_free(elements, S)
    best = set(S)
    best_size = len(S)
    excluded = [x for x in elements if x not in S]
    rng = random.Random(12345)

    def is_ok(S, x):
        # adding x must not create a+b=c with all in S∪{x}
        for b in S:
            if x + b in S or b + x in S:
                return False
        if 2 * x in S:
            return False
        return True

    improved = True
    while improved and time.time() < deadline:
        improved = False
        # --- add phase: add any excluded element that keeps sum-free ---
        new_excl = []
        for x in excluded:
            if is_ok(S, x):
                S.add(x)
                improved = True
            else:
                new_excl.append(x)
        excluded = new_excl

        if time.time() >= deadline:
            break

        # --- remove phase: for each included x, removing it may let us add others ---
        for x in list(S):
            S.remove(x)
            gains = []
            for y in excluded:
                if is_ok(S, y):
                    S.add(y)
                    gains.append(y)
            if len(gains) > 1:
                improved = True
            else:
                # revert
                S.add(x)
                for y in gains:
                    S.discard(y)
        excluded = [x for x in elements if x not in S]

        if time.time() >= deadline:
            break

        # --- swap phase: remove one included, try to add two excluded ---
        for x in list(S):
            S.remove(x)
            added = []
            for y in excluded:
                if is_ok(S, y):
                    S.add(y)
                    added.append(y)
                    if len(added) >= 2:
                        break
            if len(added) >= 2:
                improved = True
            else:
                S.add(x)
                for y in added:
                    S.discard(y)
        excluded = [x for x in elements if x not in S]

        if len(S) > best_size:
            best = set(S)
            best_size = len(S)

    if len(S) > best_size:
        best = set(S)
    return best


def _make_sum_free(elements, S):
    """Force S to be sum-free by removing offending elements (keep smaller-magnitude? just remove c)."""
    s = set(elements)
    changed = True
    while changed:
        changed = False
        for a in list(S):
            for b in list(S):
                c = a + b
                if c in S:
                    S.discard(c)
                    changed = True
                    break
            if changed:
                break
    return S


def _solve_instance(elements, deadline):
    # multiple seeds / strategies, keep the best
    candidates = []

    # 1) odd parity
    candidates.append([x for x in elements if x % 2])
    # 2) even parity (sum of two evens is even, so evens are NOT sum-free generally; but 2a=b issues)
    # skip even; instead:
    # 2) greedy by conflict count
    try:
        candidates.append(_greedy_conflict(elements))
    except Exception:
        pass
    # 3) greedy by |x| descending (large numbers less likely to be a sum of two)
    candidates.append(sorted(elements, key=lambda x: -abs(x)))
    # 4) greedy by |x| ascending
    candidates.append(sorted(elements, key=lambda x: abs(x)))

    best = None
    best_size = -1
    for cand in candidates:
        cand = _make_sum_free(elements, list(cand))
        if len(cand) > best_size:
            best_size = len(cand)
            best = cand
        if time.time() > deadline:
            break

    if best is None:
        best = candidates[0]

    # local search from best, and from a couple of other candidates if time permits
    improved = _local_search(elements, best, deadline)
    if len(improved) > len(best):
        best = improved

    # try local search from the conflict-greedy candidate too if time allows
    if time.time() < deadline - 2 and len(candidates) > 1:
        alt = _make_sum_free(elements, list(candidates[1]))
        alt2 = _local_search(elements, alt, deadline)
        if len(alt2) > len(best):
            best = alt2

    return best


def construct():
    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        elements = _instance(size, seed)
        # allocate remaining time, reserve a little for later instances
        remaining = _DEADLINE - time.time()
        # give each instance a share; earlier (smaller) instances can be quicker
        if size <= 100:
            budget = min(remaining * 0.4, 40.0)
        elif size <= 200:
            budget = min(remaining * 0.35, 50.0)
        else:
            budget = min(remaining * 0.6, 60.0)
        dl = time.time() + max(1.0, budget)
        sol = _solve_instance(elements, dl)
        results.append(sorted(sol))
    return results
