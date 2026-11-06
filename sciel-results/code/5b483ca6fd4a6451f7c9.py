"""Backtracking/greedy blocking-priority search for small maximal Sidon sets in [1,256]."""
import time
import random

N = 256

# EVOLVE-BLOCK-START
def _compute_sums(selected):
    """Compute all unordered pair sums a+b (a<=b) for a Sidon set."""
    sums = set()
    for i, a in enumerate(selected):
        for b in selected[i:]:
            sums.add(a + b)
    return sums

def _is_addable(x, selected, sums):
    """Check if x can be added to selected while preserving the Sidon property."""
    for a in selected:
        if x + a in sums:
            return False
    if 2 * x in sums:
        return False
    return True

def _is_maximal_bitset(selected, sums):
    """
    Fast maximality check.
    A missing element x is blocked iff (2*x in sums) or (x+a in sums for some a in selected).
    Returns True iff every missing element is blocked (inclusion-maximal).
    """
    sel_set = set(selected)
    if not sel_set:
        return False
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        blocked = (2 * x in sums)
        if not blocked:
            for a in selected:
                if x + a in sums:
                    blocked = True
                    break
        if not blocked:
            return False
    return True

def _greedy_construct(seed_elements, rng, time_limit):
    """
    Greedy construction: repeatedly add the addable element that blocks the most candidates.
    Returns (final_set, final_sums).
    """
    selected = list(seed_elements)
    sums = _compute_sums(selected)
    sel_set = set(selected)

    addable = set()
    for x in range(1, N + 1):
        if x not in sel_set and _is_addable(x, selected, sums):
            addable.add(x)

    while addable:
        if time.time() > time_limit:
            break

        best_x = None
        best_score = -1
        best_newly_blocked = set()

        candidates = list(addable)
        rng.shuffle(candidates)

        for x in candidates:
            new_sums = set()
            for a in selected:
                new_sums.add(x + a)
            new_sums.add(2 * x)

            newly_blocked = set()
            for y in addable:
                if y == x:
                    continue
                blocked = False
                if y + x in sums or y + x in new_sums:
                    blocked = True
                elif 2 * y in new_sums:
                    blocked = True
                else:
                    for a in selected:
                        if y + a in new_sums:
                            blocked = True
                            break
                if blocked:
                    newly_blocked.add(y)

            score = len(newly_blocked)
            if score > best_score:
                best_score = score
                best_x = x
                best_newly_blocked = newly_blocked

        if best_x is None:
            break

        sums.update([best_x + a for a in selected] + [2 * best_x])
        selected.append(best_x)
        sel_set.add(best_x)
        addable.discard(best_x)
        addable -= best_newly_blocked

        # recheck remaining addable elements
        still = set()
        for x in addable:
            if _is_addable(x, selected, sums):
                still.add(x)
        addable = still

    return selected, sums

def _support_order(selected, sums):
    """
    For each element x in selected, compute support = number of OTHER elements
    y in selected that would become addable if x were removed (i.e. x currently
    blocks y). Low-support elements are the easiest to remove while staying maximal.
    Returns indices sorted by ascending support (ties broken randomly).
    """
    sel_set = set(selected)
    n = len(selected)
    support = [0] * n
    for i, x in enumerate(selected):
        # sums involving x (x<=a): {x+a : a in selected, a>=x} union {2x}
        x_sums = set()
        for a in selected:
            if a >= x:
                x_sums.add(x + a)
        x_sums.add(2 * x)
        # y is blocked by x iff (2y in x_sums) or (y+a in x_sums for some a in selected)
        for j, y in enumerate(selected):
            if j == i:
                continue
            if 2 * y in x_sums:
                support[i] += 1
                continue
            for a in selected:
                if y + a in x_sums:
                    support[i] += 1
                    break
    # sort by ascending support, randomize ties
    order = list(range(n))
    order.sort(key=lambda idx: (support[idx], random.random()))
    return order

def _local_search_remove(selected, sums, rng, time_limit):
    """
    Repeatedly remove a low-support element while the set stays inclusion-maximal.
    Always returns a valid maximal set (the current best found before the deadline).
    """
    selected = list(selected)
    sums = _compute_sums(selected)
    sel_set = set(selected)

    best = (len(selected), list(selected))

    while time.time() < time_limit:
        if len(selected) == 0:
            break
        order = _support_order(selected, sums)
        improved = False
        for i in order:
            if time.time() > time_limit:
                break
            x = selected[i]
            # sums removed when x is dropped
            removed = set()
            for a in selected:
                if a >= x:
                    removed.add(x + a)
            removed.add(2 * x)
            new_sums = sums - removed
            new_selected = selected[:i] + selected[i + 1:]
            if _is_maximal_bitset(new_selected, new_sums):
                selected = new_selected
                sums = new_sums
                sel_set = sel_set - {x}
                improved = True
                if len(selected) < best[0]:
                    best = (len(selected), list(selected))
                break
        if not improved:
            break

    return best[1], _compute_sums(best[1])

def _verify_maximal(selected):
    """Verify selected is Sidon and inclusion-maximal."""
    seen = set()
    for i, a in enumerate(selected):
        for b in selected[i:]:
            t = a + b
            if t in seen:
                return False
            seen.add(t)
    return _is_maximal_bitset(selected, seen)

def construct():
    start_time = time.time()
    deadline = start_time + 27.0  # safety margin before the 30s limit

    # Initial upper bound from simple ascending greedy
    simple_selected, simple_sums = [], set()
    for x in range(1, N + 1):
        added = [x + a for a in simple_selected] + [2 * x]
        if len(set(added)) == len(added) and not simple_sums.intersection(added):
            simple_selected.append(x)
            simple_sums.update(added)

    best_set = list(simple_selected)
    best_size = len(best_set)

    # Force maximality if needed
    if not _verify_maximal(best_set):
        changed = True
        while changed:
            changed = False
            sums = _compute_sums(best_set)
            sel_set = set(best_set)
            for x in range(1, N + 1):
                if x not in sel_set and _is_addable(x, best_set, sums):
                    best_set.append(x)
                    changed = True
    best_size = len(best_set)

    # Shrink the initial solution
    shrunk, _ = _local_search_remove(best_set, _compute_sums(best_set), random.Random(1), deadline - 1.0)
    if _verify_maximal(shrunk) and len(shrunk) < best_size:
        best_size = len(shrunk)
        best_set = shrunk

    rng = random.Random(42)

    while time.time() < deadline - 2.0:
        # random seed
        seed_size = rng.randint(0, 5)
        seed_elements = rng.sample(range(1, N + 1), min(seed_size, N))

        # check seed is Sidon
        seed_ok = True
        seed_sums = set()
        for i, a in enumerate(seed_elements):
            for b in seed_elements[i:]:
                s = a + b
                if s in seed_sums:
                    seed_ok = False
                    break
                seed_sums.add(s)
            if not seed_ok:
                break
        if not seed_ok:
            continue

        final_set, final_sums = _greedy_construct(seed_elements, rng, deadline - 1.5)

        if not _verify_maximal(final_set):
            continue

        current_size = len(final_set)
        if current_size < best_size:
            best_size = current_size
            best_set = list(final_set)

        # shrink (always returns a valid maximal set)
        if current_size <= best_size:
            shrunk, _ = _local_search_remove(final_set, final_sums, rng, deadline - 1.0)
            if _verify_maximal(shrunk) and len(shrunk) < best_size:
                best_size = len(shrunk)
                best_set = list(shrunk)

    return best_set
# EVOLVE-BLOCK-END
