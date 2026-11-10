"""Small inclusion-maximal Sidon sets in [1,256].

Strategy:
  1. Build a maximal Sidon set by a randomized greedy (random order, with a few
     deterministic heuristics interleaved).
  2. Minimize the size by repeatedly removing any element whose removal leaves
     the set Sidon and still inclusion-maximal (with a randomized removal order
     to escape local optima).
  3. Occasionally perform a "swap" move: remove one element and add another
     missing element, provided the result is Sidon and maximal, and the total
     size does not increase (ideally decreases).

All operations are guarded by an explicit deadline.
"""
import time
import random

N = 256
TIME_LIMIT = 55.0  # safety margin before 60 s kill


def _is_sidon(selected):
    """Check whether selected is a Sidon set (all a+b for a<=b distinct)."""
    sums = set()
    n = len(selected)
    for i in range(n):
        a = selected[i]
        for j in range(i, n):
            b = selected[j]
            s = a + b
            if s in sums:
                return False
            sums.add(s)
    return True


def _is_maximal(selected, N):
    """Check whether selected is inclusion-maximal in [1, N]."""
    sel_set = set(selected)
    sums = set()
    n = len(selected)
    for i in range(n):
        a = selected[i]
        for j in range(i, n):
            sums.add(a + selected[j])

    for x in range(1, N + 1):
        if x in sel_set:
            continue
        # Check whether adding x would break the Sidon property.
        ok = True
        new_sums = set()
        for a in selected:
            s = a + x
            if s in sums or s in new_sums:
                ok = False
                break
            new_sums.add(s)
        if not ok:
            continue
        s = 2 * x
        if s in sums or s in new_sums:
            continue
        # x can be added, so the set is not maximal.
        return False
    return True


def _greedy_construct(order):
    """Greedy construction: add elements in the given order if possible."""
    selected = []
    sums = set()
    for x in order:
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not (sums & set(added)):
            selected.append(x)
            sums.update(added)
    return selected


def _try_remove_any(selected, N, rng=None):
    """Try to remove one element (random order if rng given) while keeping
    the set Sidon and maximal. Returns the new list, or None if no removal
    succeeded."""
    idxs = list(range(len(selected)))
    if rng is not None:
        rng.shuffle(idxs)
    else:
        idxs.reverse()  # remove from the end first
    for i in idxs:
        new_sel = selected[:i] + selected[i + 1:]
        if _is_sidon(new_sel) and _is_maximal(new_sel, N):
            return new_sel
    return None


def _try_swap(selected, N, rng, time_budget):
    """Try a swap: remove one element, add one missing element.
    Accept only if the result is Sidon and maximal and size <= original.
    Returns the new list or None."""
    start = time.time()
    sel_set = set(selected)
    missing = [x for x in range(1, N + 1) if x not in sel_set]
    if not missing:
        return None
    rng.shuffle(missing)
    idxs = list(range(len(selected)))
    rng.shuffle(idxs)
    for i in idxs:
        if time.time() - start > time_budget:
            return None
        for x in missing:
            if time.time() - start > time_budget:
                return None
            new_sel = selected[:i] + [x] + selected[i + 1:]
            if len(new_sel) < len(selected):
                # size decreased, must be Sidon and maximal
                if _is_sidon(new_sel) and _is_maximal(new_sel, N):
                    return new_sel
            elif len(new_sel) == len(selected):
                # same size, still worth trying for a better configuration
                if _is_sidon(new_sel) and _is_maximal(new_sel, N):
                    return new_sel
    return None


def _minimize(selected, N, deadline, rng):
    """Iteratively remove elements and try swaps to minimize size."""
    start = time.time()
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        # Try removals
        new_sel = _try_remove_any(selected, N, rng)
        if new_sel is not None:
            selected = new_sel
            improved = True
        # Try swaps (with a small time budget per call)
        remaining = deadline - time.time()
        if remaining > 1.0:
            swap_budget = min(remaining - 0.5, 2.0)
            swapped = _try_swap(selected, N, rng, swap_budget)
            if swapped is not None and len(swapped) <= len(selected):
                selected = swapped
                improved = True
    return selected


def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

    best = None
    best_size = float('inf')
    rng = random.Random(42)

    # --- Phase 1: try a few deterministic orderings ---
    asc = list(range(1, N + 1))
    desc = list(range(N, 0, -1))
    for order in (asc, desc):
        sel = _greedy_construct(order)
        if len(sel) < best_size:
            best = sel
            best_size = len(sel)

    # --- Phase 2: randomized greedy constructions followed by minimization ---
    num_trials = 300
    for trial in range(num_trials):
        if time.time() > deadline - 3.0:
            break

        # Build a random order with a slight bias towards large numbers
        # (large numbers tend to block more sums, yielding smaller maximal sets)
        order = list(range(1, N + 1))
        # Mix: 50% uniform shuffle, 30% descending-biased, 20% ascending-biased
        r = rng.random()
        if r < 0.5:
            rng.shuffle(order)
        elif r < 0.8:
            # Sort by a key that favours larger values
            order.sort(key=lambda x: (rng.random() * 0.3 + (x / N) * 0.7, rng.random()))
            order.reverse()
        else:
            # Sort by a key that favours smaller values
            order.sort(key=lambda x: (rng.random() * 0.3 + (1 - x / N) * 0.7, rng.random()))

        sel = _greedy_construct(order)
        if len(sel) >= best_size:
            continue

        # Minimize
        remaining = deadline - time.time() - 2.0
        if remaining > 1.0:
            sel = _minimize(sel, N, deadline - 1.0, rng)

        if len(sel) < best_size:
            best = sel
            best_size = len(sel)

    # --- Phase 3: final polish on the best solution ---
    if best is not None and time.time() < deadline - 1.0:
        remaining = deadline - time.time() - 0.5
        if remaining > 0.5:
            polished = _minimize(best, N, deadline - 0.5, rng)
            if len(polished) < best_size:
                best = polished
                best_size = len(best)

    # --- Fallback ---
    if best is None:
        best = _greedy_construct(asc)

    # --- Final validation and maximality fix ---
    if not _is_sidon(best):
        best = _greedy_construct(asc)

    if not _is_maximal(best, N):
        sel_set = set(best)
        sums = set()
        n = len(best)
        for i in range(n):
            a = best[i]
            for j in range(i, n):
                sums.add(a + best[j])
        for x in range(1, N + 1):
            if x in sel_set:
                continue
            added = [x + a for a in best] + [2 * x]
            if len(set(added)) == len(added) and not (sums & set(added)):
                best.append(x)
                sel_set.add(x)
                sums.update(added)
        best.sort()

    return best
