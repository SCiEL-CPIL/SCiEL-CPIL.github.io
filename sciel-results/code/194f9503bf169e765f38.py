"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
import time

_TARGET = 200
_UNIVERSE = list(range(1, 101))

def _reachable_sums(selected, target=_TARGET):
    """Return the set of all subset sums of `selected` that are <= target."""
    reachable = {0}
    for x in selected:
        # add x to each existing sum, keeping only those <= target
        new = set()
        for s in reachable:
            ns = s + x
            if ns <= target:
                new.add(ns)
        reachable |= new
    return reachable

def _has_sum(selected, target=_TARGET):
    """Check whether any subset of `selected` sums exactly to `target`."""
    reachable = {0}
    for x in selected:
        new = set()
        for s in reachable:
            ns = s + x
            if ns == target:
                return True
            if ns < target:
                new.add(ns)
        reachable |= new
    return False

def _greedy_descending():
    """Greedy: process from largest to smallest, keep if it does not create a sum of 200."""
    selected = []
    reachable = {0}
    for x in range(100, 0, -1):
        new = {s + x for s in reachable if s + x <= _TARGET}
        if _TARGET not in new:
            selected.append(x)
            reachable |= new
    return selected

def _local_improve(selected):
    """Try to add back excluded elements one at a time if safe. Repeat until no change."""
    selected_set = set(selected)
    excluded = [x for x in _UNIVERSE if x not in selected_set]
    improved = True
    while improved:
        improved = False
        for x in excluded:
            if x in selected_set:
                continue
            candidate = selected_set | {x}
            if not _has_sum(list(candidate)):
                selected_set.add(x)
                improved = True
    return list(selected_set)

def _try_orders():
    """Try multiple construction orders and keep the best (largest) valid set."""
    best = None
    orders = [
        list(range(100, 0, -1)),   # descending
        list(range(1, 101)),       # ascending
        # a heuristic: large first then fill small
    ]
    for order in orders:
        selected = []
        reachable = {0}
        for x in order:
            new = {s + x for s in reachable if s + x <= _TARGET}
            if _TARGET not in new:
                selected.append(x)
                reachable |= new
        # local improvement
        sel = _local_improve(selected)
        if best is None or len(sel) > len(best):
            best = sel
    return best

def construct():
    deadline = time.time() + 80.0  # safety margin before 90s kill
    best = _greedy_descending()
    best = _local_improve(best)

    # Try to beat the greedy with alternative orders if we have time
    if time.time() < deadline:
        cand = _try_orders()
        if cand is not None and len(cand) > len(best):
            best = cand

    # Final safety: ensure validity; if somehow invalid, fall back to a known-valid set
    if _has_sum(best):
        # Fallback: all numbers > 100 are absent; use a safe large set.
        # Numbers 101..200 unavailable, so use a verified safe subset.
        # A simple safe set: all numbers in [101,200] empty -> use greedy result re-derived.
        best = _greedy_descending()
        best = _local_improve(best)

    if not best:
        best = [100]

    return best
# EVOLVE-BLOCK-END
