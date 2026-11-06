"""Construct a small inclusion-maximal Sidon set in [1,256]."""
import time
import random

N = 256

# EVOLVE-BLOCK-START

def _is_valid_sidon(selected):
    """Check if selected is a Sidon set (all unordered sums a+b, a<=b, distinct)."""
    sums = set()
    n = len(selected)
    for i in range(n):
        a = selected[i]
        for j in range(i, n):
            s = a + selected[j]
            if s in sums:
                return False
            sums.add(s)
    return True


def _is_maximal(selected, N):
    """Check if selected is inclusion-maximal: adding any missing element breaks Sidon."""
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
        # New sums introduced by x: 2x and x+a for a in selected
        if (2 * x) in sums:
            continue
        ok = True
        for a in selected:
            if (x + a) in sums:
                ok = False
                break
        if ok:
            return False
    return True


def _addable(x, selected, sums):
    """Can x be added to selected (with current sum set) without conflict?"""
    if (2 * x) in sums:
        return False
    for a in selected:
        if (x + a) in sums:
            return False
    return True


def _max_blocking_greedy(N, deadline, seed=None):
    """
    Greedy Max-Blocking: repeatedly add the element that blocks the most
    remaining candidates, tending to produce a small (possibly non-maximal)
    Sidon set. Returns the partial set (not necessarily maximal).
    """
    rng = random.Random(seed) if seed is not None else None
    remaining = list(range(1, N + 1))
    selected = []
    sums = set()
    while remaining:
        if time.time() > deadline:
            break
        # Valid candidates: those addable to current selected
        valid = [x for x in remaining if _addable(x, selected, sums)]
        if not valid:
            break
        # For each valid x, count how many other valid y would be blocked by adding x
        best_x = None
        best_block = -1
        for x in valid:
            new_sums = sums | {(2 * x)} | {(x + a) for a in selected}
            block = 0
            for y in valid:
                if y == x:
                    continue
                # Would y still be addable after x is in?
                if (2 * y) in new_sums:
                    block += 1
                    continue
                blocked = False
                for a in selected:
                    if (y + a) in new_sums:
                        blocked = True
                        break
                if not blocked and (x + y) in new_sums:
                    blocked = True
                if blocked:
                    block += 1
            if block > best_block:
                best_block = block
                best_x = x
        if best_x is None:
            break
        selected.append(best_x)
        remaining.remove(best_x)
        sums.add(2 * best_x)
        for a in selected[:-1]:
            sums.add(best_x + a)
    return selected


def _coverage_greedy_fill(selected, N, deadline, seed=None):
    """
    Complete a Sidon set to inclusion-maximal by greedily adding elements that
    cover (block) the most remaining candidates. Minimizes added count.
    """
    rng = random.Random(seed) if seed is not None else None
    selected = list(selected)
    sums = set()
    for i, a in enumerate(selected):
        for j in range(i, len(selected)):
            sums.add(a + selected[j])
    sel_set = set(selected)
    remaining = [x for x in range(1, N + 1) if x not in sel_set]
    while remaining:
        if time.time() > deadline:
            break
        valid = [x for x in remaining if _addable(x, selected, sums)]
        if not valid:
            break
        # Pick valid x covering the most remaining candidates
        best_x = None
        best_cov = -1
        for x in valid:
            new_sums = sums | {(2 * x)} | {(x + a) for a in selected}
            cov = 0
            for y in remaining:
                if y == x:
                    continue
                if (2 * y) in new_sums:
                    cov += 1
                    continue
                blocked = False
                for a in selected:
                    if (y + a) in new_sums:
                        blocked = True
                        break
                if not blocked and (x + y) in new_sums:
                    blocked = True
                if blocked:
                    cov += 1
            if cov > best_cov:
                best_cov = cov
                best_x = x
        if best_x is None:
            break
        selected.append(best_x)
        sel_set.add(best_x)
        remaining.remove(best_x)
        sums.add(2 * best_x)
        for a in selected[:-1]:
            sums.add(best_x + a)
    return selected


def _local_search_remove(selected, N, deadline):
    """
    Local search: repeatedly remove an element while the set stays Sidon AND
    maximal. Shrinks cardinality while preserving inclusion-maximality.
    """
    best = list(selected)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try removals; prefer removing elements that leave set maximal
        for x in list(best):
            if time.time() > deadline:
                break
            cand = [v for v in best if v != x]
            if not _is_valid_sidon(cand):
                continue
            if _is_maximal(cand, N):
                best = cand
                improved = True
                break
    return best


def _random_greedy_seed(N, deadline, seed):
    """Build a Sidon seed from a randomized order (max-blocking flavor)."""
    rng = random.Random(seed)
    remaining = list(range(1, N + 1))
    selected = []
    sums = set()
    while remaining:
        if time.time() > deadline:
            break
        valid = [x for x in remaining if _addable(x, selected, sums)]
        if not valid:
            break
        # Pick weighted by coverage, with randomness
        best_x = None
        best_cov = -1
        top = []
        for x in valid:
            new_sums = sums | {(2 * x)} | {(x + a) for a in selected}
            cov = 0
            for y in valid:
                if y == x:
                    continue
                if (2 * y) in new_sums:
                    cov += 1
                    continue
                blocked = False
                for a in selected:
                    if (y + a) in new_sums:
                        blocked = True
                        break
                if not blocked and (x + y) in new_sums:
                    blocked = True
                if blocked:
                    cov += 1
            if cov > best_cov:
                best_cov = cov
                top = [x]
            elif cov == best_cov:
                top.append(x)
        x = rng.choice(top) if top else rng.choice(valid)
        selected.append(x)
        remaining.remove(x)
        sums.add(2 * x)
        for a in selected[:-1]:
            sums.add(x + a)
    return selected


def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s kill

    best_result = None
    best_size = float('inf')

    def consider(selected):
        nonlocal best_result, best_size
        if not selected:
            return
        if not _is_valid_sidon(selected):
            return
        if not _is_maximal(selected, N):
            return
        if len(selected) < best_size:
            best_size = len(selected)
            best_result = list(selected)

    # Strategy A: max-blocking greedy -> complete to maximal -> local search shrink
    for seed in range(12):
        if time.time() > deadline:
            break
        partial = _max_blocking_greedy(N, deadline, seed if seed else None)
        if not partial:
            continue
        maximal = _coverage_greedy_fill(partial, N, deadline, seed)
        consider(maximal)
        shrunken = _local_search_remove(maximal, N, deadline)
        consider(shrunken)

    # Strategy B: randomized seeds -> complete -> shrink
    for seed in range(24):
        if time.time() > deadline:
            break
        partial = _random_greedy_seed(N, deadline, 1000 + seed)
        if not partial:
            continue
        maximal = _coverage_greedy_fill(partial, N, deadline, seed)
        consider(maximal)
        shrunken = _local_search_remove(maximal, N, deadline)
        consider(shrunken)

    # Fallback: ascending greedy, then make maximal
    if best_result is None:
        selected, sums = [], set()
        for x in range(1, N + 1):
            if _addable(x, selected, sums):
                selected.append(x)
                sums.add(2 * x)
                for a in selected[:-1]:
                    sums.add(x + a)
        maximal = _coverage_greedy_fill(selected, N, deadline, 0)
        consider(maximal)
        if best_result is None:
            return maximal

    return best_result

# EVOLVE-BLOCK-END
