"""Improved program: backtracking + randomized greedy with time budget."""
import time
import random

def _valid_append(values, x, sums):
    """Check if appending x keeps all consecutive-block sums distinct.
    New sums are the consecutive blocks ending at the new element x.
    Returns (new_tails, ok) where new_tails are the new sums to add."""
    tails = []
    total = x
    tails.append(total)
    for a in reversed(values):
        total += a
        tails.append(total)
    if sums.intersection(tails):
        return None, False
    return tails, True

def _greedy_randomized(start_seed, values_init=None, sums_init=None):
    """Greedy: pick next element from a shuffled candidate list."""
    values = list(values_init) if values_init is not None else []
    sums = set(sums_init) if sums_init is not None else set()
    rng = random.Random(start_seed)
    # Candidate pool: all numbers 1..256 not yet used, in random order but we re-shuffle each step
    # Actually, to be efficient, we try candidates in a randomized order each step.
    used = set(values)
    while True:
        candidates = [x for x in range(1, 257) if x not in used]
        rng.shuffle(candidates)
        found = False
        for x in candidates:
            if x <= (values[-1] if values else 0):
                continue
            tails, ok = _valid_append(values, x, sums)
            if ok:
                values.append(x)
                sums.update(tails)
                used.add(x)
                found = True
                break
        if not found:
            break
    return values

def _greedy_sequential():
    """Original greedy: smallest valid x first."""
    values = []
    sums = set()
    for x in range(1, 257):
        if values and x <= values[-1]:
            continue
        tails, ok = _valid_append(values, x, sums)
        if ok:
            values.append(x)
            sums.update(tails)
    return values

def _backtrack_limited(depth_limit, time_limit, seed):
    """Backtracking search with depth limit and time limit.
    Tries to build a sequence by choosing elements, backtracking on conflicts."""
    rng = random.Random(seed)
    best = []
    start_time = time.time()
    
    def _search(values, sums, depth):
        nonlocal best
        if time.time() - start_time > time_limit:
            return
        if len(values) > len(best):
            best = list(values)
        if depth >= depth_limit:
            return
        # Generate candidates: numbers greater than last, not used
        last = values[-1] if values else 0
        used = set(values)
        candidates = [x for x in range(last + 1, 257) if x not in used]
        # Sort by heuristic: prefer smaller numbers (denser packing) but with some randomness
        rng.shuffle(candidates)
        candidates.sort()  # Try smaller first for density
        for x in candidates:
            tails, ok = _valid_append(values, x, sums)
            if ok:
                values.append(x)
                sums.update(tails)
                _search(values, sums, depth + 1)
                values.pop()
                for t in tails:
                    sums.discard(t)
            if time.time() - start_time > time_limit:
                return
    
    _search([], set(), 0)
    return best

def construct():
    deadline = time.time() + 85  # 85s safety margin within 90s
    
    # Phase 1: Quick greedy baseline
    best = _greedy_sequential()
    
    # Phase 2: Randomized greedy with many seeds
    for seed in range(200):
        if time.time() > deadline:
            break
        result = _greedy_randomized(seed)
        if len(result) > len(best):
            best = result
    
    # Phase 3: Limited backtracking with increasing depth limits
    # Use remaining time for backtracking
    if time.time() < deadline - 5:
        # Try backtracking with moderate depth
        for depth in [18, 20, 22, 24, 26]:
            if time.time() > deadline - 5:
                break
            remaining = deadline - 5 - time.time()
            if remaining < 2:
                break
            result = _backtrack_limited(depth, remaining, seed=42)
            if len(result) > len(best):
                best = result
    
    return best
