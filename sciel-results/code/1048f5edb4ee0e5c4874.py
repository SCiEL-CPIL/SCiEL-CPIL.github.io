"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def _get_block_sums(values):
    """Get all consecutive block sums for a sequence."""
    sums = set()
    n = len(values)
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += values[j]
            sums.add(s)
    return sums

def _can_append(values, x, sums):
    """Check if x can be appended to values without creating duplicate block sums."""
    new_sums = set()
    total = 0
    for a in reversed(values):
        total += a
        new_sums.add(total)
    new_sums.add(x)  # Wait, this is wrong. Let me fix.
    # Actually:
    new_sums = set()
    total = x
    new_sums.add(total)
    for a in reversed(values):
        total += a
        new_sums.add(total)
    return not (new_sums & sums)

def _greedy_randomized(start=1, end=257, seed=None):
    """Randomized greedy: at each step, try random candidates."""
    if seed is not None:
        random.seed(seed)
    values = []
    sums = set()
    candidates = list(range(start, end))
    while candidates:
        # Try candidates in random order
        random.shuffle(candidates)
        added = False
        for x in candidates:
            if x <= (values[-1] if values else 0):
                continue
            new_sums = set()
            total = x
            new_sums.add(total)
            for a in reversed(values):
                total += a
                new_sums.add(total)
            if not (new_sums & sums):
                values.append(x)
                sums.update(new_sums)
                candidates.remove(x)
                added = True
                break
        if not added:
            break
    return values

def _backtrack(values, sums, min_val, max_val, depth, deadline):
    """Backtracking search with pruning."""
    if time.time() > deadline:
        return values
    if depth >= 15:  # Limit depth to avoid explosion
        return values
    
    # Try to extend
    best = list(values)
    for x in range(min_val, max_val + 1):
        if x <= (values[-1] if values else 0):
            continue
        new_sums = set()
        total = x
        new_sums.add(total)
        for a in reversed(values):
            total += a
            new_sums.add(total)
        if not (new_sums & sums):
            new_sums_set = sums | new_sums
            result = _backtrack(values + [x], new_sums_set, x + 1, max_val, depth + 1, deadline)
            if len(result) > len(best):
                best = result
            if time.time() > deadline:
                return best
    return best

def construct():
    deadline = time.time() + 85.0  # 85 second deadline (5s safety margin)
    
    # First, run the original greedy to get a baseline
    values_greedy, sums_greedy = [], set()
    for x in range(1, 257):
        new_sums = set()
        total = x
        new_sums.add(total)
        for a in reversed(values_greedy):
            total += a
            new_sums.add(total)
        if not (new_sums & sums_greedy):
            values_greedy.append(x)
            sums_greedy.update(new_sums)
    
    best = values_greedy
    best_len = len(best)
    
    # Try randomized greedy with different seeds
    for seed in range(50):
        if time.time() > deadline:
            break
        result = _greedy_randomized(seed=seed)
        if len(result) > best_len:
            best = result
            best_len = len(result)
    
    # Try backtracking from the greedy result, trying to improve
    if time.time() < deadline:
        # Try to extend the greedy result with backtracking
        vals = list(best)
        sums = _get_block_sums(vals)
        min_val = vals[-1] + 1 if vals else 1
        result = _backtrack(vals, sums, min_val, 256, 0, deadline)
        if len(result) > best_len:
            best = result
            best_len = len(result)
    
    return best
# EVOLVE-BLOCK-END
