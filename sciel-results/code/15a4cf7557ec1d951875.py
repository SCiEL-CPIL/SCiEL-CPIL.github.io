"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def _check_divisibility(sums_set):
    """Check no two distinct values in sums_set divide one another."""
    ss = sorted(sums_set)
    n = len(ss)
    for i in range(n):
        si = ss[i]
        for j in range(i + 1, n):
            sj = ss[j]
            if sj % si == 0:
                return False
    return True

def _can_add(current_sums, x):
    """Check if adding x to a set with subset sums current_sums maintains validity.
    Returns new sums set if valid, None otherwise."""
    # New subset sums: old sums, old sums + x, and x itself
    new_sums = set(current_sums)
    new_sums.add(x)
    for s in current_sums:
        new_sums.add(s + x)
    # Check divisibility
    if not _check_divisibility(new_sums):
        return None
    return new_sums

def construct():
    start = time.time()
    deadline = start + 85  # 85 second deadline, 5s safety margin
    
    best = []
    best_size = 0
    
    def try_greedy(candidates):
        """Try greedy construction with given candidate list."""
        nonlocal best, best_size
        current = []
        current_sums = set()
        for cand in candidates:
            if time.time() > deadline:
                break
            if len(current) >= 30:
                break
            new_sums = _can_add(current_sums, cand)
            if new_sums is not None:
                current.append(cand)
                current_sums = new_sums
                if len(current) > best_size:
                    best_size = len(current)
                    best = list(current)
    
    # Strategy 1: Try large numbers first (256 down to 1)
    try_greedy(range(256, 0, -1))
    
    # Strategy 2: Try medium-large numbers
    if time.time() < deadline:
        try_greedy(range(200, 100, -1))
    
    # Strategy 3: Try numbers in [128, 256]
    if time.time() < deadline:
        try_greedy(range(256, 127, -1))
    
    # Strategy 4: Randomized restarts with different orderings
    import random
    rng = random.Random(42)
    for trial in range(50):
        if time.time() > deadline:
            break
        # Mix of strategies: random subset of range
        if trial % 3 == 0:
            cands = list(range(256, 64, -1))
        elif trial % 3 == 1:
            cands = list(range(256, 1, -1))
            rng.shuffle(cands)
        else:
            cands = list(range(256, 32, -1))
            rng.shuffle(cands)
        try_greedy(cands)
    
    # Strategy 5: Try to extend best found set by adding more elements
    if time.time() < deadline and best:
        current = list(best)
        current_sums = set()
        # Rebuild sums
        for mask in range(1, 1 << len(current)):
            s = 0
            for i in range(len(current)):
                if mask & (1 << i):
                    s += current[i]
            current_sums.add(s)
        
        # Try adding each remaining number
        for cand in range(256, 0, -1):
            if time.time() > deadline:
                break
            if cand in current:
                continue
            if len(current) >= 30:
                break
            new_sums = _can_add(current_sums, cand)
            if new_sums is not None:
                current.append(cand)
                current_sums = new_sums
                if len(current) > best_size:
                    best_size = len(current)
                    best = list(current)
    
    # Final verification
    if best:
        n = len(best)
        sums = set()
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += best[i]
            sums.add(s)
        if not _check_divisibility(sums):
            # Fallback to a known small valid set
            best = [31, 32, 34]
    
    return list(best)
# EVOLVE-BLOCK-END
