"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
import random

def _has_200_bitset(selected):
    """Check if any subset of selected sums to 200 using bitset DP."""
    # bit i set means sum i is reachable
    bits = 1  # sum 0 reachable
    for x in selected:
        bits |= (bits << x)
        # Mask to only keep bits 0..200
        bits &= (1 << 201) - 1
        if (bits >> 200) & 1:
            return True
    return False

def _reachable_bitset(selected):
    """Compute reachable sums as a bitset."""
    bits = 1
    for x in selected:
        bits |= (bits << x)
        bits &= (1 << 201) - 1
    return bits

def construct():
    start = time.time()
    deadline = start + 50.0  # safety margin under 90s

    best = None
    best_size = -1

    def try_set(s):
        nonlocal best, best_size
        if not s:
            return
        if not _has_200_bitset(s) and len(s) > best_size:
            best = list(s)
            best_size = len(s)

    # Strategy 1: Greedy descending
    sel = []
    reach = 1
    for x in range(100, 0, -1):
        new_bits = reach | (reach << x)
        new_bits &= (1 << 201) - 1
        if not (new_bits >> 200) & 1:
            sel.append(x)
            reach = new_bits
    try_set(sel)

    # Strategy 2: Greedy ascending
    sel = []
    reach = 1
    for x in range(1, 101):
        new_bits = reach | (reach << x)
        new_bits &= (1 << 201) - 1
        if not (new_bits >> 200) & 1:
            sel.append(x)
            reach = new_bits
    try_set(sel)

    # Strategy 3: Greedy by even/odd split - take all odds (no two odds sum to 200 if both < 200)
    # Actually odds + odds = even, so two odds can sum to 200. But let's try various heuristics.
    
    # Strategy 4: Take numbers > 100? None exist. Take numbers in [101,100]? Empty.
    # Try: all numbers from 51 to 100 (50 numbers) - check if valid
    sel = list(range(51, 101))
    if not _has_200_bitset(sel):
        try_set(sel)
    
    # Strategy 5: Random permutations (fewer, since bitset is fast)
    rng = random.Random(42)
    for trial in range(500):
        if time.time() > deadline:
            break
        order = list(range(1, 101))
        rng.shuffle(order)
        sel = []
        reach = 1
        for x in order:
            new_bits = reach | (reach << x)
            new_bits &= (1 << 201) - 1
            if not (new_bits >> 200) & 1:
                sel.append(x)
                reach = new_bits
        try_set(sel)

    if best is None:
        best = [100]

    # Local search: try adding back excluded elements
    best = sorted(best)
    best_set = set(best)
    excluded = [x for x in range(1, 101) if x not in best_set]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in excluded[:]:
            if time.time() > deadline:
                break
            if x in best_set:
                continue
            cand = best + [x]
            if not _has_200_bitset(cand):
                best = sorted(cand)
                best_set = set(best)
                if x in excluded:
                    excluded.remove(x)
                improved = True

    return sorted(best)

# EVOLVE-BLOCK-END
