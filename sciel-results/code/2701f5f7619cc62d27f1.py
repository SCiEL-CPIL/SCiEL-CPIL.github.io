"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START

def _has_200_bitset(selected):
    """Check if any subset of selected sums to 200 using bitset DP."""
    bits = 1  # sum 0 reachable
    for x in selected:
        bits |= (bits << x)
        # Mask to only keep bits 0..200
        bits &= (1 << 201) - 1
        if (bits >> 200) & 1:
            return True
    return False

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

    # Strategy 3: Known valid set [66, 100]
    sel = list(range(66, 101))
    if not _has_200_bitset(sel):
        try_set(sel)

    # Strategy 4: Random permutations
    rng = random.Random(42)
    # Increase number of trials if time allows
    max_trials = 2000
    for trial in range(max_trials):
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
                
    # Local search: try swapping (remove one, add another)
    # This helps escape local optima where adding is blocked but swapping is not
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try to swap each element in best with each element in excluded
        # To save time, limit swaps per iteration or break early
        for i in range(len(best)):
            if time.time() > deadline:
                break
            for x in excluded:
                if time.time() > deadline:
                    break
                # Remove best[i], add x
                cand = best[:i] + best[i+1:] + [x]
                if not _has_200_bitset(cand):
                    # If size is same, we only accept if it allows future additions?
                    # Actually, swap keeps size same. 
                    # We only accept swap if it leads to a state where we can add more.
                    # For now, just accept valid swap to change the structure.
                    # But we need to track the new excluded set.
                    best = sorted(cand)
                    best_set = set(best)
                    excluded = [y for y in range(1, 101) if y not in best_set]
                    improved = True
                    break # Restart inner loop with new excluded
            if improved:
                break
    return sorted(best)

# EVOLVE-BLOCK-END
