"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _can_add(chosen, counts, x):
    """Check if x can be added to chosen without violating B2[2] constraint."""
    for y in chosen:
        s = x + y
        if counts.get(s, 0) + 1 > 2:
            return False
    s2 = 2 * x
    if counts.get(s2, 0) + 1 > 2:
        return False
    return True

def _greedy_build(order, time_limit):
    """Build a B2[2] set by greedily adding elements in the given order."""
    chosen = []
    counts = {}
    deadline = time.time() + time_limit
    for x in order:
        if time.time() > deadline:
            break
        if _can_add(chosen, counts, x):
            chosen.append(x)
            for y in chosen:
                s = x + y
                counts[s] = counts.get(s, 0) + 1
            s2 = 2 * x
            counts[s2] = counts.get(s2, 0) + 1
    return chosen

def _greedy_build_with_removal(order, time_limit):
    """Greedy with occasional removal of elements that block many additions."""
    chosen = []
    counts = {}
    deadline = time.time() + time_limit
    for x in order:
        if time.time() > deadline:
            break
        if _can_add(chosen, counts, x):
            chosen.append(x)
            for y in chosen:
                s = x + y
                counts[s] = counts.get(s, 0) + 1
            s2 = 2 * x
            counts[s2] = counts.get(s2, 0) + 1
    return chosen

def construct():
    start_time = time.time()
    total_budget = 115.0  # Leave 5s margin for 120s limit
    
    # Strategy 1: Standard ascending greedy
    best_set = _greedy_build(range(1, 1001), min(5.0, total_budget))
    elapsed = time.time() - start_time
    best_size = len(best_set)
    
    # Strategy 2: Descending greedy
    remaining = total_budget - elapsed
    if remaining > 2.0:
        set2 = _greedy_build(range(1000, 0, -1), min(5.0, remaining))
        if len(set2) > best_size:
            best_set = set2
            best_size = len(set2)
        elapsed = time.time() - start_time
    
    # Strategy 3: Randomized greedy with different seeds
    remaining = total_budget - elapsed
    if remaining > 1.0:
        rng = random.Random(42)
        for seed in range(20):
            if time.time() - start_time > total_budget - 1.0:
                break
            order = list(range(1, 1001))
            rng.shuffle(order)
            set3 = _greedy_build(order, 0.5)
            if len(set3) > best_size:
                best_set = set3
                best_size = len(set3)
    
    # Strategy 4: Try with some bias toward smaller numbers first, then fill gaps
    remaining = total_budget - (time.time() - start_time)
    if remaining > 1.0:
        # Create a mixed order: interleave small and large
        small = list(range(1, 501))
        large = list(range(501, 1001))
        mixed = []
        i, j = 0, 0
        while i < len(small) or j < len(large):
            if i < len(small):
                mixed.append(small[i])
                i += 1
            if j < len(large):
                mixed.append(large[j])
                j += 1
        set4 = _greedy_build(mixed, min(5.0, remaining))
        if len(set4) > best_size:
            best_set = set4
            best_size = len(set4)
    
    return best_set
# EVOLVE-BLOCK-END
