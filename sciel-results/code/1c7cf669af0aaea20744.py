"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

_DEADLINE_SECONDS = 85  # safety margin before 90s kill

def _precompute_gcd_table(n):
    """Precompute gcd for all pairs in [1..n]."""
    table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            g = gcd(i, j)
            table[i][j] = g
            table[j][i] = g
    return table

def _can_add_fast(x, selected, table):
    """Check if adding x to selected keeps the set valid.
    selected is a list. table is the GCD table.
    Returns True if valid.
    """
    n = len(selected)
    # For each pair (a, b) in selected, check if gcd(a,b) == gcd(a,x) == gcd(b,x)
    # Optimization: Iterate a, then b.
    for i in range(n):
        a = selected[i]
        g_ax = table[a][x]
        for j in range(i + 1, n):
            b = selected[j]
            g_ab = table[a][b]
            if g_ab == g_ax:
                if table[b][x] == g_ax:
                    return False
    return True

def _greedy_pass(order, table):
    """Greedy pass: add elements in given order if valid."""
    selected = []
    for x in order:
        if _can_add_fast(x, selected, table):
            selected.append(x)
    return selected

def _local_search(selected, all_nums, table, deadline, rng):
    """Local search with add, remove, and swap moves."""
    current = list(selected)
    in_set = set(current)
    excluded = [c for c in all_nums if c not in in_set]
    best = len(current)
    
    # Phase 1: Try to add as many as possible (Greedy Add)
    while True:
        if time.time() > deadline:
            break
        added_any = False
        # Shuffle excluded to randomize
        rng.shuffle(excluded)
        for x in excluded:
            if time.time() > deadline:
                break
            if _can_add_fast(x, current, table):
                current.append(x)
                in_set.add(x)
                excluded.remove(x)
                added_any = True
                if len(current) > best:
                    best = len(current)
        if not added_any:
            break
            
    # Phase 2: Random Walk (Remove/Add)
    # We try to escape local optima by removing elements and then re-adding
    max_steps = 500
    for step in range(max_steps):
        if time.time() > deadline:
            break
        
        # With some probability, try a swap (remove 1, add 1)
        # Or just remove 1 to allow future adds
        if rng.random() < 0.5 and len(current) > 1:
            # Remove a random element
            idx = rng.randint(0, len(current) - 1)
            rem = current.pop(idx)
            in_set.remove(rem)
            excluded.append(rem)
            
            # Try to add random excluded elements
            rng.shuffle(excluded)
            for x in excluded:
                if time.time() > deadline:
                    break
                if _can_add_fast(x, current, table):
                    current.append(x)
                    in_set.add(x)
                    excluded.remove(x)
                    break # Only add one per removal step to keep it simple
            
        else:
            # Try a swap: remove a, add b
            if not excluded:
                continue
            # Pick a random element to remove
            if len(current) == 0:
                break
            idx = rng.randint(0, len(current) - 1)
            a = current[idx]
            # Pick a random element to add
            b_idx = rng.randint(0, len(excluded) - 1)
            b = excluded[b_idx]
            
            # Check if swapping a for b is valid
            # Remove a temporarily
            current.pop(idx)
            # Check if b can be added
            if _can_add_fast(b, current, table):
                current.append(b)
                in_set.remove(a)
                in_set.add(b)
                excluded.remove(b)
                excluded.append(a)
            else:
                # Revert
                current.insert(idx, a)

        if len(current) > best:
            best = len(current)
            
    return current

def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE_SECONDS
    
    all_nums = list(range(1, 129))
    
    # Precompute gcd table
    table = _precompute_gcd_table(128)
    
    best_set = []
    best_len = 0
    
    # Strategy 1: Sequential greedy
    if time.time() < deadline:
        s = _greedy_pass(all_nums, table)
        if len(s) > best_len:
            best_set, best_len = s, len(s)
            
    # Strategy 2: Reverse order greedy
    if time.time() < deadline:
        s = _greedy_pass(list(reversed(all_nums)), table)
        if len(s) > best_len:
            best_set, best_len = s, len(s)

    # Strategy 3: Many random greedy passes with local search
    rng = random.Random(42)
    num_random_passes = 100 # Reduced count, but local search is heavier
    
    for i in range(num_random_passes):
        if time.time() > deadline:
            break
        
        # Random order
        order = all_nums[:]
        rng.shuffle(order)
        s = _greedy_pass(order, table)
        
        # Local search on this result
        if time.time() < deadline:
            improved = _local_search(s, all_nums, table, deadline, rng)
            if len(improved) > best_len:
                best_set, best_len = improved, len(improved)
        
        # Also try local search on the global best if it hasn't changed much?
        # Or just restart from best?
        if len(s) > best_len:
             best_set, best_len = s, len(s)

    # Final polish on the best set found
    if time.time() < deadline:
        final_improved = _local_search(best_set, all_nums, table, deadline, rng)
        if len(final_improved) > best_len:
            best_set, best_len = final_improved, len(final_improved)

    # Ensure nonempty
    if not best_set:
        best_set = [1]
    
    return best_set
# EVOLVE-BLOCK-END
