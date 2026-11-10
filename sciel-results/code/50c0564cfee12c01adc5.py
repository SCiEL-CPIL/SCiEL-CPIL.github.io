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

def _is_valid_triple_fast(a, b, c, table):
    """Return True if the triple (a,b,c) does NOT have all three pairwise gcds equal."""
    g_ab = table[a][b]
    g_ac = table[a][c]
    g_bc = table[b][c]
    return not (g_ab == g_ac == g_bc)

def _can_add(x, selected, table):
    """Check if adding x to selected keeps the set valid."""
    for i in range(len(selected)):
        a = selected[i]
        for j in range(i + 1, len(selected)):
            b = selected[j]
            if not _is_valid_triple_fast(a, b, x, table):
                return False
    return True

def _greedy_pass(order, table):
    """Greedy pass: add elements in given order if valid."""
    selected = []
    for x in order:
        if _can_add(x, selected, table):
            selected.append(x)
    return selected

def _local_improve_add(selected, candidates, table, deadline):
    """Try to add excluded candidates."""
    improved = list(selected)
    in_set = set(improved)
    to_try = [c for c in candidates if c not in in_set]
    random.shuffle(to_try)
    for x in to_try:
        if time.time() > deadline:
            break
        if _can_add(x, improved, table):
            improved.append(x)
    return improved

def _local_improve_swap(selected, candidates, table, deadline, max_swaps=200):
    """Try swap-based local search: remove one, add one or more."""
    improved = list(selected)
    in_set = set(improved)
    excluded = [c for c in candidates if c not in in_set]
    
    for _ in range(max_swaps):
        if time.time() > deadline:
            break
        if not excluded:
            break
        
        # Pick a random element to remove
        idx = random.randint(0, len(improved) - 1)
        removed = improved[idx]
        improved.pop(idx)
        
        # Try to add multiple excluded elements
        added_any = False
        remaining = [e for e in excluded if e not in improved]
        random.shuffle(remaining)
        for x in remaining:
            if time.time() > deadline:
                break
            if _can_add(x, improved, table):
                improved.append(x)
                added_any = True
        
        if not added_any or len(improved) < len(selected):
            # Revert
            improved.insert(idx, removed)
        else:
            # Update excluded list
            in_set = set(improved)
            excluded = [c for c in candidates if c not in in_set]
    
    return improved

def _factor_count(n):
    """Count total prime factors with multiplicity."""
    cnt = 0
    d = 2
    while d * d <= n:
        while n % d == 0:
            cnt += 1
            n //= d
        d += 1
    if n > 1:
        cnt += 1
    return cnt

def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE_SECONDS
    
    all_nums = list(range(1, 129))
    
    # Precompute gcd table
    table = _precompute_gcd_table(128)
    
    # Precompute factor counts for heuristic ordering
    fc = {x: _factor_count(x) for x in all_nums}
    
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
    
    # Strategy 3: Prime-factor-rich first greedy (from inspiration 1)
    if time.time() < deadline:
        order = sorted(all_nums, key=lambda x: (-fc[x], x))
        s = _greedy_pass(order, table)
        if len(s) > best_len:
            best_set, best_len = s, len(s)
    
    # Strategy 4: Prime-factor-poor first greedy (complementary heuristic)
    if time.time() < deadline:
        order = sorted(all_nums, key=lambda x: (fc[x], x))
        s = _greedy_pass(order, table)
        if len(s) > best_len:
            best_set, best_len = s, len(s)
    
    # Strategy 5: Many random greedy passes with local improvement
    rng = random.Random(42)
    num_random_passes = 800
    for i in range(num_random_passes):
        if time.time() > deadline:
            break
        order = all_nums[:]
        rng.shuffle(order)
        s = _greedy_pass(order, table)
        if len(s) > best_len:
            best_set, best_len = s, len(s)
            if time.time() < deadline:
                improved = _local_improve_add(best_set, all_nums, table, deadline)
                if len(improved) > best_len:
                    best_set, best_len = improved, len(improved)
                if time.time() < deadline:
                    improved2 = _local_improve_swap(best_set, all_nums, table, deadline, max_swaps=50)
                    if len(improved2) > best_len:
                        best_set, best_len = improved2, len(improved2)
    
    # Final local improvement
    if time.time() < deadline:
        improved = _local_improve_add(best_set, all_nums, table, deadline)
        if len(improved) > best_len:
            best_set, best_len = improved, len(improved)
        if time.time() < deadline:
            improved2 = _local_improve_swap(best_set, all_nums, table, deadline, max_swaps=100)
            if len(improved2) > best_len:
                best_set, best_len = improved2, len(improved2)
    
    # Ensure nonempty
    if not best_set:
        best_set = [1]
    
    return best_set
# EVOLVE-BLOCK-END
