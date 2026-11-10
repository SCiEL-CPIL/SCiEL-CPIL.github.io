"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from math import gcd

_DEADLINE_SECONDS = 80  # safety margin before 90s kill

# Precompute GCD table for 1..128
_MAX_VAL = 128
_gcd_table = [[0] * (_MAX_VAL + 1) for _ in range(_MAX_VAL + 1)]
for _i in range(1, _MAX_VAL + 1):
    for _j in range(1, _MAX_VAL + 1):
        _gcd_table[_i][_j] = gcd(_i, _j)

def _is_valid_triple(a, b, c):
    """Return True if the triple (a,b,c) does NOT have all three pairwise gcds equal."""
    g_ab = _gcd_table[a][b]
    g_ac = _gcd_table[a][c]
    g_bc = _gcd_table[b][c]
    return not (g_ab == g_ac == g_bc)

def _can_add(x, selected):
    """Check if adding x to selected keeps the set valid (no bad triple formed)."""
    for i, a in enumerate(selected):
        for b in selected[i+1:]:
            g_ab = _gcd_table[a][b]
            g_ax = _gcd_table[a][x]
            g_bx = _gcd_table[b][x]
            if g_ab == g_ax == g_bx:
                return False
    return True

def _greedy_pass(order):
    """Greedy pass: add elements in given order if valid."""
    selected = []
    for x in order:
        if _can_add(x, selected):
            selected.append(x)
    return selected

def _local_improve_add(selected, candidates, deadline):
    """Try to add back excluded candidates in random order."""
    improved = list(selected)
    in_set = set(improved)
    to_try = [c for c in candidates if c not in in_set]
    random.shuffle(to_try)
    for x in to_try:
        if time.time() > deadline:
            break
        if _can_add(x, improved):
            improved.append(x)
    return improved

def _local_improve_swap(selected, candidates, deadline):
    """Try swap operations: remove one element, then try to add multiple others."""
    improved = list(selected)
    in_set = set(improved)
    to_try = [c for c in candidates if c not in in_set]
    random.shuffle(to_try)
    
    for _ in range(len(improved)):
        if time.time() > deadline:
            break
        # Pick a random element to remove
        rem_idx = random.randint(0, len(improved) - 1)
        removed = improved[rem_idx]
        improved.pop(rem_idx)
        in_set.discard(removed)
        
        # Try to add multiple candidates
        added_any = False
        for x in to_try:
            if time.time() > deadline:
                break
            if x in in_set:
                continue
            if _can_add(x, improved):
                improved.append(x)
                in_set.add(x)
                added_any = True
        
        if not added_any:
            # Put it back
            improved.append(removed)
            in_set.add(removed)
    
    return improved

def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE_SECONDS
    
    # Precompute all numbers 1..128
    all_nums = list(range(1, 129))
    
    # Helper: count total prime factors (with multiplicity) for ordering heuristic
    def factor_count(n):
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
    
    fc = {x: factor_count(x) for x in all_nums}
    
    best_set = []
    best_len = 0
    
    # Strategy 1: Sequential greedy (original)
    if time.time() < deadline:
        s = _greedy_pass(all_nums)
        if len(s) > best_len:
            best_set, best_len = s, len(s)
    
    # Strategy 2: Prime-factor-rich first greedy
    if time.time() < deadline:
        order = sorted(all_nums, key=lambda x: (-fc[x], x))
        s = _greedy_pass(order)
        if len(s) > best_len:
            best_set, best_len = s, len(s)
    
    # Strategy 3: Many random shuffle greedy passes
    rng = random.Random(42)
    num_random_passes = 500
    for _ in range(num_random_passes):
        if time.time() > deadline:
            break
        order = all_nums[:]
        rng.shuffle(order)
        s = _greedy_pass(order)
        if len(s) > best_len:
            best_set, best_len = s, len(s)
            # Try local improvement on the new best
            if time.time() < deadline:
                improved = _local_improve_add(best_set, all_nums, deadline)
                if len(improved) > best_len:
                    best_set, best_len = improved, len(improved)
                if time.time() < deadline:
                    improved = _local_improve_swap(best_set, all_nums, deadline)
                    if len(improved) > best_len:
                        best_set, best_len = improved, len(improved)
    
    # Final local improvement on best set
    if time.time() < deadline:
        improved = _local_improve_add(best_set, all_nums, deadline)
        if len(improved) > best_len:
            best_set, best_len = improved, len(improved)
    if time.time() < deadline:
        improved = _local_improve_swap(best_set, all_nums, deadline)
        if len(improved) > best_len:
            best_set, best_len = improved, len(improved)
    
    # Ensure nonempty
    if not best_set:
        best_set = [1]
    
    return best_set
# EVOLVE-BLOCK-END
