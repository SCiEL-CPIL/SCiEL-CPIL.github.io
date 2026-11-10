"""Rewritten program for large integer sets avoiding equal-gcd triples."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from itertools import combinations
from collections import defaultdict

# Precompute gcd matrix for 1..128
_N = 128
_gcd_table = [[0] * (_N + 1) for _ in range(_N + 1)]
for _i in range(1, _N + 1):
    for _j in range(_i, _N + 1):
        _g = gcd(_i, _j)
        _gcd_table[_i][_j] = _g
        _gcd_table[_j][_i] = _g

def _is_bad_triple(a, b, c):
    """Check if three distinct elements form a bad triple (all pairwise gcds equal)."""
    g_ab = _gcd_table[a][b]
    g_ac = _gcd_table[a][c]
    if g_ab != g_ac:
        return False
    return _gcd_table[b][c] == g_ab

def _conflict_count(x, selected):
    """Count how many pairs in selected form a bad triple with x."""
    cnt = 0
    for i in range(len(selected)):
        a = selected[i]
        for j in range(i + 1, len(selected)):
            b = selected[j]
            if _is_bad_triple(a, b, x):
                cnt += 1
    return cnt

def _is_valid_set(selected):
    """Check if the set has no bad triples."""
    for i in range(len(selected)):
        a = selected[i]
        for j in range(i + 1, len(selected)):
            b = selected[j]
            for k in range(j + 1, len(selected)):
                c = selected[k]
                if _is_bad_triple(a, b, c):
                    return False
    return True

def _greedy_add_order():
    """Greedy construction: add elements in order of minimal conflict."""
    selected = []
    remaining = list(range(1, _N + 1))
    while remaining:
        # Find element with minimum conflicts
        best_elem = None
        best_conf = float('inf')
        for x in remaining:
            conf = _conflict_count(x, selected)
            if conf < best_conf:
                best_conf = conf
                best_elem = x
        if best_elem is None:
            break
        selected.append(best_elem)
        remaining.remove(best_elem)
    return selected

def _local_improve(selected, max_iters=2000, time_limit=5.0):
    """Local search: try swapping or adding/removing elements to improve size."""
    selected = sorted(selected)
    best_size = len(selected)
    start_time = time.time()
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Try adding a random element not in selected
        if random.random() < 0.4:
            candidates = [x for x in range(1, _N + 1) if x not in selected]
            if candidates:
                x = random.choice(candidates)
                new_selected = selected + [x]
                if _is_valid_set(new_selected):
                    selected = sorted(new_selected)
                    best_size = len(selected)
                    continue
        
        # Try swapping: remove one, add another
        if len(selected) > 0:
            a = random.choice(selected)
            candidates = [x for x in range(1, _N + 1) if x not in selected]
            if candidates:
                b = random.choice(candidates)
                new_selected = [x for x in selected if x != a] + [b]
                if _is_valid_set(new_selected) and len(new_selected) >= len(selected):
                    selected = sorted(new_selected)
                    best_size = len(selected)
        
        # Try removing a random element and re-adding a different one
        if len(selected) > 2:
            a = random.choice(selected)
            new_selected = [x for x in selected if x != a]
            candidates = [x for x in range(1, _N + 1) if x not in new_selected]
            if candidates:
                b = random.choice(candidates)
                test_selected = new_selected + [b]
                if _is_valid_set(test_selected) and len(test_selected) > len(selected):
                    selected = sorted(test_selected)
                    best_size = len(selected)
    
    return selected

def _structured_init():
    """Initialize with prime powers and other structured elements."""
    selected = set()
    # Add all primes up to 128
    for p in range(2, _N + 1):
        is_prime = True
        for d in range(2, int(p**0.5) + 1):
            if p % d == 0:
                is_prime = False
                break
        if is_prime:
            selected.add(p)
    # Add prime powers
    for p in range(2, _N + 1):
        is_prime = True
        for d in range(2, int(p**0.5) + 1):
            if p % d == 0:
                is_prime = False
                break
        if is_prime:
            power = p
            while power * p <= _N:
                power *= p
                selected.add(power)
    return list(selected)

def construct():
    deadline = time.time() + 80.0  # Safety margin before 90s kill
    best_set = None
    best_size = 0
    
    # Strategy 1: Greedy
    greedy_set = _greedy_add_order()
    if _is_valid_set(greedy_set):
        best_set = greedy_set
        best_size = len(greedy_set)
    
    # Strategy 2: Structured init + local improve
    if time.time() < deadline - 5:
        struct_set = _structured_init()
        improved = _local_improve(struct_set, max_iters=3000, time_limit=10.0)
        if _is_valid_set(improved) and len(improved) > best_size:
            best_set = improved
            best_size = len(improved)
    
    # Strategy 3: Randomized greedy with multiple restarts
    if time.time() < deadline - 5:
        for trial in range(20):
            if time.time() > deadline - 5:
                break
            random.shuffle(range(1, _N + 1))
            order = list(range(1, _N + 1))
            random.shuffle(order)
            sel = []
            for x in order:
                if all(not _is_bad_triple(a, b, x) for a, b in combinations(sel, 2)):
                    sel.append(x)
            if _is_valid_set(sel) and len(sel) > best_size:
                best_set = sel
                best_size = len(sel)
            # Local improve on this
            if time.time() < deadline - 10:
                imp = _local_improve(sel, max_iters=1000, time_limit=3.0)
                if _is_valid_set(imp) and len(imp) > best_size:
                    best_set = imp
                    best_size = len(imp)
    
    # Strategy 4: More aggressive local search from best
    if best_set and time.time() < deadline - 10:
        imp = _local_improve(list(best_set), max_iters=5000, time_limit=15.0)
        if _is_valid_set(imp) and len(imp) > best_size:
            best_set = imp
            best_size = len(imp)
    
    if best_set is None:
        best_set = [1]
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
