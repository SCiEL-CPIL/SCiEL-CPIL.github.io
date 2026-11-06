"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from math import gcd

def _is_valid_triple(a, b, c, gcd_cache):
    """Check if triple (a,b,c) has all three pairwise GCDs equal."""
    g_ab = gcd_cache[a][b]
    g_ac = gcd_cache[a][c]
    g_bc = gcd_cache[b][c]
    return g_ab == g_ac == g_bc

def _greedy(order, gcd_cache, n=128):
    """Greedy construction given an order of elements to consider."""
    selected = set()
    for x in order:
        # Check if adding x creates any bad triple with two existing elements
        ok = True
        sel_list = list(selected)
        for i in range(len(sel_list)):
            for j in range(i+1, len(sel_list)):
                a = sel_list[i]
                b = sel_list[j]
                if _is_valid_triple(a, b, x, gcd_cache):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            selected.add(x)
    return selected

def _local_search(selected, gcd_cache, n=128, time_limit=None):
    """Try to improve the set by swapping elements."""
    selected = set(selected)
    excluded = set(range(1, n+1)) - selected
    improved = True
    while improved:
        improved = False
        for x in list(excluded):
            for y in list(selected):
                # Try swapping y for x
                new_sel = (selected - {y}) | {x}
                # Check validity
                valid = True
                sel_list = list(new_sel)
                for i in range(len(sel_list)):
                    for j in range(i+1, len(sel_list)):
                        for k in range(j+1, len(sel_list)):
                            if _is_valid_triple(sel_list[i], sel_list[j], sel_list[k], gcd_cache):
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if valid and len(new_sel) > len(selected):
                    selected = new_sel
                    excluded = set(range(1, n+1)) - selected
                    improved = True
                    break
            if improved:
                break
            if time_limit and time.time() > time_limit:
                break
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline for safety
    
    # Precompute GCD cache
    n = 128
    gcd_cache = [[0]*(n+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(i, n+1):
            g = gcd(i, j)
            gcd_cache[i][j] = g
            gcd_cache[j][i] = g
    
    best_set = set()
    
    # Try multiple random orderings
    for trial in range(50):
        if time.time() > deadline:
            break
        order = list(range(1, n+1))
        random.shuffle(order)
        result = _greedy(order, gcd_cache, n)
        if len(result) > len(best_set):
            best_set = result
    
    # Also try natural order
    result = _greedy(list(range(1, n+1)), gcd_cache, n)
    if len(result) > len(best_set):
        best_set = result
    
    # Try reverse order
    result = _greedy(list(range(n, 0, -1)), gcd_cache, n)
    if len(result) > len(best_set):
        best_set = result
    
    # Try sorted by number of divisors (ascending) - primes first
    def num_divisors(x):
        count = 0
        for d in range(1, int(x**0.5)+1):
            if x % d == 0:
                count += 1
                if d != x//d:
                    count += 1
        return count
    
    order_by_divs = sorted(range(1, n+1), key=num_divisors)
    result = _greedy(order_by_divs, gcd_cache, n)
    if len(result) > len(best_set):
        best_set = result
    
    # Local search to improve
    if time.time() < deadline - 5:
        best_set = _local_search(best_set, gcd_cache, n, time_limit=deadline - 2)
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
