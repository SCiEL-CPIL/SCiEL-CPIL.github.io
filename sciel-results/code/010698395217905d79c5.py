"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations
from math import gcd

DEADLINE = 85.0  # safety margin before 90s kill

def _gcd_matrix(n, limit):
    """Precompute gcd for all pairs up to limit."""
    m = [[0] * (limit + 1) for _ in range(limit + 1)]
    for i in range(1, limit + 1):
        for j in range(i, limit + 1):
            m[i][j] = gcd(i, j)
            m[j][i] = m[i][j]
    return m

def _is_valid_set(selected, gcd_m):
    """Check if selected set has no three distinct elements with all pairwise gcds equal."""
    n = len(selected)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                g1 = gcd_m[selected[i]][selected[j]]
                g2 = gcd_m[selected[i]][selected[k]]
                g3 = gcd_m[selected[j]][selected[k]]
                if g1 == g2 == g3:
                    return False
    return True

def _greedy_construct(gcd_m, limit, start_order=None):
    """Greedy construction: add elements in given order if valid."""
    if start_order is None:
        start_order = list(range(1, limit + 1))
    selected = []
    for x in start_order:
        # Check if adding x violates any triple condition
        valid = True
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                a, b = selected[i], selected[j]
                g_ab = gcd_m[a][b]
                g_ax = gcd_m[a][x]
                g_bx = gcd_m[b][x]
                if g_ab == g_ax == g_bx:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            selected.append(x)
    return selected

def _local_search(selected, gcd_m, limit, time_budget):
    """Local search: try to swap out an element for a better one, or add/remove to maximize size."""
    start = time.time()
    best = list(selected)
    best_size = len(best)
    
    while time.time() - start < time_budget:
        improved = False
        
        # Try adding each unused element
        used_set = set(best)
        for x in range(1, limit + 1):
            if x in used_set:
                continue
            candidate = best + [x]
            if _is_valid_set(candidate, gcd_m):
                best = candidate
                best_size += 1
                improved = True
                break
        
        if improved:
            continue
        
        # Try swapping: remove one, add another
        for i in range(len(best)):
            removed = best[i]
            base = best[:i] + best[i+1:]
            for x in range(1, limit + 1):
                if x in set(base):
                    continue
                candidate = base + [x]
                if _is_valid_set(candidate, gcd_m):
                    # Check if this is an improvement (should be same size, so no size gain)
                    # But maybe it enables future additions? For now, just accept if valid
                    # Actually, swapping keeps same size, so we need a different criterion
                    # Let's try: after swap, can we add more?
                    # For simplicity, accept the swap if it doesn't decrease size (it won't)
                    # and hope it helps. But this could loop. Let's be more careful.
                    # Only accept if the new element is "better" in some sense.
                    # For now, skip swaps that don't increase size.
                    pass
            # If we found a swap that allows future growth, we'd need lookahead.
            # Skip for now.
        
        if not improved:
            break
    
    return best

def construct():
    start_time = time.time()
    limit = 128
    gcd_m = _gcd_matrix(limit, limit)
    
    # Try multiple orderings
    orderings = [
        list(range(1, limit + 1)),
        list(range(limit, 0, -1)),
        # Primes first
        [x for x in range(2, limit + 1) if all(x % d for d in range(2, int(x**0.5) + 1))] +
        [x for x in range(1, limit + 1) if not (x >= 2 and all(x % d for d in range(2, int(x**0.5) + 1)))],
        # Powers of 2 first
        [2**i for i in range(1, 8) if 2**i <= limit] +
        [x for x in range(1, limit + 1) if x not in set(2**i for i in range(1, 8) if 2**i <= limit)],
        # Odd first
        [x for x in range(1, limit + 1, 2)] + [x for x in range(2, limit + 1, 2)],
        # Even first
        [x for x in range(2, limit + 1, 2)] + [x for x in range(1, limit + 1, 2)],
    ]
    
    best = []
    best_size = 0
    
    for order in orderings:
        if time.time() - start_time > DEADLINE - 10:
            break
        result = _greedy_construct(gcd_m, limit, order)
        if len(result) > best_size:
            best = result
            best_size = len(result)
    
    # Local search on the best result
    remaining = DEADLINE - (time.time() - start_time) - 2
    if remaining > 1:
        improved = _local_search(best, gcd_m, limit, remaining)
        if len(improved) > best_size:
            best = improved
            best_size = len(improved)
    
    # Ensure nonempty
    if not best:
        best = [1]
    
    return best
# EVOLVE-BLOCK-END
