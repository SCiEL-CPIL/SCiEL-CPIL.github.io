"""Improved program for Large sets with Property P."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def _is_valid(S):
    """Check if set S satisfies: for all a<b<=c in S, a does NOT divide b+c."""
    S_sorted = sorted(S)
    n = len(S_sorted)
    for i in range(n):
        a = S_sorted[i]
        for j in range(i + 1, n):
            b = S_sorted[j]
            for k in range(j, n):
                c = S_sorted[k]
                if (b + c) % a == 0:
                    return False
    return True

def _precompute_conflicts(n=256):
    """
    Precompute for each a in [1,n], the set of pairs (b,c) with b<=c, b,c in [1,n],
    such that a divides b+c.
    Returns a list where conflicts[a] is a set of (b,c) pairs.
    Also returns for each a, a set of 'bad' sums: sums s such that a|s.
    """
    conflicts = [set() for _ in range(n + 1)]
    for a in range(1, n + 1):
        for b in range(1, n + 1):
            for c in range(b, n + 1):
                if (b + c) % a == 0:
                    conflicts[a].add((b, c))
    return conflicts

def _check_validity_fast(S, n=256):
    """
    Fast validity check. For each a in S, check if there exist b,c in S with a<b<=c
    such that a divides b+c.
    """
    S_set = set(S)
    S_sorted = sorted(S)
    n_elems = len(S_sorted)
    
    for i in range(n_elems):
        a = S_sorted[i]
        # Find all b, c in S with b > a, c >= b
        larger = [x for x in S_sorted[i+1:]]
        for j in range(len(larger)):
            b = larger[j]
            for k in range(j, len(larger)):
                c = larger[k]
                if (b + c) % a == 0:
                    return False
    return True

def _check_add_valid(S, x, n=256):
    """
    Check if adding x to S maintains validity.
    S is assumed to be valid.
    x is not in S.
    """
    S_set = set(S)
    S_sorted = sorted(S)
    n_elems = len(S_sorted)
    
    # x can play role of a, b, or c
    # Case 1: x is a (smallest in the triple)
    # Need: for all b < c in S with b > x, c >= b, x does NOT divide b+c
    larger_than_x = [v for v in S_sorted if v > x]
    for i in range(len(larger_than_x)):
        b = larger_than_x[i]
        for j in range(i + 1, len(larger_than_x)):
            c = larger_than_x[j]
            if (b + c) % x == 0:
                return False
    
    # Case 2: x is b or c (larger in the triple)
    # For each a in S with a < x, check:
    #   - x as b: a < x <= c, check if a divides x + c for c in S, c >= x
    #   - x as c: a < b <= x, check if a divides b + x for b in S, a < b <= x
    smaller_than_x = [v for v in S_sorted if v < x]
    for a in smaller_than_x:
        # x as b: need c in S with c >= x, check a | (x + c)
        for c in S_sorted:
            if c >= x and (x + c) % a == 0:
                return False
        # x as c: need b in S with a < b <= x, check a | (b + x)
        for b in S_sorted:
            if a < b <= x and (b + x) % a == 0:
                return False
    
    return True

def _check_swap_valid(S, x_remove, x_add, n=256):
    """
    Check if swapping x_remove for x_add in S maintains validity.
    S is assumed to be valid, x_remove in S, x_add not in S.
    """
    S_new = (S - {x_remove}) | {x_add}
    # Just do full check on the new set - it's simpler and for n~170 it's O(n^3)
    # But we can be smarter: only check triples involving x_add
    S_set = set(S_new)
    S_sorted = sorted(S_new)
    n_elems = len(S_sorted)
    
    # Check all triples involving x_add
    for i in range(n_elems):
        a = S_sorted[i]
        for j in range(i + 1, n_elems):
            b = S_sorted[j]
            for k in range(j, n_elems):
                c = S_sorted[k]
                if a == x_add or b == x_add or c == x_add:
                    if (b + c) % a == 0:
                        return False
    # Also need to verify that removing x_remove doesn't break anything
    # (it can't, since S was valid and removing elements can't create violations)
    return True

def _greedy_construct():
    """Greedy construction: start from upper tail and extend downward."""
    # Start with [171, 256] which is guaranteed valid (3*171=513 > 2*256=512)
    S = set(range(171, 257))
    
    # Try adding elements from 1 to 170, in order of increasing value
    # (smaller elements are more likely to conflict, so try them last? No, try in order)
    for x in range(1, 171):
        if _check_add_valid(S, x):
            S.add(x)
    
    return S

def _local_search(S, max_iterations=500000, timeout=55.0):
    """Local search to improve the set."""
    start_time = time.time()
    best_S = set(S)
    best_size = len(best_S)
    
    all_elements = list(range(1, 257))
    
    iteration = 0
    while True:
        elapsed = time.time() - start_time
        if elapsed > timeout:
            break
        if iteration >= max_iterations:
            break
        iteration += 1
        
        # Try adding a random element not in S
        if len(best_S) < 256:
            not_in_S = [x for x in all_elements if x not in best_S]
            if not_in_S:
                x = random.choice(not_in_S)
                if _check_add_valid(best_S, x):
                    best_S.add(x)
                    best_size = len(best_S)
                    continue
        
        # Try swapping: remove one, add another
        if len(best_S) > 1:
            elements_in_S = list(best_S)
            x_to_remove = random.choice(elements_in_S)
            not_in_S = [x for x in all_elements if x not in best_S]
            if not_in_S:
                x_to_add = random.choice(not_in_S)
                if _check_swap_valid(best_S, x_to_remove, x_to_add):
                    best_S.discard(x_to_remove)
                    best_S.add(x_to_add)
                    best_size = len(best_S)
    
    return best_S

def _greedy_with_order():
    """Try different greedy orders to find better starting points."""
    best_S = set()
    best_size = 0
    
    # Try different orderings
    orderings = [
        list(range(1, 257)),           # ascending
        list(range(256, 0, -1)),       # descending
        list(range(129, 257)) + list(range(1, 129)),  # upper half first
        list(range(171, 257)) + list(range(1, 171)),  # known valid start
    ]
    
    for order in orderings:
        S = set()
        for x in order:
            if _check_add_valid(S, x):
                S.add(x)
        if len(S) > best_size:
            best_size = len(S)
            best_S = S
    
    return best_S

def construct():
    start_time = time.time()
    
    # Start with greedy construction
    S = _greedy_construct()
    size_greedy = len(S)
    
    # Also try different greedy orders
    S2 = _greedy_with_order()
    if len(S2) > size_greedy:
        S = S2
        size_greedy = len(S2)
    
    best_S = set(S)
    best_size = size_greedy
    
    # Use a fixed seed for reproducibility
    random.seed(42)
    
    # Local search
    remaining_time = 55.0 - (time.time() - start_time)
    if remaining_time > 3:
        best_S = _local_search(best_S, max_iterations=1000000, timeout=remaining_time - 2)
        best_size = len(best_S)
    
    # Additional restarts if time permits
    remaining_time = 55.0 - (time.time() - start_time)
    restart_count = 0
    while remaining_time > 8 and restart_count < 3:
        # Perturb the best solution
        perturbed = set(best_S)
        for _ in range(2):
            if perturbed:
                x_to_remove = random.choice(list(perturbed))
                perturbed.discard(x_to_remove)
            not_in_S = [x for x in range(1, 257) if x not in perturbed]
            if not_in_S:
                x_to_add = random.choice(not_in_S)
                perturbed.add(x_to_add)
        
        result = _local_search(perturbed, max_iterations=200000, timeout=remaining_time - 4)
        if len(result) > best_size:
            best_S = result
            best_size = len(result)
        restart_count += 1
        remaining_time = 55.0 - (time.time() - start_time)
    
    # Final safety check
    if not _is_valid(best_S) and len(best_S) > 0:
        # Fall back to the guaranteed valid set
        return list(range(171, 257))
    
    return list(best_S)
# EVOLVE-BLOCK-END
