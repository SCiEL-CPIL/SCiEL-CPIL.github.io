"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _is_valid(S):
    """Check if set S has no three distinct elements a,b,c with all pair sums in S."""
    S_set = set(S)
    n = len(S)
    for i in range(n):
        a = S[i]
        for j in range(i+1, n):
            b = S[j]
            ab = a + b
            if ab not in S_set:
                continue
            for k in range(j+1, n):
                c = S[k]
                if a + c in S_set and b + c in S_set:
                    return False
    return True

def _can_add(S_set, x):
    """Check if adding x to S_set creates no forbidden triple."""
    elems = list(S_set)
    n = len(elems)
    
    # 1. Check triples involving x: (a, b, x)
    # Sums: a+b, a+x, b+x must be in S_set U {x}.
    # a+x > x, b+x > x, so a+x, b+x in S_set.
    # a+b in S_set or a+b == x.
    for i in range(n):
        a = elems[i]
        xa = x + a
        if xa not in S_set:
            continue
        for j in range(i+1, n):
            b = elems[j]
            xb = x + b
            if xb not in S_set:
                continue
            ab = a + b
            if ab in S_set or ab == x:
                return False

    # 2. Check triples in S_set that become forbidden because x completes a sum.
    # Triple (a, b, c) in S_set. Sums a+b, a+c, b+c.
    # One sum must be x (since S_set was valid, at least one sum was missing).
    # If a+b == x, then we need a+c in S_set and b+c in S_set.
    # c must be distinct from a and b.
    
    # Find pairs (a, b) in S_set such that a + b == x
    pairs_summing_to_x = []
    for i in range(n):
        a = elems[i]
        b = x - a
        if b in S_set and b != a and a < b:
            pairs_summing_to_x.append((a, b))
            
    for a, b in pairs_summing_to_x:
        # Check if there exists c in S_set such that a+c in S_set and b+c in S_set
        # c must be distinct from a and b
        for c in elems:
            if c == a or c == b:
                continue
            if a + c in S_set and b + c in S_set:
                return False

    return True

def _greedy_add(base_set, candidates, deadline):
    """Greedily add candidates to base_set, checking validity."""
    current = set(base_set)
    for x in candidates:
        if time.time() > deadline - 3:
            break
        if _can_add(current, x):
            current = current | {x}
    return current

def _local_search(S_set, deadline, max_swaps=200):
    """Try to improve by removing one element and adding one or two others."""
    current = set(S_set)
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        removed = None
        added = None
        # Try all single swaps: remove r, add a
        for r in list(current):
            if time.time() > deadline - 5:
                break
            without = current - {r}
            for a in range(1, 129):
                if a in without:
                    continue
                if _can_add(without, a):
                    new_set = without | {a}
                    if len(new_set) > len(current):
                        current = new_set
                        improved = True
                        break
            if improved:
                break
    return current

def construct():
    deadline = time.time() + 80  # safety margin before 90s limit
    n = 128
    
    best_size = 0
    best_set = set()
    rng = random.Random(42)
    
    # Try various base ranges: from {65..128} down to {50..128}
    for start in range(65, 49, -1):
        if time.time() > deadline - 10:
            break
        base = set(range(start, n+1))
        candidates = list(range(1, start))
        
        # Greedy add in natural order
        cur = _greedy_add(base, candidates, deadline)
        if len(cur) > best_size:
            best_size = len(cur)
            best_set = set(cur)
        
        # Try several random orderings
        for trial in range(30):
            if time.time() > deadline - 10:
                break
            order = list(range(1, start))
            rng.shuffle(order)
            cur = _greedy_add(base, order, deadline)
            if len(cur) > best_size:
                best_size = len(cur)
                best_set = set(cur)
        
        # Local search on best found for this base
        if time.time() < deadline - 10:
            cur_ls = _local_search(best_set, deadline)
            if len(cur_ls) > best_size:
                best_size = len(cur_ls)
                best_set = set(cur_ls)
    
    # Also try starting with just high numbers and being more aggressive
    for start in range(70, 80):
        if time.time() > deadline - 5:
            break
        base = set(range(start, n+1))
        candidates = list(range(1, start))
        for trial in range(20):
            if time.time() > deadline - 5:
                break
            order = list(range(1, start))
            rng.shuffle(order)
            cur = _greedy_add(base, order, deadline)
            if len(cur) > best_size:
                best_size = len(cur)
                best_set = set(cur)
    
    # Final local search
    if time.time() < deadline - 5:
        cur_ls = _local_search(best_set, deadline)
        if len(cur_ls) > best_size:
            best_size = len(cur_ls)
            best_set = set(cur_ls)
    
    result = sorted(best_set)
    # Verify final result
    assert _is_valid(result), "Final result is invalid!"
    # Ensure all are Python ints
    return [int(x) for x in result]
# EVOLVE-BLOCK-END
