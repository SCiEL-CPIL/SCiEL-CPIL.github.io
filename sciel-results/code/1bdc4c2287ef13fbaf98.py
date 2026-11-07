"""Improved program: randomized greedy + local search for max sum-free triple set."""
import time
import random

DEADLINE = 85  # seconds, safety margin before 90s kill

def _check_addition(s_set, x):
    """Check if adding x to s_set creates a forbidden triple."""
    # A forbidden triple involving x: for distinct a,b in s_set,
    # we need x+a, x+b, a+b all in s_set ∪ {x}
    # Since x is new, x+a and x+b are new unless they equal some element
    # Actually: the triple (a,b,x) is forbidden if (a+b) in s_set, (a+x) in s_set∪{x}, (b+x) in s_set∪{x}
    # a+x = x means a=0, not possible since we're in 1..128
    # So we need: a+b in s_set, and (a+x) in s_set, and (b+x) in s_set
    # OR the triple could be (a,x,b) but that's the same
    # Also triples where x is one of the "sum" elements: but x itself isn't a sum of two others necessarily
    # Actually the condition is: three DISTINCT members a,b,c such that a+b, a+c, b+c are also selected.
    # So if we add x, new triples involving x: for each pair (a,b) in s_set, check if x,a,b form a forbidden triple.
    # That means: a+b in S, a+x in S∪{x}, b+x in S∪{x}
    # Since a+x > a and b+x > b, and x is new, a+x is in S only if a+x was already there (and a+x != x since a>0)
    # Similarly b+x in S only if already there.
    # So: for each pair (a,b) in s_set with a+b in s_set, check if (a+x) in s_set and (b+x) in s_set
    # Also need to check: could x be one of the sum values? i.e., x = a+b for some a,b in s_set?
    # If x = a+b, then for triple (a,b,c): a+b=x must be in set (it is, we're adding it), a+c and b+c must be in set.
    # So we also need: for each pair (a,b) in s_set with a+b == x, check if there exists c in s_set such that a+c in s_set and b+c in s_set.
    
    # Case 1: x is one of the three "base" elements
    # For pairs (a,b) in s_set: if a+b in s_set, and (a+x) in s_set, and (b+x) in s_set -> violation
    for a in s_set:
        for b in s_set:
            if a == b:
                continue
            if a + b in s_set:
                if (a + x) in s_set and (b + x) in s_set:
                    return False
    
    # Case 2: x is one of the sum values, i.e., x = a+b for some a,b in s_set
    # Then for any c in s_set distinct from a,b: need a+c in s_set and b+c in s_set
    for a in s_set:
        b = x - a
        if b in s_set and a != b and b > 0:
            # x = a + b, now check for any c in s_set, c != a, c != b
            for c in s_set:
                if c == a or c == b:
                    continue
                if (a + c) in s_set and (b + c) in s_set:
                    return False
    
    return True

def _is_valid_full(s):
    """Full validation of a set."""
    s_set = set(s)
    for a in s_set:
        for b in s_set:
            if a >= b:
                continue
            for c in s_set:
                if c <= b:
                    continue
                if (a + b) in s_set and (a + c) in s_set and (b + c) in s_set:
                    return False
    return True

def _greedy_construct(order, universe):
    """Greedily add elements in given order."""
    selected = set()
    for x in order:
        if x in selected:
            continue
        if _check_addition(selected, x):
            selected.add(x)
    return selected

def _local_search(selected, universe, time_budget):
    """Swap-based local search."""
    start_time = time.time()
    selected_set = set(selected)
    
    improved = True
    while improved and (time.time() - start_time) < time_budget:
        improved = False
        # Try adding
        for x in universe:
            if (time.time() - start_time) >= time_budget:
                break
            if x not in selected_set and _check_addition(selected_set, x):
                selected_set.add(x)
                improved = True
                break
        # Try swaps if no addition
        if not improved:
            for y in list(selected_set):
                if (time.time() - start_time) >= time_budget:
                    break
                for x in universe:
                    if x in selected_set:
                        continue
                    candidate = (selected_set - {y}) | {x}
                    if _is_valid_full(candidate):
                        selected_set = candidate
                        improved = True
                        break
                if improved:
                    break
    return selected_set

def construct():
    start_time = time.time()
    universe = list(range(1, 129))
    best_set = set(range(65, 129))
    best_size = len(best_set)
    
    rng = random.Random(42)
    
    # Deterministic orderings
    orders = [
        list(reversed(universe)),
        list(universe),
        sorted(universe, key=lambda x: x % 7),
        sorted(universe, key=lambda x: x % 11),
    ]
    
    for order in orders:
        if time.time() - start_time > DEADLINE - 5:
            break
        s = _greedy_construct(order, universe)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
    
    # Random restarts
    while time.time() - start_time < DEADLINE - 10:
        order = universe[:]
        rng.shuffle(order)
        s = _greedy_construct(order, universe)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
            remaining = DEADLINE - 15 - (time.time() - start_time)
            if remaining > 2:
                best_set = _local_search(best_set, universe, remaining)
                best_size = len(best_set)
    
    # Final local search
    remaining = DEADLINE - 3 - (time.time() - start_time)
    if remaining > 1:
        best_set = _local_search(best_set, universe, remaining)
        best_size = len(best_set)
    
    result = [int(x) for x in sorted(best_set)]
    return result

# EVOLVE-BLOCK-END
