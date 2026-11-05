import time
import random
from math import gcd
from functools import reduce

N = 24

def _lcm(a, b):
    return a * b // gcd(a, b)

L = reduce(_lcm, range(1, N + 1), 1)
A = [0] * (N + 1)
for i in range(1, N + 1):
    A[i] = L // i

def _can_form_target(target, others):
    """Check if target can be formed by a subset of others (positive integers)."""
    if target == 0:
        return True
    if not others:
        return False
    if sum(others) < target:
        return False
    # Bitset DP: bit i set means sum i is reachable
    bits = 1  # bit 0 set (sum 0 reachable)
    for b in others:
        if b <= target:
            bits |= (bits << b)
    return (bits >> target) & 1

def _is_valid_set(S):
    """Check if S (list of elements) is a valid reciprocal-subsum-free set."""
    S_set = set(S)
    for x in S:
        target = A[x]
        others = [A[t] for t in S_set if t != x]
        if _can_form_target(target, others):
            return False
    return True

def _greedy_add(order):
    """Greedy: add elements in given order if valid."""
    S = set()
    for c in order:
        new_S = S | {c}
        if _is_valid_set(list(new_S)):
            S = new_S
    return S

def _greedy_remove(order):
    """Greedy: start with all, remove in given order if needed for validity."""
    S = set(range(1, N + 1))
    for c in order:
        if c in S:
            new_S = S - {c}
            if _is_valid_set(list(new_S)):
                S = new_S
    return S

def _local_search(S, deadline):
    """Local search to improve set S."""
    best = set(S)
    best_size = len(best)
    cur = set(S)
    
    while time.time() < deadline:
        # Try all single additions
        improved = False
        for c in range(1, N + 1):
            if c not in cur:
                new_S = cur | {c}
                if _is_valid_set(list(new_S)):
                    cur = new_S
                    improved = True
                    if len(cur) > best_size:
                        best = set(cur)
                        best_size = len(cur)
        
        # Try all single removals (to enable future additions)
        if not improved:
            for c in list(cur):
                if len(cur) > 1:
                    new_S = cur - {c}
                    # Check if removing c allows any addition
                    for d in range(1, N + 1):
                        if d not in new_S:
                            test_S = new_S | {d}
                            if _is_valid_set(list(test_S)):
                                cur = test_S
                                if len(cur) > best_size:
                                    best = set(cur)
                                    best_size = len(cur)
                                break
                    else:
                        # Try removing c alone to see if it helps
                        pass
        
        # Try swaps
        if not improved:
            for c in list(cur):
                for d in range(1, N + 1):
                    if d not in cur:
                        new_S = (cur - {c}) | {d}
                        if _is_valid_set(list(new_S)) and len(new_S) == len(cur):
                            # Check if this swap enables further additions
                            for e in range(1, N + 1):
                                if e not in new_S:
                                    test_S = new_S | {e}
                                    if _is_valid_set(list(test_S)):
                                        cur = test_S
                                        if len(cur) > best_size:
                                            best = set(cur)
                                            best_size = len(cur)
                                        break
        
        # If no improvement possible, break
        if not improved:
            break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin (90s limit)
    
    best_set = set()
    best_size = 0
    
    # Strategy 1: Greedy addition with various orderings
    orders = [
        list(range(N, 0, -1)),       # Large to small
        list(range(1, N + 1)),       # Small to large
        list(range(N, 0, -1, 2)) + list(range(N - 1, 0, -1, 2)),  # Odds then evens (large first)
        list(range(1, N + 1, 2)) + list(range(2, N + 1, 2)),      # Odds then evens (small first)
    ]
    
    # Add some random orderings
    for _ in range(20):
        random_order = list(range(1, N + 1))
        random.shuffle(random_order)
        orders.append(random_order)
    
    for order in orders:
        if time.time() > deadline:
            break
        S = _greedy_add(order)
        if len(S) > best_size:
            best_set = S
            best_size = len(S)
    
    # Strategy 2: Greedy removal
    for order in orders[:10]:
        if time.time() > deadline:
            break
        S = _greedy_remove(order)
        if len(S) > best_size:
            best_set = S
            best_size = len(S)
    
    # Strategy 3: Local search from best found
    if best_set and time.time() < deadline:
        improved = _local_search(best_set, deadline)
        if len(improved) > best_size:
            best_set = improved
            best_size = len(best_set)
    
    # Strategy 4: More random restarts with local search
    while time.time() < deadline:
        random_order = list(range(1, N + 1))
        random.shuffle(random_order)
        S = _greedy_add(random_order)
        if len(S) >= best_size:
            S = _local_search(S, deadline)
            if len(S) > best_size:
                best_set = S
                best_size = len(S)
    
    # Final: ensure nonempty
    if not best_set:
        best_set = {1}
    
    return list(best_set)
