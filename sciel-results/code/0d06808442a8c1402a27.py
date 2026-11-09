"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction
from math import gcd
from functools import reduce

def _lcm(a, b):
    return a * b // gcd(a, b)

def _lcm_list(lst):
    if not lst:
        return 1
    return reduce(_lcm, lst)

def _is_valid_set(S):
    """Check if S is reciprocal-subsum-free."""
    if not S:
        return True
    S_list = list(S)
    L = _lcm_list(S_list)
    # For each x in S, check if L/x equals sum of L/y for some non-empty subset of S\{x}
    for i, x in enumerate(S_list):
        target = L // x
        others = [S_list[j] for j in range(len(S_list)) if j != i]
        # Convert others to integer weights: L // y
        weights = [L // y for y in others]
        # Check if any non-empty subset of weights sums to target
        n = len(weights)
        # Use set of achievable sums (subset sum)
        achievable = {0}
        for w in weights:
            new_sums = set()
            for s in achievable:
                ns = s + w
                if ns == target:
                    return False
                if ns < target:
                    new_sums.add(ns)
            achievable |= new_sums
            if target in achievable:
                return False
    return True

def _can_add(S, x):
    """Check if x can be added to S while maintaining validity."""
    S_list = list(S)
    if x in S_list:
        return False
    new_S = S_list + [x]
    return _is_valid_set(new_S)

def _greedy_construct(order, deadline):
    """Greedily construct a valid set following the given order."""
    S = []
    S_set = set()
    for x in order:
        if time.time() > deadline:
            break
        if x in S_set:
            continue
        if _can_add(S, x):
            S.append(x)
            S_set.add(x)
    return S

def _local_search(S, deadline):
    """Try to improve S by adding elements or swapping."""
    S = list(S)
    S_set = set(S)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try adding each missing element
        for x in range(1, 25):
            if time.time() > deadline:
                break
            if x not in S_set:
                if _can_add(S, x):
                    S.append(x)
                    S_set.add(x)
                    improved = True
        # Try swapping: remove one, add two
        for i in range(len(S)):
            if time.time() > deadline:
                break
            for x in range(1, 25):
                if time.time() > deadline:
                    break
                if x not in S_set:
                    new_S = S[:i] + S[i+1:] + [x]
                    if len(new_S) > len(S) and _is_valid_set(new_S):
                        S = new_S
                        S_set = set(S)
                        improved = True
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    best = [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    best_size = len(best)
    
    # Try multiple orders
    random.seed(42)
    orders = []
    # Descending
    orders.append(list(range(24, 0, -1)))
    # Ascending
    orders.append(list(range(1, 25)))
    # Various random orders
    for _ in range(50):
        if time.time() > deadline:
            break
        perm = list(range(1, 25))
        random.shuffle(perm)
        orders.append(perm)
    
    for order in orders:
        if time.time() > deadline:
            break
        S = _greedy_construct(order, deadline)
        if len(S) > best_size:
            best = S
            best_size = len(S)
        # Local search on this result
        S2 = _local_search(S, deadline)
        if len(S2) > best_size:
            best = S2
            best_size = len(S2)
    
    # Final local search on best
    best_final = _local_search(best, deadline)
    if len(best_final) > best_size:
        best = best_final
    
    return best
# EVOLVE-BLOCK-END
