import random
import time

def is_sum_reachable_non_empty(diffs, target):
    if not diffs:
        return False
    
    all_reachable = {0}
    non_empty_reachable = set()
    
    for d in diffs:
        new_sums = {s + d for s in all_reachable}
        all_reachable = all_reachable | new_sums
        non_empty_reachable = non_empty_reachable | new_sums
        
    if target == 0:
        return 0 in non_empty_reachable
    else:
        return target in all_reachable

def check_addition(S, c):
    # S is a set of ints
    # c is the candidate to add
    
    # Case 1: c is the mean of a subset of S
    # Differences: d_i = x - c for x in S
    # Target: 0
    # Condition: Non-empty subset sums to 0.
    diffs = [x - c for x in S]
    if is_sum_reachable_non_empty(diffs, 0):
        return False
        
    # Case 2: a in S is the mean of a subset of (S \ {a}) U {c}
    # This requires the subset to include c.
    # Sum_{x in T} x + c = a * (|T| + 1)
    # Sum_{x in T} (x - a) = a - c
    # T is subset of S \ {a}, T non-empty.
    for a in S:
        target = a - c
        diffs_a = [x - a for x in S if x != a]
        if is_sum_reachable_non_empty(diffs_a, target):
            return False
            
    return True

def is_valid_set(S):
    S_list = list(S)
    for a in S_list:
        others = [x for x in S_list if x != a]
        diffs = [x - a for x in others]
        if is_sum_reachable_non_empty(diffs, 0):
            return False
    return True

def construct():
    start_time = time.time()
    deadline = 85.0
    
    # Greedy init
    S = set()
    for c in range(1, 513):
        if check_addition(S, c):
            S.add(c)
        if time.time() > deadline:
            return list(S)
    
    best_S = set(S)
    
    # Local Search
    while time.time() < deadline:
        added = False
        unused = [x for x in range(1, 513) if x not in S]
        random.shuffle(unused)
        for c in unused:
            if check_addition(S, c):
                S.add(c)
                added = True
                break
        
        if not added:
            swapped = False
            a_list = list(S)
            random.shuffle(a_list)
            for a in a_list:
                S_prime = set(S)
                S_prime.remove(a)
                for c in unused:
                    if check_addition(S_prime, c):
                        S = S_prime
                        S.add(c)
                        swapped = True
                        break
                if swapped:
                    break
            
            if not swapped:
                # If stuck, try a random restart of the greedy phase with a different seed/order?
                # Or just continue.
                pass
        
        if len(S) > len(best_S):
            best_S = set(S)
            
    return list(best_S)
