import time
import sys
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    n = 256
    
    def is_valid_set(S):
        """Check: for all a < b <= c in S, a does not divide b+c"""
        S_list = sorted(S)
        for i, a in enumerate(S_list):
            for j in range(i + 1, len(S_list)):
                b = S_list[j]
                for k in range(j, len(S_list)):
                    c = S_list[k]
                    if (b + c) % a == 0:
                        return False
        return True

    def can_add_element(S, x):
        """Check if x can be added to S without violating constraints.
        Assumes x is smaller than all elements in S."""
        S_list = sorted(S)
        for i, b in enumerate(S_list):
            for k in range(i, len(S_list)):
                c = S_list[k]
                if (b + c) % x == 0:
                    return False
        return True

    def greedy_extend(S, candidates, deadline):
        """Try to add elements from candidates to S, largest first."""
        current_set = set(S)
        for x in sorted(candidates, reverse=True):
            if time.time() > deadline:
                break
            if can_add_element(current_set, x):
                current_set.add(x)
        return current_set

    def local_search_polish(S, deadline):
        """Try to improve S by removing elements that block others, then re-adding."""
        current_set = set(S)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try removing each element and see if we can add more
            for a in list(current_set):
                if time.time() > deadline:
                    break
                test_set = set(current_set)
                test_set.remove(a)
                # Try to add back a and also try other elements
                candidates = [x for x in range(1, n+1) if x not in test_set]
                extended = greedy_extend(test_set, candidates, deadline)
                if len(extended) > len(current_set):
                    current_set = extended
                    improved = True
                    break
        return current_set

    # 1. Find the best interval [L, 256]
    best_set = set()
    best_size = 0
    
    for L in range(1, n + 1):
        if time.time() > deadline:
            break
        
        # Quick check: does L divide any b+c where b,c in [L+1, n]?
        k_min = (2 * (L + 1) + L - 1) // L
        k_max = (2 * n) // L
        if k_min <= k_max:
            continue
        
        # Full check for promising L
        if L >= 140:
            S_test = set(range(L, n + 1))
            if is_valid_set(S_test):
                size = len(S_test)
                if size > best_size:
                    best_size = size
                    best_set = S_test

    if not best_set:
        return list(range(192, 257))

    # 2. Multi-start greedy with random restarts
    best_result = set(best_set)
    best_result_size = len(best_result)
    
    num_restarts = 10
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        if restart == 0:
            # Deterministic: start from best interval
            current_set = set(best_set)
        else:
            # Random: pick a random upper-tail interval
            L = random.randint(150, 256)
            S_test = set(range(L, n + 1))
            if not is_valid_set(S_test):
                continue
            current_set = S_test
        
        # Greedy extend
        candidates = [x for x in range(1, min(list(current_set))) if x not in current_set]
        current_set = greedy_extend(current_set, candidates, deadline)
        
        # Local search polish
        if time.time() < deadline - 2:
            current_set = local_search_polish(current_set, deadline)
        
        if len(current_set) > best_result_size:
            best_result = current_set
            best_result_size = len(current_set)

    return list(best_result)

# EVOLVE-BLOCK-END
