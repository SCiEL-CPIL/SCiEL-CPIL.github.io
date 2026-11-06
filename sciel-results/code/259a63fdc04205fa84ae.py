import time
import sys
import random

def construct():
    start_time = time.time()
    deadline = start_time + 87.0  # 3s safety margin
    
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

    def swap_local_search(S, deadline):
        """Try to improve S by swapping: remove one element, try to add two new ones."""
        current_set = set(S)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for a in sorted(current_set):
                if time.time() > deadline:
                    break
                test_set = set(current_set)
                test_set.remove(a)
                # Try to add back a and also try other elements
                candidates = [x for x in range(1, n+1) if x not in test_set and x != a]
                extended = greedy_extend(test_set, candidates, deadline)
                if len(extended) > len(current_set):
                    current_set = extended
                    improved = True
                    break
        return current_set

    def final_polish(S, deadline):
        """Final greedy pass: try to add any remaining elements."""
        current_set = set(S)
        all_elements = set(range(1, n+1))
        candidates = list(all_elements - current_set)
        current_set = greedy_extend(current_set, candidates, deadline)
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
    
    num_restarts = 15
    for restart in range(num_restarts):
        if time.time() > deadline - 3:
            break
        
        if restart == 0:
            # Deterministic: start from best interval
            current_set = set(best_set)
        else:
            # Random: pick a random upper-tail interval
            L = random.randint(140, 256)
            S_test = set(range(L, n + 1))
            if not is_valid_set(S_test):
                continue
            current_set = S_test
        
        # Greedy extend
        candidates = [x for x in range(1, min(list(current_set))) if x not in current_set]
        current_set = greedy_extend(current_set, candidates, deadline)
        
        # Swap-based local search
        if time.time() < deadline - 2:
            current_set = swap_local_search(current_set, deadline)
        
        # Final polish pass
        if time.time() < deadline - 1:
            current_set = final_polish(current_set, deadline)
        
        if len(current_set) > best_result_size:
            best_result = current_set
            best_result_size = len(current_set)

    # Final polish on best result
    if time.time() < deadline - 0.5:
        best_result = final_polish(best_result, deadline)

    return list(best_result)

# EVOLVE-BLOCK-END
