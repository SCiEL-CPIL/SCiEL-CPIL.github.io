import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin
    
    N = 256
    numbers = list(range(2, N + 1))
    n = len(numbers)
    
    # Precompute gcds? math.gcd is fast enough.
    
    def is_valid_set(S):
        # S is a set of ints
        # Check all triples
        S_list = list(S)
        L = len(S_list)
        for i in range(L):
            a = S_list[i]
            for j in range(i + 1, L):
                b = S_list[j]
                for k in range(j + 1, L):
                    c = S_list[k]
                    if (b * c) % a == 0 or (a * c) % b == 0 or (a * b) % c == 0:
                        return False
        return True

    def can_add(x, S):
        # Check if x can be added to S
        # S is a set
        S_list = list(S)
        L = len(S_list)
        for i in range(L):
            b = S_list[i]
            for j in range(i + 1, L):
                c = S_list[j]
                if (b * c) % x == 0:
                    return False
                if (x * c) % b == 0:
                    return False
                if (x * b) % c == 0:
                    return False
        return True

    def can_swap(a, b, S):
        # Check if S - {a} + {b} is valid
        # a in S, b not in S
        S_new = S - {a}
        # Check validity of S_new
        # And then check if b can be added to S_new?
        # No, we just need to check if the new set is valid.
        # Since S was valid, we only need to check triples involving b.
        # Triples in S_new are already valid.
        # So we just check if b can be added to S_new.
        return can_add(b, S_new)

    # Greedy Construction
    # Sort by value descending (large numbers are less restrictive)
    current_set = set()
    candidates = sorted(numbers, reverse=True)
    
    for x in candidates:
        if can_add(x, current_set):
            current_set.add(x)
            
    best_set = set(current_set)
    best_size = len(best_set)
    
    # Local Search
    # We try to improve by swapping elements
    # To avoid getting stuck, we use a randomized approach with a time limit
    
    swaps_performed = 0
    max_swaps = 5000
    
    while time.time() - start_time < deadline:
        if swaps_performed > max_swaps:
            break
            
        # 1. Try to add any candidate
        added = False
        # Shuffle candidates to break symmetry or just iterate
        # Iterating in fixed order is deterministic, might get stuck.
        # Let's try to add any valid candidate.
        remaining = [x for x in numbers if x not in current_set]
        for x in remaining:
            if can_add(x, current_set):
                current_set.add(x)
                added = True
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = set(current_set)
                break
                
        if added:
            continue
            
        # 2. If no addition, try a swap
        swapped = False
        # To avoid cycling, we can randomize the order of a and b
        # Or just pick the first valid swap found
        a_list = list(current_set)
        random.shuffle(a_list)
        
        for a in a_list:
            if time.time() - start_time > deadline:
                break
            b_list = [x for x in numbers if x not in current_set]
            random.shuffle(b_list)
            
            for b in b_list:
                if can_swap(a, b, current_set):
                    current_set.remove(a)
                    current_set.add(b)
                    swapped = True
                    swaps_performed += 1
                    break
            if swapped:
                break
                
        if not swapped:
            # No swap possible, try a different strategy or break
            # Maybe try to remove an element and add two? (Too slow)
            # Or just break to avoid infinite loop
            break
            
    # Final check and return
    # Ensure best_set is the largest found
    # If current_set is larger, update best_set
    if len(current_set) > len(best_set):
        best_set = current_set
        
    # Verify best_set is valid (just in case)
    # is_valid_set(best_set) -- this is O(N^3) which is 255^3 ~ 16M ops, might take 1-2s.
    # Given the time spent in search, we might have little time left.
    # The search logic ensures validity, so we can trust it.
    
    return sorted(best_set)
