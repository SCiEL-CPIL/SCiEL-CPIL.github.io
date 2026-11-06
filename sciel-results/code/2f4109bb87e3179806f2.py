import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    
    n = 256
    
    def is_valid_set(S):
        # S is a set of integers
        # Check: for all a < b <= c in S, a does not divide b+c
        S_list = list(S)
        S_list.sort()
        for i, a in enumerate(S_list):
            # b must be > a
            # c must be >= b
            for j in range(i + 1, len(S_list)):
                b = S_list[j]
                for k in range(j, len(S_list)):
                    c = S_list[k]
                    if (b + c) % a == 0:
                        return False
        return True

    def check_add(S, x):
        """Check if adding x to S keeps it valid. Only need to check constraints where a=x."""
        S_list = list(S)
        for i, b in enumerate(S_list):
            for k in range(i, len(S_list)):
                c = S_list[k]
                if (b + c) % x == 0:
                    return False
        return True

    def check_remove(S, x):
        """Check if removing x from S keeps it valid. Removing can only help, so always valid."""
        return True

    # 1. Find the best interval [L, 256]
    best_L = 1
    best_size = 0
    best_set = set()
    
    for L in range(1, n + 1):
        if time.time() > deadline:
            break
        S = set(range(L, n + 1))
        
        # Quick check for interval using the smallest element L
        min_sum = 2 * (L + 1)
        max_sum = 2 * n
        k_min = (2 * (L + 1) + L - 1) // L  # ceiling(2(L+1)/L)
        k_max = (2 * n) // L
        if k_min <= k_max:
            valid = False
        else:
            if L >= 150:
                S_test = set(range(L, n + 1))
                if is_valid_set(S_test):
                    size = len(S_test)
                    if size > best_size:
                        best_size = size
                        best_set = S_test
                        best_L = L
            else:
                pass

    if not best_set:
        return list(range(192, 257))

    # 2. Greedy Expansion (top-down)
    current_set = set(best_set)
    candidates = list(range(best_L - 1, 0, -1))
    
    for x in candidates:
        if time.time() > deadline:
            break
        
        if check_add(current_set, x):
            current_set.add(x)

    # 3. Local Search: try to improve by removing and re-adding elements
    # Multiple passes of local search
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        current_list = sorted(current_set)
        
        # Try removing each element and see if we can add more
        for i, x in enumerate(current_list):
            if time.time() > deadline:
                break
            
            # Tentatively remove x
            test_set = set(current_set)
            test_set.discard(x)
            
            # Try to add elements that were previously rejected
            # Focus on elements smaller than the current minimum or near x
            min_elem = min(test_set) if test_set else n + 1
            
            # Try adding elements from 1 to min_elem - 1, and also around x
            candidates_to_add = []
            for y in range(1, min_elem):
                if y not in test_set:
                    candidates_to_add.append(y)
            # Also try elements near x
            for y in range(max(1, x - 10), min(n + 1, x + 11)):
                if y not in test_set and y != x:
                    candidates_to_add.append(y)
            
            # Try to add as many as possible
            new_count = 0
            for y in sorted(candidates_to_add):
                if check_add(test_set, y):
                    test_set.add(y)
                    new_count += 1
            
            # If we added more than we removed, accept
            if new_count > 1:  # removed 1, added more than 1
                current_set = test_set
                improved = True
                break
            # If we added exactly 1, it's neutral; skip for now to avoid cycling

    # 4. Final greedy pass: try to add any remaining elements
    for x in range(1, n + 1):
        if time.time() > deadline:
            break
        if x not in current_set and check_add(current_set, x):
            current_set.add(x)

    return list(current_set)

# EVOLVE-BLOCK-END
