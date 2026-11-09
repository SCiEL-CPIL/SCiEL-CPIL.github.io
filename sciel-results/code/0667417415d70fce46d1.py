import time
import sys
import random

START_TIME = time.time()
DEADLINE = 55.0

def elapsed():
    return time.time() - START_TIME

def is_done():
    return elapsed() > DEADLINE

def get_subset_sum_bitsets(A):
    """
    Returns a list of bitsets S where S[j] has bit s set if sum s is achievable with cardinality j.
    Returns None if invalid.
    """
    n = len(A)
    if n == 0:
        return [1]
    S = [0] * (n + 1)
    S[0] = 1
    
    for i in range(n):
        x = A[i]
        for j in range(i, -1, -1):
            new_part = S[j] << x
            # Check collision with other cardinalities
            for k in range(n + 1):
                if k == j + 1:
                    continue
                if new_part & S[k]:
                    return None
            S[j+1] |= new_part
            
    # Final check
    for j in range(n + 1):
        for k in range(j + 1, n + 1):
            if S[j] & S[k]:
                return None
    return S

def dfs_find_set(K, candidates, current_set, start_idx, deadline):
    """
    Depth-first search to find a valid set of size K using candidates from start_idx onwards.
    """
    if is_done() or time.time() > deadline:
        return None
    if len(current_set) == K:
        return current_set[:]
    
    # Pruning: Not enough candidates left
    if len(candidates) - start_idx < K - len(current_set):
        return None
        
    for i in range(start_idx, len(candidates)):
        x = candidates[i]
        new_set = current_set + [x]
        
        # Fast validity check
        S_new = get_subset_sum_bitsets(new_set)
        
        if S_new is not None:
            res = dfs_find_set(K, candidates, new_set, i + 1, deadline)
            if res is not None:
                return res
                
    return None

def solve_with_backtracking(K, order_type, max_time=10.0):
    """
    Attempts to find a set of size K using backtracking with a specific candidate order.
    """
    start_time = time.time()
    deadline = start_time + max_time
    
    all_elements = list(range(1, 257))
    
    if order_type == 'asc':
        candidates = all_elements
    elif order_type == 'desc':
        candidates = all_elements[::-1]
    elif order_type == 'rand':
        candidates = all_elements[:]
        random.shuffle(candidates)
    else:
        candidates = all_elements

    # We don't need to pass current_bitsets explicitly if we recompute, 
    # but passing it would allow incremental updates. 
    # For simplicity and correctness, we recompute in get_subset_sum_bitsets.
    # This is O(N^2 * M/W) which is fast for N<=12.
    
    return dfs_find_set(K, candidates, [], 0, deadline)

def construct():
    # Base solution: Powers of 2, size 9
    base_solution = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    
    # Try to find larger sets using backtracking with different heuristics
    # We try K from 12 down to 9
    # For each K, we try different orders
    
    orders = ['asc', 'desc', 'rand']
    
    for K in range(12, 8, -1):
        if elapsed() > DEADLINE - 2:
            break
        for order in orders:
            if elapsed() > DEADLINE - 2:
                break
            # Allocate remaining time to this attempt, with a safety margin
            remaining = DEADLINE - elapsed() - 1
            if remaining < 1:
                continue
            res = solve_with_backtracking(K, order, max_time=remaining)
            if res is not None:
                return res
        
    return base_solution

if __name__ == "__main__":
    res = construct()
    print(f"Found set of size {len(res)}: {res}")
