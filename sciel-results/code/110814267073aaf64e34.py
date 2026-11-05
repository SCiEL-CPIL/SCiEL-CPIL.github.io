import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin
    
    # Initial set
    S = set(range(1, 101))
    
    def is_valid(s):
        # Check if 200 is a subset sum of s
        possible = {0}
        for x in s:
            new = {p + x for p in possible if p + x <= 200}
            possible |= new
            if 200 in possible:
                return False
        return True

    # Greedy removal: remove min element from witness
    while is_valid(S) == False:
        if time.time() > deadline:
            break
        # Find a witness subset
        # Backtracking to find one subset
        witness = find_witness(S)
        if not witness:
            break
        # Remove the smallest element in witness
        x_to_remove = min(witness)
        S.remove(x_to_remove)
        
    # Local search: try to add elements back
    # Iterate through all elements not in S, try to add them
    # Sort candidates by value? Smaller values are more likely to be safe?
    # Or larger values?
    # Let's try adding in increasing order
    candidates = sorted([x for x in range(1, 101) if x not in S])
    
    for x in candidates:
        if time.time() > deadline:
            break
        if is_valid(S | {x}):
            S.add(x)
            
    # Repeat local search a few times
    for _ in range(5):
        if time.time() > deadline:
            break
        changed = False
        for x in candidates:
            if x in S:
                continue
            if is_valid(S | {x}):
                S.add(x)
                changed = True
        if not changed:
            break
            
    return list(S)

def find_witness(S):
    # Find a subset of S summing to 200
    # DP with path tracking
    # target = 200
    # dp[s] = set of elements used? No, too big.
    # Just backtrack.
    S_list = sorted(S)
    
    def backtrack(idx, current_sum, current_set):
        if current_sum == 200:
            return current_set
        if idx == len(S_list) or current_sum > 200:
            return None
        # Option 1: include S_list[idx]
        res = backtrack(idx + 1, current_sum + S_list[idx], current_set | {S_list[idx]})
        if res:
            return res
        # Option 2: exclude
        res = backtrack(idx + 1, current_sum, current_set)
        if res:
            return res
        return None

    return backtrack(0, 0, set())
