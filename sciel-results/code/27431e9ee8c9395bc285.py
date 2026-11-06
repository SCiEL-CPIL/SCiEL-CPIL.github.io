import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    N = 256
    
    # Precompute weights as float for speed
    weights_float = [0.0] * (N + 1)
    for i in range(1, N + 1):
        weights_float[i] = 1.0 / i
        
    # Precompute suffix sums for pruning (float)
    suffix_float = [0.0] * (N + 2)
    for i in range(N, 0, -1):
        suffix_float[i] = suffix_float[i + 1] + weights_float[i]
        
    # Precompute AP pairs for each x
    # pairs[x] = list of (a, b) such that a, b, x is AP
    ap_pairs = [[] for _ in range(N + 1)]
    for x in range(3, N + 1):
        for d in range(1, x // 2 + 1):
            a = x - 2 * d
            b = x - d
            if a >= 1:
                ap_pairs[x].append((a, b))
                
    # Greedy initialization to get a good initial lower bound
    mask = 0
    in_S = [False] * (N + 1)
    S_list = []
    
    for i in range(1, N + 1):
        valid = True
        for a, b in ap_pairs[i]:
            if in_S[a] and in_S[b]:
                valid = False
                break
        if valid:
            in_S[i] = True
            S_list.append(i)
            mask |= (1 << (i - 1))
            
    # Calculate initial best score (float)
    best_score_float = sum(weights_float[x] for x in S_list)
    best_mask = mask
    best_set = list(S_list)
    
    # Helper to check validity
    def is_valid_add(x, mask):
        for a, b in ap_pairs[x]:
            if (mask & (1 << (a - 1))) and (mask & (1 << (b - 1))):
                return False
        return True

    def dfs(idx, current_sum, mask):
        nonlocal best_score_float, best_mask, best_set
        
        if time.time() > deadline:
            return
            
        # Pruning
        if current_sum + suffix_float[idx] - 1e-12 <= best_score_float:
            return
            
        if idx > N:
            if current_sum > best_score_float:
                best_score_float = current_sum
                best_mask = mask
                s = []
                for i in range(1, N + 1):
                    if mask & (1 << (i - 1)):
                        s.append(i)
                best_set = s
            return
            
        x = idx
        
        # Option 1: Include x
        if is_valid_add(x, mask):
            new_mask = mask | (1 << (x - 1))
            dfs(idx + 1, current_sum + weights_float[x], new_mask)
            
        # Option 2: Exclude x
        dfs(idx + 1, current_sum, mask)

    dfs(1, 0.0, 0)
    
    # Local Search Polish
    # We take the best_set found by DFS and try to improve it via 1-swaps
    
    current_set = list(best_set)
    current_set_set = set(current_set)
    current_score = best_score_float
    
    # Precompute a list of all elements 1..N for iteration
    all_elements = list(range(1, N + 1))
    
    improved = True
    # Limit local search time to 5 seconds
    ls_deadline = start_time + 50.0 
    
    while improved and time.time() < ls_deadline:
        improved = False
        # Iterate over elements in current_set to remove
        for x in current_set:
            if time.time() > ls_deadline:
                break
            # Create a copy of the set without x for checking
            # We don't need to copy the list, just use a set difference for checks
            # But we need to check validity of adding y to (current_set - {x})
            
            # Optimization: If x is large, its weight is small.
            # Swapping x for y > x might not improve if weights are small.
            # But y < x might be better.
            
            for y in all_elements:
                if y == x or y in current_set_set:
                    continue
                    
                # Check if y can be added to current_set - {x}
                # This is valid if no pair (a, b) in ap_pairs[y] is in current_set - {x}
                valid = True
                for a, b in ap_pairs[y]:
                    # a and b must be in current_set, and neither is x
                    if (a in current_set_set) and (b in current_set_set):
                        # If x is one of them, it's removed, so the pair is broken
                        if x == a or x == b:
                            continue
                        else:
                            valid = False
                            break
                
                if valid:
                    new_score = current_score - weights_float[x] + weights_float[y]
                    if new_score > current_score + 1e-12:
                        # Perform swap
                        current_set.remove(x)
                        current_set.append(y)
                        current_set_set.remove(x)
                        current_set_set.add(y)
                        current_score = new_score
                        improved = True
                        break # Restart the outer loop
            if improved:
                break
                
    # Update best
    if current_score > best_score_float:
        best_score_float = current_score
        best_set = current_set

    return best_set
