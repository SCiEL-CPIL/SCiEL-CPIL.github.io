import time
import random

def construct():
    N = 512
    start_time = time.time()
    time_limit = 58.0  # Safety margin
    
    # 1. Precompute all 4-term arithmetic progressions
    # For each element x, precompute a bitmask of all elements that form a 4-AP with x
    # and two other elements. When adding x, we need to check if any pair in S completes an AP.
    # Actually, for speed, let's precompute for each x, the list of (y,z,w) as before,
    # but also maintain a bitmask representation of S for fast membership testing.
    
    aps_containing = [[] for _ in range(N + 1)]
    
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            # AP is (a, b, c, e)
            aps_containing[a].append((b, c, e))
            aps_containing[b].append((a, c, e))
            aps_containing[c].append((a, b, e))
            aps_containing[e].append((a, b, c))

    # Use a list as a boolean array for O(1) membership testing
    def make_bool_array(S):
        arr = [False] * (N + 1)
        for x in S:
            arr[x] = True
        return arr

    def is_valid_add(arr, x):
        """Check if adding x to set represented by arr creates a 4-AP."""
        for (y, z, w) in aps_containing[x]:
            if arr[y] and arr[z] and arr[w]:
                return False
        return True

    def is_valid_swap(arr, i, j):
        """Check if swapping i (in S) with j (not in S) keeps S valid."""
        for (y, z, w) in aps_containing[j]:
            if i == y or i == z or i == w:
                continue
            if arr[y] and arr[z] and arr[w]:
                return False
        return True

    def local_search(S):
        """Perform local search to maximize size of S."""
        S = set(S)
        arr = make_bool_array(S)
        
        while time.time() - start_time < time_limit:
            added = False
            # Try to add elements
            candidates = [x for x in range(1, N + 1) if not arr[x]]
            random.shuffle(candidates)
            
            for x in candidates:
                if is_valid_add(arr, x):
                    S.add(x)
                    arr[x] = True
                    added = True
                    break
            
            if added:
                continue
                
            # If no element could be added, try swaps
            swapped = False
            elements_in_S = [x for x in range(1, N + 1) if arr[x]]
            random.shuffle(elements_in_S)
            
            for i in elements_in_S:
                if time.time() - start_time > time_limit:
                    break
                candidates_out = [x for x in range(1, N + 1) if not arr[x]]
                random.shuffle(candidates_out)
                
                for j in candidates_out:
                    if is_valid_swap(arr, i, j):
                        S.remove(i)
                        arr[i] = False
                        S.add(j)
                        arr[j] = True
                        swapped = True
                        break
                if swapped:
                    break
            
            if not swapped:
                break
                
        return S

    # Strategy: Multiple restarts with randomization
    best_S = set()
    
    # Start 1: Greedy (sorted order)
    S_curr = set()
    arr_curr = [False] * (N + 1)
    for x in range(1, N + 1):
        if is_valid_add(arr_curr, x):
            S_curr.add(x)
            arr_curr[x] = True
    best_S = local_search(S_curr)
    
    # Start 2: Randomized Greedy
    if time.time() - start_time < time_limit - 10:
        S_curr = set()
        arr_curr = [False] * (N + 1)
        order = list(range(1, N + 1))
        random.shuffle(order)
        for x in order:
            if is_valid_add(arr_curr, x):
                S_curr.add(x)
                arr_curr[x] = True
        S_opt = local_search(S_curr)
        if len(S_opt) > len(best_S):
            best_S = S_opt

    # Start 3: Another Randomized Greedy
    if time.time() - start_time < time_limit - 10:
        S_curr = set()
        arr_curr = [False] * (N + 1)
        order = list(range(1, N + 1))
        random.shuffle(order)
        for x in order:
            if is_valid_add(arr_curr, x):
                S_curr.add(x)
                arr_curr[x] = True
        S_opt = local_search(S_curr)
        if len(S_opt) > len(best_S):
            best_S = S_opt
            
    # Start 4: Greedy with different randomization
    if time.time() - start_time < time_limit - 5:
        S_curr = set()
        arr_curr = [False] * (N + 1)
        order = list(range(1, N + 1))
        random.shuffle(order)
        for x in order:
            if is_valid_add(arr_curr, x):
                S_curr.add(x)
                arr_curr[x] = True
        S_opt = local_search(S_curr)
        if len(S_opt) > len(best_S):
            best_S = S_opt

    # Start 5: If time remains, try another
    if time.time() - start_time < time_limit - 5:
        S_curr = set()
        arr_curr = [False] * (N + 1)
        order = list(range(1, N + 1))
        random.shuffle(order)
        for x in order:
            if is_valid_add(arr_curr, x):
                S_curr.add(x)
                arr_curr[x] = True
        S_opt = local_search(S_curr)
        if len(S_opt) > len(best_S):
            best_S = S_opt

    return sorted(best_S)

# EVOLVE-BLOCK-END
