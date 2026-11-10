"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time

# EVOLVE-BLOCK-START
def construct():
    N = 512
    start_time = time.time()
    time_limit = 58.0  # Safety margin before 60s kill
    
    # Precompute all 4-term arithmetic progressions containing each element
    aps_containing = [[] for _ in range(N + 1)]
    
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            aps_containing[a].append((b, c, e))
            aps_containing[b].append((a, c, e))
            aps_containing[c].append((a, b, e))
            aps_containing[e].append((a, b, c))

    def make_bool_array(S):
        arr = [False] * (N + 1)
        for x in S:
            arr[x] = True
        return arr

    def is_valid_add(arr, x):
        for (y, z, w) in aps_containing[x]:
            if arr[y] and arr[z] and arr[w]:
                return False
        return True

    def is_valid_swap(arr, i, j):
        # Check if removing i and adding j keeps S valid
        # j must not form a 4-AP with any three elements in S\{i}
        for (y, z, w) in aps_containing[j]:
            if i == y or i == z or i == w:
                continue
            if arr[y] and arr[z] and arr[w]:
                return False
        return True

    def is_valid_double_swap(arr, i1, i2, j1, j2):
        """Check if removing i1,i2 and adding j1,j2 keeps S valid."""
        arr[i1] = False
        arr[i2] = False
        ok1 = True
        for (y, z, w) in aps_containing[j1]:
            if arr[y] and arr[z] and arr[w]:
                ok1 = False
                break
        if ok1:
            ok2 = True
            for (y, z, w) in aps_containing[j2]:
                if arr[y] and arr[z] and arr[w]:
                    ok2 = False
                    break
            valid = ok2
        else:
            valid = False
        arr[i1] = True
        arr[i2] = True
        return valid

    def local_search(S, time_budget):
        S = set(S)
        arr = make_bool_array(S)
        
        while time.time() - start_time < time_limit - time_budget:
            # Phase 1: Try to add elements
            candidates = [x for x in range(1, N + 1) if not arr[x]]
            random.shuffle(candidates)
            
            added = False
            for x in candidates:
                if is_valid_add(arr, x):
                    S.add(x)
                    arr[x] = True
                    added = True
                    break
            
            if added:
                continue
            
            # Phase 2: 1-exchange swaps
            elements_in_S = [x for x in range(1, N + 1) if arr[x]]
            random.shuffle(elements_in_S)
            candidates_out = [x for x in range(1, N + 1) if not arr[x]]
            random.shuffle(candidates_out)
            
            swapped = False
            for i in elements_in_S:
                if time.time() - start_time > time_limit - time_budget:
                    break
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
            
            if swapped:
                continue
            
            # Phase 3: 2-exchange moves
            elements_in_S2 = [x for x in range(1, N + 1) if arr[x]]
            random.shuffle(elements_in_S2)
            candidates_out2 = [x for x in range(1, N + 1) if not arr[x]]
            random.shuffle(candidates_out2)
            
            max_i1 = min(len(elements_in_S2), 30)
            max_j1 = min(len(candidates_out2), 30)
            
            double_swapped = False
            for idx_i1 in range(max_i1):
                if time.time() - start_time > time_limit - time_budget:
                    break
                i1 = elements_in_S2[idx_i1]
                for idx_i2 in range(idx_i1 + 1, min(idx_i1 + 8, len(elements_in_S2))):
                    i2 = elements_in_S2[idx_i2]
                    for idx_j1 in range(max_j1):
                        if time.time() - start_time > time_limit - time_budget:
                            break
                        j1 = candidates_out2[idx_j1]
                        for idx_j2 in range(idx_j1 + 1, min(idx_j1 + 8, len(candidates_out2))):
                            j2 = candidates_out2[idx_j2]
                            if is_valid_double_swap(arr, i1, i2, j1, j2):
                                S.discard(i1)
                                S.discard(i2)
                                arr[i1] = False
                                arr[i2] = False
                                S.add(j1)
                                S.add(j2)
                                arr[j1] = True
                                arr[j2] = True
                                double_swapped = True
                                break
                        if double_swapped:
                            break
                    if double_swapped:
                        break
                if double_swapped:
                    break
            
            if not double_swapped:
                break
                
        return S

    best_S = set()
    best_size = 0
    
    # Start 1: Greedy (sorted order)
    S_curr = set()
    arr_curr = [False] * (N + 1)
    for x in range(1, N + 1):
        if is_valid_add(arr_curr, x):
            S_curr.add(x)
            arr_curr[x] = True
    S_opt = local_search(S_curr, 5.0)
    if len(S_opt) > best_size:
        best_size = len(S_opt)
        best_S = set(S_opt)
    
    # Multiple randomized restarts
    num_restarts = 0
    while time.time() - start_time < time_limit - 3.0:
        S_curr = set()
        arr_curr = [False] * (N + 1)
        order = list(range(1, N + 1))
        random.shuffle(order)
        for x in order:
            if is_valid_add(arr_curr, x):
                S_curr.add(x)
                arr_curr[x] = True
        S_opt = local_search(S_curr, 2.0)
        if len(S_opt) > best_size:
            best_size = len(S_opt)
            best_S = set(S_opt)
        num_restarts += 1

    return sorted(best_S)
# EVOLVE-BLOCK-END
