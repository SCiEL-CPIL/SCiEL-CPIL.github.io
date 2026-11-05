import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    n = 256
    all_elements = list(range(1, n + 1))
    
    def is_valid_set(S):
        S = list(S)
        S.sort()
        for i in range(len(S)):
            a = S[i]
            for j in range(i + 1, len(S)):
                b = S[j]
                for k in range(j, len(S)):
                    c = S[k]
                    if a < b <= c:
                        if (b + c) % a == 0:
                            return False
        return True

    def can_add(x, S_list):
        # S_list is sorted
        # Check triples involving x
        L = len(S_list)
        # 1. x is smallest: x < b <= c
        # b, c in S_list, b > x
        idx_b = 0
        while idx_b < L and S_list[idx_b] <= x:
            idx_b += 1
        for j in range(idx_b, L):
            b = S_list[j]
            for k in range(j, L):
                c = S_list[k]
                if (b + c) % x == 0:
                    return False
        
        # 2. x is middle: a < x <= c
        # a in S_list, a < x
        # c in S_list, c >= x
        idx_a_max = 0
        while idx_a_max < L and S_list[idx_a_max] < x:
            idx_a_max += 1
        idx_c_min = 0
        while idx_c_min < L and S_list[idx_c_min] < x:
            idx_c_min += 1
        
        for i in range(idx_a_max):
            a = S_list[i]
            for k in range(idx_c_min, L):
                c = S_list[k]
                if (x + c) % a == 0:
                    return False

        # 3. x is largest: a < b <= x
        # a, b in S_list, b <= x
        # a < b
        idx_b_max = 0
        while idx_b_max < L and S_list[idx_b_max] <= x:
            idx_b_max += 1
        
        for i in range(idx_b_max):
            a = S_list[i]
            # b must be > a and <= x
            start_j = i + 1
            # Find first j >= start_j such that S_list[j] > a
            # Since sorted, S_list[i] = a, so j starts at i+1
            for j in range(start_j, idx_b_max):
                b = S_list[j]
                if (b + x) % a == 0:
                    return False
        return True

    def greedy_random():
        S = []
        candidates = all_elements[:]
        random.shuffle(candidates)
        for x in candidates:
            if can_add(x, S):
                S.append(x)
                S.sort()
        return S

    def local_search(S):
        S = list(S)
        S.sort()
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try to add elements
            for x in all_elements:
                if x in S:
                    continue
                if can_add(x, S):
                    S.append(x)
                    S.sort()
                    improved = True
            # Try to swap: remove one, add one? 
            # Or remove one to allow adding two?
            # Simple swap: remove u, add v. Size same.
            # We want to increase size.
            # Try: for each u in S, remove u.
            # Then try to add any v not in S.
            # If we can add v, we are back to size |S|.
            # But maybe we can add MORE?
            # Let's try a 1-step swap that increases size:
            # Remove u. Add v. If valid, size same.
            # Then try to add w. If valid, size +1.
            # This is a 2-move.
            
            # Alternative: Random restarts are often enough for N=256.
            # Let's do multiple greedy restarts.
        
        return S

    best_S = []
    best_size = 0
    
    # Initial good seed: upper tail
    S0 = list(range(192, 257))
    if is_valid_set(S0):
        best_S = S0
        best_size = len(S0)
        
    # Run multiple random greedy searches
    num_restarts = 50
    for i in range(num_restarts):
        if time.time() > deadline - 10:
            break
        S = greedy_random()
        if len(S) > best_size:
            best_size = len(S)
            best_S = S
            
        # Try to improve S locally
        # Simple local search: try to add any element that fits
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for x in all_elements:
                if x in S:
                    continue
                if can_add(x, S):
                    S.append(x)
                    S.sort()
                    improved = True
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = list(S)
        
        # Try swaps to escape local maxima
        # This is expensive, so limit it
        if time.time() < deadline - 10:
            for _ in range(5): # 5 swap attempts
                if time.time() > deadline - 10:
                    break
                u = random.choice(S)
                S_temp = [x for x in S if x != u]
                # Try to add random elements to S_temp
                added = False
                for x in all_elements:
                    if x == u or x in S_temp:
                        continue
                    if can_add(x, S_temp):
                        S_temp.append(x)
                        S_temp.sort()
                        added = True
                        break
                if added:
                    # Now try to add more to S_temp
                    improved2 = True
                    while improved2 and time.time() < deadline - 10:
                        improved2 = False
                        for x in all_elements:
                            if x in S_temp:
                                continue
                            if can_add(x, S_temp):
                                S_temp.append(x)
                                S_temp.sort()
                                improved2 = True
                                if len(S_temp) > best_size:
                                    best_size = len(S_temp)
                                    best_S = list(S_temp)
                    if len(S_temp) > len(S):
                        S = S_temp

    return best_S
