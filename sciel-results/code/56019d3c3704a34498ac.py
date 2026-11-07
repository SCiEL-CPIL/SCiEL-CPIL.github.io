import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 85.0
    
    def time_left():
        return deadline - time.time()
    
    N = 128
    
    def is_bad_triple(S, a, b, c):
        s_ab = a + b
        s_ac = a + c
        s_bc = b + c
        if s_ab > N or s_ac > N or s_bc > N:
            return False
        return s_ab in S and s_ac in S and s_bc in S
    
    def check_add(S, x):
        """Check if adding x to S creates any bad triple."""
        for a, b in combinations(S, 2):
            s_ab = a + b
            s_ax = a + x
            s_bx = b + x
            if s_ab <= N and s_ax <= N and s_bx <= N:
                if s_ab in S and s_ax in S and s_bx in S:
                    return False
        return True
    
    def is_valid(S):
        for a, b, c in combinations(S, 3):
            if is_bad_triple(S, a, b, c):
                return False
        return True
    
    # Phase 1: Greedy - largest first
    S = set()
    for x in range(N, 0, -1):
        if time_left() < 5.0:
            break
        if check_add(S, x):
            S.add(x)
    
    best_S = set(S)
    best_size = len(S)
    
    # Phase 2: Local search
    def local_search(S, max_rounds=300):
        S = set(S)
        for _ in range(max_rounds):
            if time_left() < 5.0:
                break
            improved = False
            
            # Try all single additions
            for x in range(1, N + 1):
                if x in S:
                    continue
                if check_add(S, x):
                    S.add(x)
                    improved = True
                    break
            if improved:
                continue
            
            # Try swaps: remove one, add one
            for rem in list(S):
                if time_left() < 5.0:
                    return S
                S_temp = S - {rem}
                for x in range(1, N + 1):
                    if x in S_temp:
                        continue
                    if check_add(S_temp, x):
                        S = S_temp | {x}
                        improved = True
                        break
                if improved:
                    break
            if improved:
                continue
            
            # Try remove 2, add 2
            if time_left() > 10.0:
                S_list = list(S)
                found = False
                for i in range(len(S_list)):
                    if found:
                        break
                    for j in range(i + 1, len(S_list)):
                        if time_left() < 5.0:
                            return S
                        S_no2 = S - {S_list[i], S_list[j]}
                        for x in range(1, N + 1):
                            if found:
                                break
                            if x in S_no2:
                                continue
                            for y in range(x + 1, N + 1):
                                if y in S_no2:
                                    continue
                                S_new = S_no2 | {x, y}
                                if len(S_new) > len(S) and is_valid(S_new):
                                    S = S_new
                                    found = True
                                    break
                        if found:
                            break
                if found:
                    continue
            
            if not improved:
                break
        
        return S
    
    S = local_search(S)
    if len(S) > best_size:
        best_S = set(S)
        best_size = len(S)
    
    # Final polish
    if time_left() > 3.0:
        for x in range(1, N + 1):
            if x in best_S:
                continue
            if check_add(best_S, x):
                best_S.add(x)
    
    if not is_valid(best_S):
        best_S = set(range(N // 2 + 1, N + 1))
    
    return sorted(list(best_S))
