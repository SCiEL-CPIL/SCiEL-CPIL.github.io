import time
import random

def construct():
    N = 512
    start_time = time.time()
    timeout = 85.0
    
    # 1. Precomputation
    aps = []
    for d in range(1, (N - 1) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            aps.append((a, a+d, a+2*d, a+3*d))
            
    num_aps = len(aps)
    
    elem_to_aps = [[] for _ in range(N + 1)]
    for i, ap in enumerate(aps):
        for x in ap:
            elem_to_aps[x].append(i)
            
    def is_valid_add(in_set, x):
        for ap_idx in elem_to_aps[x]:
            a, b, c, d = aps[ap_idx]
            # Check if other 3 are in set
            if a != x and in_set[a] and (b != x and in_set[b]) and (c != x and in_set[c]) and (d != x and in_set[d]):
                return False
        return True

    def greedy_remove():
        in_set = [False] * (N + 1)
        for i in range(1, N + 1):
            in_set[i] = True
            
        is_active = [True] * num_aps
        active_ap_list = list(range(num_aps))
        
        while active_ap_list:
            pos = random.randint(0, len(active_ap_list) - 1)
            ap_idx = active_ap_list[pos]
            
            if not is_active[ap_idx]:
                last_idx = len(active_ap_list) - 1
                active_ap_list[pos] = active_ap_list[last_idx]
                active_ap_list.pop()
                continue
                
            a, b, c, d = aps[ap_idx]
            candidates = [a, b, c, d]
            
            best_candidate = -1
            max_degree = -1
            
            for x in candidates:
                degree = 0
                for other_ap_idx in elem_to_aps[x]:
                    if is_active[other_ap_idx]:
                        degree += 1
                if degree > max_degree:
                    max_degree = degree
                    best_candidate = x
                    
            in_set[best_candidate] = False
            
            for ap_idx_to_check in elem_to_aps[best_candidate]:
                if is_active[ap_idx_to_check]:
                    is_active[ap_idx_to_check] = False
                    
        return [i for i in range(1, N+1) if in_set[i]]

    def add_back(S):
        S_set = set(S)
        in_set = [False] * (N + 1)
        for i in S:
            in_set[i] = True
            
        candidates = [i for i in range(1, N+1) if i not in S_set]
        random.shuffle(candidates)
        
        for x in candidates:
            if is_valid_add(in_set, x):
                in_set[x] = True
                S_set.add(x)
                
        return sorted(S_set)

    best_S = []
    best_size = 0
    
    while time.time() - start_time < timeout:
        S = greedy_remove()
        S = add_back(S)
        if len(S) > best_size:
            best_size = len(S)
            best_S = S
            
    return best_S
