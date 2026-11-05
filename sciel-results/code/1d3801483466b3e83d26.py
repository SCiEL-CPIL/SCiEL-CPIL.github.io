import time
import random

def construct():
    n = 512
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    best_set = set()
    best_size = 0
    
    max_d = n // 3
    
    def can_add(v, in_set):
        # Check if v can be added to the set represented by in_set
        # without creating a 4-AP.
        for d in range(1, max_d + 1):
            # Pattern 1: v, v+d, v+2d, v+3d
            x3 = v + 3*d
            if x3 <= n:
                x1 = v + d
                x2 = v + 2*d
                if in_set[x1] and in_set[x2] and in_set[x3]:
                    return False
            else:
                break
                
            # Pattern 4: v-3d, v-2d, v-d, v
            x1 = v - 3*d
            if x1 >= 1:
                x2 = v - 2*d
                x3 = v - d
                if in_set[x1] and in_set[x2] and in_set[x3]:
                    return False
            else:
                break
                
            # Pattern 2: v-d, v, v+d, v+2d
            x1 = v - d
            x3 = v + 2*d
            if x1 >= 1 and x3 <= n:
                x2 = v + d
                if in_set[x1] and in_set[x2] and in_set[x3]:
                    return False
            
            # Pattern 3: v-2d, v-d, v, v+d
            x1 = v - 2*d
            x3 = v + d
            if x1 >= 1 and x3 <= n:
                x2 = v - d
                if in_set[x1] and in_set[x2] and in_set[x3]:
                    return False
        
        return True

    def greedy_init(random_order=False):
        in_set = [False] * (n + 1)
        S = set()
        elements = list(range(1, n + 1))
        if random_order:
            random.shuffle(elements)
        for x in elements:
            if can_add(x, in_set):
                in_set[x] = True
                S.add(x)
        return S, in_set

    def local_search(S, in_set):
        V = [i for i in range(1, n + 1) if not in_set[i]]
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 1. Try to add
            random.shuffle(V)
            for v in V:
                if can_add(v, in_set):
                    in_set[v] = True
                    S.add(v)
                    V.remove(v)
                    improved = True
                    break
            
            if improved:
                continue
            
            # 2. Try swaps
            S_list = list(S)
            random.shuffle(S_list)
            
            for u in S_list:
                if time.time() > deadline:
                    break
                if u not in S:
                    continue # Safety check
                
                # Remove u
                in_set[u] = False
                S.remove(u)
                
                found_swap = False
                for v in V:
                    if can_add(v, in_set):
                        in_set[v] = True
                        S.add(v)
                        V.remove(v)
                        found_swap = True
                        break
                        
                if not found_swap:
                    # Revert
                    in_set[u] = True
                    S.add(u)
                
                # We don't set improved=True here as size didn't change.
                # But we might have enabled an add.
                # The outer while loop will check for adds next iteration.
                
        return S, in_set

    # Main loop: Multiple restarts with randomized greedy
    for restart in range(200):
        if time.time() > deadline:
            break
            
        # Alternate between ordered and randomized greedy
        use_random = (restart % 2 == 1)
        S, in_set = greedy_init(use_random)
        S, in_set = local_search(S, in_set)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
            
    return sorted(list(best_set))
