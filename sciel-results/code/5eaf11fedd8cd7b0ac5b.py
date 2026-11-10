import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    k = 2
    
    # Precompute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, k, m))
    forbidden.discard(0)
    
    # Precompute conflict matrix or just use the set for O(1) lookup
    # conflict[i][j] = 1 if (i-j)%m in forbidden
    # Since m=205, a 205x205 matrix is fine (42k entries)
    conflict = np.zeros((m, m), dtype=bool)
    for i in range(m):
        for j in range(m):
            if i != j:
                d = (i - j) % m
                if d in forbidden:
                    conflict[i][j] = True
                    conflict[j][i] = True # Symmetric

    def is_valid_addition(S_set, v):
        # Check if v can be added to S_set using numpy for speed
        if not S_set:
            return True
        S_arr = np.fromiter(S_set, dtype=np.int64, count=len(S_set))
        return not np.any(conflict[v][S_arr])

    def greedy_init(random_order=True):
        S = []
        S_set = set()
        candidates = list(range(m))
        if random_order:
            random.shuffle(candidates)
        
        for v in candidates:
            if is_valid_addition(S_set, v):
                S.append(v)
                S_set.add(v)
        return S, S_set

    def local_search(S, S_set, max_steps=1000):
        # Try to improve S by adding vertices, 1-swaps, or 2-swaps
        best_S = list(S)
        best_S_set = set(S)
        best_size = len(S)
        
        steps = 0
        while steps < max_steps:
            steps += 1
            
            # 1. Try to add any vertex
            added = False
            candidates = [v for v in range(m) if v not in S_set]
            random.shuffle(candidates)
            
            for v in candidates:
                if is_valid_addition(S_set, v):
                    S.append(v)
                    S_set.add(v)
                    added = True
                    break
                    
            if added:
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = list(S)
                    best_S_set = set(S)
                continue
            
            # 2. No vertex can be added. Try 1-swaps (remove u, add v)
            swap_made = False
            u_candidates = list(S)
            random.shuffle(u_candidates)
            
            for u in u_candidates:
                S_set.remove(u)
                S.remove(u)
                
                candidates = [v for v in range(m) if v not in S_set]
                random.shuffle(candidates)
                
                for v in candidates:
                    if is_valid_addition(S_set, v):
                        S.append(v)
                        S_set.add(v)
                        swap_made = True
                        break
                
                if swap_made:
                    break
                else:
                    S.append(u)
                    S_set.add(u)
                    continue

            if swap_made:
                continue
            
            # 3. Try 2-swaps: remove u1, u2 from S, add v1, v2 not in S
            two_swap_made = False
            if len(S) >= 2:
                u_pairs = [(u1, u2) for i1, u1 in enumerate(S) for u2 in S[i1+1:]]
                random.shuffle(u_pairs)
                
                for u1, u2 in u_pairs:
                    S_set.discard(u1)
                    S_set.discard(u2)
                    S.remove(u1)
                    S.remove(u2)
                    
                    # Try to add two vertices
                    candidates = [v for v in range(m) if v not in S_set]
                    random.shuffle(candidates)
                    
                    added_v = []
                    for v in candidates:
                        if is_valid_addition(S_set, v):
                            S.append(v)
                            S_set.add(v)
                            added_v.append(v)
                            if len(added_v) == 2:
                                break
                    
                    if len(added_v) == 2:
                        two_swap_made = True
                        break
                    else:
                        # Revert partial additions
                        for v in added_v:
                            S_set.discard(v)
                            S.remove(v)
                        S.append(u1)
                        S.append(u2)
                        S_set.add(u1)
                        S_set.add(u2)
                        continue

            if not two_swap_made:
                # Stuck completely
                return best_S, best_S_set, best_size

        return best_S, best_S_set, best_size

    start_time = time.time()
    deadline = start_time + 590 # Safe margin before 600s limit
    
    best_overall_S = []
    best_overall_size = 0
    
    # Initial greedy with multiple orders
    for order in range(5):
        S, S_set = greedy_init(random_order=(order > 0))
        if len(S) > best_overall_size:
            best_overall_size = len(S)
            best_overall_S = list(S)
    
    # Local search on best initial
    S, S_set = greedy_init(random_order=False)
    S, S_set, size = local_search(S, S_set, max_steps=2000)
    if size > best_overall_size:
        best_overall_size = size
        best_overall_S = list(S)
        
    # Multiple restarts with local search
    restart_count = 0
    while time.time() < deadline:
        restart_count += 1
        S, S_set = greedy_init(random_order=True)
        
        # Quick check if this start is promising
        if len(S) < best_overall_size - 3:
            continue
            
        S, S_set, size = local_search(S, S_set, max_steps=2000)
        
        if size > best_overall_size:
            best_overall_size = size
            best_overall_S = list(S)
        
        # If we found a large set, do additional polish
        if best_overall_size >= 15 and time.time() < deadline - 10:
            # Try more aggressive local search on the best solution
            S, S_set = list(best_overall_S), set(best_overall_S)
            S, S_set, size = local_search(S, S_set, max_steps=5000)
            if size > best_overall_size:
                best_overall_size = size
                best_overall_S = list(S)

    return np.array(best_overall_S, dtype=np.int64)
