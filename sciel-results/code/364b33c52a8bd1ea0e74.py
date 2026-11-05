import numpy as np
import random
import time

def is_valid_addition(S, x):
    """
    Check if x can be added to set S (list of ints) without violating constraints.
    S is assumed to be valid.
    Returns True if S U {x} is valid.
    """
    S_arr = np.array(S)
    N = len(S_arr)
    
    # 1. Check if x divides any 3-sum in S
    if N >= 3:
        pair_sums_mod_x = (S_arr[:, None] + S_arr[None, :]) % x
        for k in range(N):
            target = (-S_arr[k]) % x
            mask = np.ones((N, N), dtype=bool)
            mask[k, :] = False
            mask[:, k] = False
            mask[np.diag_indices(N)] = False
            if np.any(pair_sums_mod_x[mask] == target):
                return False

    # 2. For each a in S, check if a divides (b+c+x) for some distinct b,c in S\{a}.
    for idx_a, a in enumerate(S_arr):
        target = (-x) % a
        pair_sums_mod_a = (S_arr[:, None] + S_arr[None, :]) % a
        mask = np.ones((N, N), dtype=bool)
        mask[idx_a, :] = False
        mask[:, idx_a] = False
        mask[np.diag_indices(N)] = False
        if np.any(pair_sums_mod_a[mask] == target):
            return False
            
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55
    rng = random.Random(42)
    
    # Initial greedy: add largest first
    S = []
    for x in range(192, 0, -1):
        if time.time() > deadline:
            break
        if is_valid_addition(S, x):
            S.append(x)
    
    best_size = len(S)
    best_set = set(S)
    
    all_numbers = set(range(1, 193))
    
    def local_search(S, rng, max_iters=1000):
        S = list(S)
        cur_size = len(S)
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            improved = False
            
            # Try adding
            candidates = list(all_numbers - set(S))
            rng.shuffle(candidates)
            for x in candidates[:20]:
                if is_valid_addition(S, x):
                    S.append(x)
                    cur_size += 1
                    improved = True
                    break
            
            if improved:
                continue
                
            # Try swap
            if S:
                elems = list(S)
                rng.shuffle(elems)
                for rem in elems[:10]:
                    candidates = list(all_numbers - set(S))
                    rng.shuffle(candidates)
                    for add in candidates[:20]:
                        S_temp = [x for x in S if x != rem]
                        if is_valid_addition(S_temp, add):
                            S = S_temp + [add]
                            improved = True
                            break
                    if improved:
                        break
            
            if not improved:
                break
        return S

    # Run local search multiple times
    for i in range(200):
        if time.time() > deadline:
            break
        
        if i == 0:
            current = list(best_set)
        else:
            current = list(best_set)
            rng.shuffle(current)
            for _ in range(rng.randint(1, 5)):
                if current:
                    current.pop(rng.randint(0, len(current)-1))
            
        result = local_search(current, rng, max_iters=200)
        
        if len(result) > best_size:
            best_size = len(result)
            best_set = set(result)
            
    return list(best_set)
