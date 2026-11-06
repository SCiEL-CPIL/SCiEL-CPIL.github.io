import time
import random

def construct():
    start_time = time.time()
    timeout = 140.0
    
    N = 32
    R = 512
    
    best_set = list(range(1, 33))
    best_in_set = [False] * (R + 1)
    for x in best_set:
        best_in_set[x] = True
        
    def calc_score(A_list):
        mask = 0
        for x in A_list:
            mask |= (1 << x)
            
        diffs_mask = 0
        for i in range(N):
            ai = A_list[i]
            for j in range(i+1, N):
                aj = A_list[j]
                d = aj - ai
                target = aj + d
                if target <= R:
                    if (mask >> target) & 1:
                        diffs_mask |= (1 << d)
        return bin(diffs_mask).count("1")
        
    best_score = calc_score(best_set)
    
    current_set = best_set[:]
    current_in_set = best_in_set[:]
    current_score = best_score
    
    T = 1.0
    T_min = 0.01
    cooling = 0.9999
    
    while time.time() - start_time < timeout:
        idx_remove = random.randint(0, N-1)
        x = current_set[idx_remove]
        
        y = random.randint(1, R)
        
        if current_in_set[y]:
            continue
            
        new_set = current_set[:]
        new_set[idx_remove] = y
        
        new_score = calc_score(new_set)
        
        delta = new_score - current_score
        
        if delta >= 0 or random.random() < pow(2.718281828, delta / T):
            # Update current state
            current_in_set[x] = False
            current_in_set[y] = True
            current_set = new_set
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_set = current_set[:]
                # best_in_set doesn't need to be updated for scoring, only for return
                # But if we use it for restarts, we should update it.
                # We don't use best_in_set for logic, only current_in_set.
                
        T = max(T_min, T * cooling)
        
    return best_set
