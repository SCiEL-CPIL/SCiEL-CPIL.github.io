import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    N = 511
    weights = [0.0] * (N + 1)
    for a in range(1, N + 1):
        weights[a] = 1.0 / (512 - a)
        
    conflict_masks = [0] * (N + 1)
    for a in range(1, N + 1):
        mask = 0
        for b in range(1, N + 1):
            if math.gcd(a, b) > 1:
                mask |= (1 << b)
        conflict_masks[a] = mask
        
    order = list(range(N, 0, -1))
    
    suffix_sum = [0.0] * (N + 2)
    for i in range(N, 0, -1):
        suffix_sum[i] = suffix_sum[i+1] + weights[order[i]]
        
    best_score = 0.0
    best_set = []
    
    # Greedy initialization
    greedy_set = []
    greedy_mask = 0
    greedy_score = 0.0
    for a in order:
        if (conflict_masks[a] & greedy_mask) == 0:
            greedy_set.append(a)
            greedy_mask |= (1 << a)
            greedy_score += weights[a]
            
    best_score = greedy_score
    best_set = greedy_set[:]
    
    # Local variables for speed
    cm = conflict_masks
    w = weights
    ord_list = order
    ss = suffix_sum
    n = N
    dl = deadline
    
    def dfs(idx, occupied_mask, current_score, current_set):
        nonlocal best_score, best_set
        
        if time.time() > dl:
            return
            
        # Loose bound
        if current_score + ss[idx] <= best_score:
            return
            
        # Tight bound
        bound = current_score
        for k in range(idx, n + 1):
            a = ord_list[k]
            if (cm[a] & occupied_mask) == 0:
                bound += w[a]
                
        if bound <= best_score:
            return
            
        if idx > n:
            if current_score > best_score:
                best_score = current_score
                best_set = current_set[:]
            return
            
        a = ord_list[idx]
        
        # Branch 1: Include
        if (cm[a] & occupied_mask) == 0:
            current_set.append(a)
            dfs(idx + 1, occupied_mask | (1 << a), current_score + w[a], current_set)
            current_set.pop()
            
        # Branch 2: Exclude
        dfs(idx + 1, occupied_mask, current_score, current_set)
        
    dfs(1, 0, 0.0, [])
    
    return best_set
